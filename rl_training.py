import os
import random
import logging
import numpy as np
from tensorboardX import SummaryWriter

import data_p
import model
import utils

import torch
import torch.optim as optim
import torch.nn.functional as F

SAVES_DIR = "saves"

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_EPOCHS = 10000

log = logging.getLogger("train")



def run_test(test_data, net, end_token, device='cpu'):
  """Runs every epoch to calculate the BLEU score on the test dataset. Test data is now group by the first phrase, with a shape
  (first_phrase, [second_phrases]). BLEU score now calculated by a function accepting several reference sequences & returns the best score for them."""
  bleu_sum = 0.0
  bleu_count = 0
  for p1, p2 in test_data:
    input_seq = model.pack_input(p1, net.emb, device)
    enc = net.encode(input_seq)
    _, tokens = net.decode_chain_argmax(enc, input_seq.data[0:1], seq_len=data_p.MAX_TOKENS, stop_at_token=end_token)
    ref_indices = [indices[1:] for indices in p2]
    bleu_sum += utils.calc_bleu_many(tokens, ref_indices)
    bleu_count += 1
  return bleu_sum / bleu_count



if __name__ == "__main__":
  logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
  
  genre = "comedy"
  name = "inigo"
  load = "/epoch_040_0.579_0.113_.dat"
  samples = 4
  disable_skip = False
  device = "cuda"

  saves_path = os.path.join(SAVES_DIR, name)
  os.makedirs(saves_path, exist_ok=True)
  #loads the training data
  phrase_pairs, emb_dict = data_p.load_data(genre_filter=genre)
  log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
  data_p.save_emb_dict(saves_path, emb_dict)
  end_token = emb_dict[data_p.END_TOKEN]
  train_data = data_p.encode_phrase_pairs(phrase_pairs, emb_dict)
  rand = np.random.RandomState(data_p.SHUFFLE_SEED)
  rand.shuffle(train_data)
  train_data, test_data = data_p.split_train_test(train_data)
  log.info("Training data converted, got %d samples", len(train_data))
  train_data = data_p.group_train_data(train_data) #groups the training data by the first phrase
  test_data = data_p.group_train_data(test_data) #groups the test data by the first phrase
  log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))
  #creates model and loads its weights from the given file (from the previous cross-entropy training in our case)
  rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

  net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE).to(device)
  log.info("Model: %s", net)

  writer = SummaryWriter(comment="-" + name)
  net.load_state_dict(torch.load(load))
  log.info("Model loaded from %s, continue training in RL mode...")
  #create tensor with #BEG token that will be used to look up the embeddings and pass the result to the decoder
  beg_token = torch.LongTensor([emb_dict[data_p.BEGIN_TOKEN]]).to(device)

  with utils.Tracker(writer, batch_size=100) as tracker:
    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    batch_idx = 0
    best_bleu = None
    for epoch in range(MAX_EPOCHS):
      random.shuffle(train_data)
      dial_shown = False
      total_samples = 0
      skipped_samples = 0
      bleus_argmax = []
      bleus_sample = []
      #pack the batch, encode all first sequences, then declare lists that will be populated during individual decoding of batch entries
      for batch in data_p.iterate_batches(train_data, BATCH_SIZE):
        batch_idx += 1
        optimiser.zero_grad()
        input_seq, input_batch, output_batch = model.pack_batch_no_out(batch, net.emb, device)
        enc = net.encode(input_seq)

        net_policies = []
        net_actions = []
        net_advantages = []
        beg_embeddings = net.emb(beg_token)
        #process individual entries: first strip the #BEG token from reference seqs and obtain individual entry for encoded batch
        for idx, inp_idx in enumerate(input_batch):
          total_samples += 1
          ref_indices = [indices[1:] for indices in output_batch[idx]]
          item_enc = net.get_encoded_item(enc, idx)
          #then decode the batch entry in argmax mode and calculate its BLEU score, which will be used as a baseline in REINFORCE
          r_argmax, actions = net.decode_chain_argmax(item_enc, beg_embeddings, data_p.MAX_TOKENS, stop_at_token=end_token)
          argmax_bleu = utils.calc_bleu_many(actions, ref_indices)
          bleus_argmax.append(argmax_bleu)

          if not disable_skip and argmax_bleu > 0.99:
            skipped_samples += 1
            continue
          #the following lines are executed once every epoch and only for the sake of getting some information during training
          if not dial_shown:
            log.info("Input: %s", utils.untokenize(data_p.decode_words(inp_idx, rev_emb_dict)))
            ref_words = [utils.untokenize(data_p.decode_words(ref, rev_emb_dict)) for ref in ref_indices]
            log.info("Refer: %s", " ~~|~~ ".join(ref_words))
            log.info("Argmax: %s, bleu=%.4f", utils.untokenize(data_p.decode_words(actions, rev_emb_dict)), argmax_bleu)
          
          for _ in range(samples):
            r_sample, actions = net.decode_chain_sampling(item_enc, beg_embeddings, data_p.MAX_TOKENS, stop_at_token=end_token, device=device)
            sample_bleu = utils.calc_bleu_many(actions, ref_indices)

            if not dial_shown:
            
              log.info("Sample: %s, bleu=%.4f", utils.untokenize(data_p.decode_words(actions, rev_emb_dict)), sample_bleu)
              net_policies.append(r_sample)
              net_actions.extend(actions)
              net_advantages.extend([sample_bleu - argmax_bleu] * len(actions))
              bleus_sample.append(sample_bleu)

          dial_shown = True
        
        if not net_policies:
          continue
        
        policies_v = torch.cat(net_policies)
        actions_t = torch.LongTensor(net_actions).to(device)
        adv_v = torch.FloatTensor(net_advantages).to(device)

        log_prob_v = F.log_softmax(policies_v, dim=1)
        log_prob_actions_v = adv_v * log_prob_v[range(len(net_actions)), actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        loss_v = loss_policy_v
        loss_v.backward()
        optimiser.step()

        tracker.track("advantage", adv_v, batch_idx)
        tracker.track("loss_policy", loss_policy_v, batch_idx)

      bleu_test = run_test(test_data, net, end_token, device)
      bleu = np.mean(bleus_argmax)
      writer.add_scalar("bleu_test", bleu_test, batch_idx)
      writer.add_scalar("bleu_argmax", bleu, batch_idx)
      writer.add_scalar("bleu_sample", np.mean(bleus_sample), batch_idx)
      writer.add_scalar("skipped_samples", skipped_samples / total_samples, batch_idx)
      writer.add_scalar("epoch", batch_idx, epoch)
      log.info("Epoch %d, test BLEU: %.3f", epoch, bleu_test)

      if best_bleu is None or best_bleu < bleu_test:
        best_bleu = bleu_test
        log.info("Best bleu updated: %.4f", bleu_test)
        torch.save(net.state_dict(), os.path.join(saves_path, "bleu_%.3f_%02d.dat" % (bleu_test, epoch)))
      if epoch % 10 == 0:
        torch.save(net.state_dict(), os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" % (epoch, bleu, bleu_test)))
