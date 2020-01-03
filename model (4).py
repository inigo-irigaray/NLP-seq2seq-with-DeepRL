import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import utils

#Hyperparams defining the size of the hidden state used in the encoder and decoder RNNs
HIDDEN_STATE_SIZE = 512
EMBEDDING_DIM = 50

class PhraseModel(nn.Module):
  def __init__(self, emb_size, dict_size, hid_size):
    super(PhraseModel, self).__init__()

    self.emb = nn.Embedding(dict_size, emb_size)
    self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=1, batch_first=True)

    self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=1, batch_first=True)

    self.output = nn.Sequential(nn.Linear(hid_size, dict_size))

  def encode(self, x):
    """Encodes the input sequence and returns the hidden state from the last step of the encoder RNN."""
    _, hid = self.encoder(x) #All RNN classes output a tuple of 2 objects: the output of the RNN first and the hidden state from the last item in
    return hid               #the input sequence second. We're only interested in the hidden state

  def get_encoded_item(self, encoded, index):
    """Gives access to the hidden state of the individual components of the input batch. Since encode() encodes the whole batch of sequences in 
    one call, but decoding is performed for every batch sequence individually, this method becomes necessary."""

    #for vanilla RNN and GRU, since they have a hidden state represented as a single tensor
    ##return encoded[:, index:index+1]

    #for LSTM, since it has a hidden state represented as a tuple of two tensors: the cell state and the hidden state
    return encoded[0][:, index:index+1].contiguous(), encoded[1][:, index:index+1].contiguous()

  def decode_teacher(self, hid, input_seq):
    """Teacher-forcing mode: Applies the decoder RNN to the reference sequence. The input of every step is known in advance, the only dependency the RNN
    has between steps is its hidden state, allowing RNN transformations very efficiently w/o transferring data to and from the GPU and implemented in
    the underlying CuDNN library."""
    #method assumes batch of size=1
    out, _ = self.decoder(input_seq, hid)
    out = self.output(out.data)
    return out

  def decode_one(self, hid, input_x):
    """Performs one single decoding step for one example. It passes the hidden state for the decoder and input the tensor with the embeddings vector
    for the input token. The result of the decoder is passed to the output net to obtain the logits for every item in the dictionary. It outputs those
    logits and the new hidden state returned by the decoder."""
    out, new_hid = self.decoder(input_x.unsqueeze(0), hid)
    out = self.output(out)
    return out.squeeze(dim=0), new_hid

  def decode_chain_argmax(self, hid, begin_emb, seq_len, stop_at_token=None):
    """
    Decodes sequence by feeding token into net again and acts according to probabilities.
    
    Input:
      -hid: hidden state returned by the encoder of the input sequence.
      -begin_emb: embedding vector for the #BEG token used to start decoding.
      -seq_len: maximum length of the decoded sequence. Could be shorter if decoder returns #END token.
      -stop_at_token: optional token ID (normally #END token) that stops the decoding process.
    
    Output:
      -logits tensor: used for training, as we need output tensors to calculate the loss.
      -list of token IDs produced: value passed tot he quality metric function(BLEU score in this case).
      
      """
    res_logits = []
    res_tokens = []
    cur_emb = begin_emb

    for _ in range(seq_len):
      out_logits, hid = self.decode_one(hid, cur_emb)
      out_token_v = torch.max(out_logits, dim=1)[1] #uses argmax to go from logits to the decoded token ID
      out_token = out_token_v.data.cpu().numpy()[0]

      cur_emb = self.emb(out_token_v) #obtains embeddings for the decoded token to iterate over

      res_logits.append(out_logits)
      res_tokens.append(out_token)
      if stop_at_token is not None and out_token == stop_at_token:
        break
      
    return torch.cat(res_logits), res_tokens

  def decode_chain_sampling(self, hid, begin_emb, seq_len, stop_at_token=None, device='cpu'):
    """Almost the same as decode_chain_argmax(), but instead of using argmax, it performs the random sampling from the returned probability distribution."""
    res_logits = []
    res_actions = []
    cur_emb = begin_emb

    for _ in range(seq_len):
      out_logits, hid = self.decode_one(hid, cur_emb)
      out_probs_v = F.softmax(out_logits, dim=1)
      out_probs = out_probs_v.data.cpu().numpy()[0]
      action = int(np.random.choice(out_probs.shape[0], p=out_probs))
      action_v = torch.LongTensor([action]).to(device)
      cur_emb = self.emb(action_v)

      res_logits.append(out_logits)
      res_actions.append(action)
      if stop_at_token is not None and action == stop_at_token:
        break

    return torch.cat(res_logits), res_actions

def pack_batch_no_out(batch, embeddings, device='cpu'):
  assert isinstance(batch, list)
  #sort descending (CuDNN requirements)
  batch.sort(key=lambda s: len(s[0]), reverse=True)
  input_idx, output_idx = zip(*batch)
  #create padded matrix of inputs
  lens = list(map(len, input_idx))
  input_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)
  for idx, x in enumerate(input_idx):
    input_mat[idx, :len(x)] = x
  input_v = torch.tensor(input_mat).to(device)
  input_seq = rnn_utils.pack_padded_sequence(input_v, lens, batch_first=True)
  #lookup embeddings
  r = embeddings(input_seq.data)
  emb_input_seq = rnn_utils.PackedSequence(r, input_seq.batch_sizes)
  return emb_input_seq, input_idx, output_idx

def pack_input(input_data, embeddings, device='cpu'):
  input_v = torch.LongTensor([input_data]).to(device)
  r = embeddings(input_v)
  return rnn_utils.pack_padded_sequence(r, [len(input_data)], batch_first=True)

def pack_batch(batch, embeddings, device='cpu'):
  emb_input_seq, input_idx, output_idx = pack_batch_no_out(batch, embeddings, device)
  output_seq_list = []
  for out in output_idx:
    output_seq_list.append(pack_input(out[:-1], embeddings, device))
  return emb_input_seq, output_seq_list, input_idx, output_idx

def seq_bleu(model_out, ref_seq):
  model_seq = torch.max(model_out.data, dim=1)[1]
  model_seq = model_seq.cpu().numpy()
  return utils.calc_bleu(model_seq, ref_seq)