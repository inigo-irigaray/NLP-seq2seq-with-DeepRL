import string
import collections
import numpy as np
import torch
from nltk.translate import bleu_score
from nltk.tokenize import TweetTokenizer

def calc_bleu_many(cand_seq, ref_seq):
  """Calculates BLEU score when there are several reference sequences to compare against the candidate."""
  sf = bleu_score.SmoothingFunction()
  return bleu_score.sentence_bleu(ref_seq, cand_seq, smoothing_function=sf.method1, weights=(0.5, 0.5))

def calc_bleu(cand_seq, ref_seq):
  """Calculates BLEU score when there is only one reference sequence and one candidate sequence."""
  return calc_bleu_many(cand_seq, [ref_seq])

def tokenize(s):
  """Converts sentences into tokens."""
  return TweetTokenizer(preserve_case=False).tokenize(s)

def untokenize(words):
  """Converts a list of tokens back into a string."""
  return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in words]).strip()

class Tracker:
    """
    TensorBoard value tracker: allows to batch fixed amount of historical values and write their mean into TB
    Designed and tested with pytorch-tensorboard in mind
    """
    def __init__(self, writer, batch_size):
        """
        :param writer: writer with close() and add_scalar() methods
        :param batch_size: integer size of batch to track
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = collections.defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()
