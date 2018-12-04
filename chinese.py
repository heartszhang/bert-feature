"""Extract pre-computed feature vectors from a PyTorch BERT model.
Some codes are copied from bert-pretrained-pytorch/examples."""
import torch as t
from sys import stdin, stderr, stdout
from typing import Iterator, List
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from fire import Fire


def log(*args, **kwargs):
  print(*args, file = stderr, **kwargs)


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def convert_example_to_features(uid: int, text_a: str, text_b: str,
                                seq_length: int,
                                tokenizer: BertTokenizer, ) -> InputFeatures:
  tokens_a = tokenizer.tokenize(text_a)
  tokens_b = text_b and tokenizer.tokenize(text_b) or []

  _truncate_seq(tokens_a, tokens_b, seq_length)

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0   0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
  input_type_ids = [0] * (len(tokens_a) + 2)

  if tokens_b:
    tokens = tokens + tokens_b + ["[SEP]"]
    input_type_ids = [1] * (len(tokens_b) + 1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  x = [0] * (seq_length - len(input_ids))
  input_ids += x
  input_type_ids += x
  input_mask += x

  assert len(input_ids) == seq_length
  assert len(input_mask) == seq_length
  assert len(input_type_ids) == seq_length

  return InputFeatures(unique_id = uid,
                       tokens = tokens,
                       input_ids = input_ids,
                       input_mask = input_mask,
                       input_type_ids = input_type_ids)


def _truncate_seq(a: List[str], b: List[str], max_length: int) -> None:
  """Truncates a sequence pair in place to the maximum length."""
  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  # Modifies `tokens_a` and `tokens_b` in place so that the total
  # length is less than the specified length.

  # Account for [CLS], [SEP], [SEP] with "- 3" if b exists
  # Account for [CLS] and [SEP] with "- 2" if b not exists
  max_length = b and (max_length - 3) or (max_length - 2)
  b = b or []  # for suppress len(b)'s exception
  while max_length < len(a) + len(b):
    x = len(a) >= len(b) and a or b
    x.pop()


def each_line(max_seq_len: int,
              batch_n: int,
              sep: str,
              tokenizer) -> Iterator[List[InputFeatures]]:
  batch = []

  for line in stdin:
    line = line.strip()
    if not line:
      continue
    lid, line = line.split(sep, 1)
    ab = line.split('|||', 1)
    text_a = ab[0]
    text_b = len(ab) == 2 and ab[1] or None
    # ie = InputExample(uid, text_a, text_b)
    f = convert_example_to_features(int(lid), text_a, text_b, max_seq_len, tokenizer)

    batch.append(f)
    if len(batch) >= batch_n:
      yield batch
      batch = []
  if batch:
    yield batch


def main(bert_model: str = '/repo/bert-models/ch-base',
         sep: str = ',',
         with_lower_case: bool = True,
         layer: int = -2,
         max_seq_length: int = 128,
         batch_n: int = 32,
         without_cuda: bool = False) -> None:
  """bert-model: bert-base-uncased, bert-large-uncased,
                 bert-base-cased, bert-base-multilingual, bert-base-chinese.
  with-lower-case: Set this flag if you are using an uncased model.
  layer: -1, -2, -3, -4...-12, we just use the last layer
  max-seq-length: The maximum total input sequence length after
                  WordPiece tokenization. Sequences longer
                  than this will be truncated, and sequences shorter than
                  this will be padded.
  batch-n: Batch size for predictions.
  without-cuda: Whether not to use CUDA.  """

  without_cuda = without_cuda or not t.cuda.is_available()
  device = t.device(without_cuda and "cpu" or "cuda")
  n_gpu = t.cuda.device_count()

  log(f"device: {device} n_gpu: {n_gpu} ")

  tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case = with_lower_case)

  for features in each_line(max_seq_length, batch_n, sep, tokenizer):
    unique_id_to_feature = {}
    for feature in features:
      unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained(bert_model)
    model.to(device)

    all_input_ids = t.tensor([f.input_ids for f in features], dtype = t.long)
    all_input_mask = t.tensor([f.input_mask for f in features], dtype = t.long)
    # all_example_index = t.arange(all_input_ids.size(0), dtype = t.long)
    all_example_index = t.tensor([f.unique_id for f in features], dtype = t.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    eval_sampler = SequentialSampler(eval_data)

    eval_dataloader = DataLoader(eval_data, sampler = eval_sampler, batch_size = batch_n)

    model.eval()
    for input_ids, input_mask, uids in eval_dataloader:
      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)

      all_encoder_layers, _ = model(input_ids, token_type_ids = None, attention_mask = input_mask)
      all_encoder_layers = all_encoder_layers[layer]
      pooled = all_encoder_layers.mean(dim = 1)  # 平均池化
      o = t.cat([uids.reshape(-1, 1).float(), pooled], dim = 1)
      o.detach().numpy().tofile(stdout)


if __name__ == "__main__":
  Fire(main)
