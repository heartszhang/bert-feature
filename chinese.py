"""Extract pre-computed feature vectors from a PyTorch BERT model."""
import argparse
import collections
import json
import re
import torch as t
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from sys import stderr
from fire import Fire


def log(*args, **kwargs):
  print(*args, file = stderr, **kwargs)


class InputExample(object):
  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

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
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5:
      log("*** Example ***")
      log("unique_id: %s" % example.unique_id)
      log("tokens: %s" % " ".join([str(x) for x in tokens]))
      log("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      log("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      log("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(InputFeatures(unique_id = example.unique_id,
                                  tokens = tokens,
                                  input_ids = input_ids,
                                  input_mask = input_mask,
                                  input_type_ids = input_type_ids))
  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with open(input_file, "r") as reader:
    while True:
      line = reader.readline()
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      examples.append(InputExample(unique_id = unique_id, text_a = text_a, text_b = text_b))
      unique_id += 1
  return examples


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--input_file", default = 'sample.txt', type = str, )
  parser.add_argument("--output_file", default = 'sample.result', type = str, )
  parser.add_argument("--bert_model", default = '/repo/bert-models/ch-base',
                      type = str,
                      help = "Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, "
                             "bert-base-chinese.", )

  # Other parameters
  parser.add_argument("--do_lower_case", default = False, action = 'store_true',
                      help = "Set this flag if you are using an uncased model.")
  parser.add_argument("--layers", default = "-2", type = str)
  parser.add_argument("--max_seq_length", default = 128, type = int,
                      help = "The maximum total input sequence length after "
                             "WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than "
                             "this will be padded.")
  parser.add_argument("--batch_size", default = 32, type = int,
                      help = "Batch size for predictions.")
  parser.add_argument("--without_cuda",
                      default = False,
                      action = 'store_true',
                      help = "Whether not to use CUDA when available")

  args = parser.parse_args()
  args.without_cuda = not t.cuda.is_available() and args.without_cuda
  device = t.device("cuda" if args.without_cuda else "cpu")
  n_gpu = t.cuda.device_count()

  log("device: {} n_gpu: {} ".format(device, n_gpu))

  layer_indexes = [int(x) for x in args.layers.split(",")]

  tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case = args.do_lower_case)

  examples = read_examples(args.input_file)

  features = convert_examples_to_features(examples = examples, seq_length = args.max_seq_length,
                                          tokenizer = tokenizer, )

  unique_id_to_feature = {}
  for feature in features:
    unique_id_to_feature[feature.unique_id] = feature

  model = BertModel.from_pretrained(args.bert_model)
  model.to(device)

  all_input_ids = t.tensor([f.input_ids for f in features], dtype = t.long)
  all_input_mask = t.tensor([f.input_mask for f in features], dtype = t.long)
  all_example_index = t.arange(all_input_ids.size(0), dtype = t.long)

  eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
  eval_sampler = SequentialSampler(eval_data)

  eval_dataloader = DataLoader(eval_data, sampler = eval_sampler, batch_size = args.batch_size)

  model.eval()
  with open(args.output_file, "w", encoding = 'utf-8') as writer:
    for input_ids, input_mask, example_indices in eval_dataloader:
      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)

      all_encoder_layers, _ = model(input_ids, token_type_ids = None, attention_mask = input_mask)
      all_encoder_layers = all_encoder_layers[-2]
      pooled = all_encoder_layers.mean(dim = 1)
      print(pooled.shape)


if __name__ == "__main__":
  main()
