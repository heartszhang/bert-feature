"""Extract pre-computed feature vectors from a PyTorch BERT model.
Some codes are copied from bert-pretrained-pytorch/examples."""
import torch as t
import numpy as np
from sys import stdin, stderr, stdout
from typing import Iterator, List
from torch.utils.data import TensorDataset, DataLoader
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from fire import Fire


def log(*args, **kwargs):
  print(*args, file = stderr, **kwargs)


# InputFeatures 表示一个视频标题
# unique_id: video-id
# input_ids: 字编号
# input_masks: 变长句子变成定长句子的时候，会填充一些空token，masks标记那些是填充的
# input_type_ids: bert两个句子组成一个输入，type_id标记每个token属于第一句还是第二句
# 因为我们没有第二句，input_type是 全0
# 中文内容，一个Token就是一个汉字
class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def convert_example_to_features(uid: int, text_a: str,
                                seq_length: int,
                                tokenizer: BertTokenizer, ) -> InputFeatures:
  tokens_a = tokenizer.tokenize(text_a)

  # 我们只处理一个句子，对长句截断, 所以只需要头尾附加CLS/SEP
  tokens_a = tokens_a[:seq_length - 2]

  # For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
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
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # Zero-pad up to the sequence length.
  x = [0] * (seq_length - len(input_ids))
  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids) + x
  input_ids += x
  input_type_ids += x

  assert len(input_ids) == seq_length
  assert len(input_mask) == seq_length
  assert len(input_type_ids) == seq_length
  return InputFeatures(unique_id = uid,
                       tokens = tokens,
                       input_ids = input_ids,
                       input_mask = input_mask,
                       input_type_ids = input_type_ids)


# 从stdin逐行读入短时频标题
# 两个字段用逗号分割, video_id, name, name是经过去标点的后的文字
# batch-n: 每次预测的批次大小, 根据我们的GPU选择 合适大小
def each_line(max_seq_len: int,
              batch_n: int,
              sep: str,
              tokenizer) -> Iterator[List[InputFeatures]]:
  batch = []
  for line in stdin:
    line = line.strip()  # 去掉空白
    if not line:
      continue
    lid, text_a = line.split(sep, 1)
    f = convert_example_to_features(int(lid), text_a, max_seq_len, tokenizer)

    batch.append(f)
    if len(batch) >= batch_n:
      yield batch
      batch = []
  if batch:
    yield batch


def main(bert_model: str = '/repo/bert-models/ch-base',
         sep: str = ',',
         layer: int = -2,
         max_seq_length: int = 128,
         batch_n: int = 64,
         without_cuda: bool = False) -> None:
  """--bert-model: bert-base-uncased, bert-large-uncased,
                 bert-base-cased, bert-base-multilingual, bert-base-chinese.
  --with-lower-case: Set this flag if you are using an uncased model.
  --layer: -1, -2, -3, -4...-12, we just use the second last layer
  --max-seq-length: The maximum total input sequence length after WordPiece tokenization.
                    Sequences longer than this will be truncated,
                    and sequences shorter than this will be padded.
  --batch-n: Batch size for predictions.
  --without-cuda: Whether not to use CUDA.  """

  without_cuda = without_cuda or not t.cuda.is_available()
  device = t.device(without_cuda and "cpu" or "cuda")
  n_gpu = t.cuda.device_count()

  log(f"device: {device} n_gpu: {n_gpu} ")

  tokenizer = BertTokenizer.from_pretrained(bert_model)

  for features in each_line(max_seq_length, batch_n, sep, tokenizer):
    model = BertModel.from_pretrained(bert_model)
    model.to(device)

    input_ids = t.tensor([f.input_ids for f in features], dtype = t.long)
    input_mask = t.tensor([f.input_mask for f in features], dtype = t.long)
    unique_ids = t.tensor([f.unique_id for f in features], dtype = t.long)

    # 为了把每个句子和它的视频id对应起来，我们把unique_ids也传进去
    eval_data = TensorDataset(input_ids, input_mask, unique_ids)
    eval_dataloader = DataLoader(eval_data, batch_size = batch_n)

    model.eval()
    for input_ids, input_mask, uids in eval_dataloader:
      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)

      pools, _ = model(input_ids,
                       token_type_ids = None,
                       attention_mask = input_mask)
      pools = pools[layer]
      # 需要转换成64位浮点，否则会出现video_id表示错乱的情况
      pooled = pools.mean(dim = 1).double()  # 用什么池也是个问题, 平均池化是因为简单
      o = t.cat([uids.reshape(-1, 1).double(), pooled], dim = 1)
      # 把ndarray按行输出
      np.savetxt(stdout, o.detach().numpy(), fmt = '%.8g')


if __name__ == "__main__":
  Fire(main)
