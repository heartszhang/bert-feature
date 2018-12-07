import numpy as np
import pandas as pd
import faiss
from fire import Fire
from pathlib import Path


def main(vec: str,
         name: str,
         nlist: int = 32,
         nprobe: int = 2,
         k: int = 10,
         sample_n: int = 100, ) -> None:
  names = pd.read_csv(name, header = None, names = ['vid', 'name'])
  data = np.loadtxt(vec)
  # 我们的csv不是标准方式写入的，保不齐出现两个shape不对应的情况
  assert data.shape[0] == names.shape[0]
  # vids = data[:, 0].astype(int), 程序中第一列是video-id，但我们这个程序没有用这个
  vecs = data[:, 1:].astype(np.float32)
  dim = vecs.shape[1]

  quantizer = faiss.IndexFlatL2(dim)  # 我们用内积量化向量距离就可以了, 也应该尝试下别的
  # 对向量集合进行索引
  # 我不知道nlist应该多少合适，选了和CPU差不多的一个数
  index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
  index.train(vecs)
  index.add(vecs)

  # 探查次数越多就越准确，时间也翻倍
  index.nprobe = nprobe
  _, i = index.search(vecs, k)
  to = Path(name).with_suffix(f'.{k}-search').resolve()
  np.savetxt(to, i, fmt = '%d')
  # 肉眼观察一些数据
  samples = np.random.choice(len(i), sample_n)
  for sample in i[samples]:
    g = names.iloc[sample]
    print(g)


if __name__ == '__main__':
  Fire(main)
