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
  assert data.shape[0] == names.shape[0]
  vids = data[:, 0].astype(int)
  vecs = data[:, 1:].astype(np.float32)
  dim = vecs.shape[1]

  quantizer = faiss.IndexFlatL2(dim)
  index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
  index.train(vecs)
  index.add(vecs)

  index.nprobe = nprobe
  _, i = index.search(vecs, k)
  to = Path(name).with_suffix(f'.{k}-search').resolve()
  np.savetxt(to, i, fmt = '%d')
  samples = np.random.choice(len(i), sample_n)
  for sample in i[samples]:
    g = names.iloc[sample]
    print(g)


if __name__ == '__main__':
  Fire(main)
