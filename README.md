# bert-feature

![Python: 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)
![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)

`bert-feature` 是一个利用google `bert` 预训练模型对橙子标题进行向量化的一个例子，然后我们利用`FAISS`搜索每个视频标题最临近的k个标题。

*希望说明白了*

## Requirements

- Anaconda or `Miniconda`
- Python ==3.6 (and Python 2 is NOT Supported)
- `PyTorch` nightly build
- `TensorboardX`
- pandas,  numpy...
- `fire`
- *pytorch-pretrained-BERT* (`pip install pytorch-pretrained-bert`)
- `faiss-gpu` or `faiss-cpu`

## Usage

```bash
python chinese.py --bert-model /repo/bert-models/ch-base \
  --seq=',' \
  --batch-n=64 </repo/svideo.name >svideo-768.nd 
```

### Read the nd like this

```python
import numpy as np
vecs = np.fromfile('svideo-768.nd')
vecs = vecs.reshape(-1, 768+1)  
id, vec = vecs[:,0].astype(int), vecs[:,1:].astype(np.float32)
assert vec.shape[1]==768
```

### 搜索和某个视频相近的k个视频 (via `FAISS`)

```python
import pandas as pd
import numpy as np
import faiss 
from pathlib import Path

name = 'orange.name'
vec = 'orange-768.nd'
nlist = 32
nprobe = 2
k = 10
sample_n = 100

names = pd.read_csv(name, header = None, names = ['vid', 'name'])
data = np.loadtxt(vec)
# 我们的csv不是标准方式写入的，保不齐出现两个shape不对应的情况
assert data.shape[0] == names.shape[0]

# vids = data[:, 0].astype(int), 程序中第一列是video-id，但我们这个程序没有用这个
vecs = data[:, 1:].astype(np.float32)
dim = vecs.shape[1]

# 对向量集合进行索引
# 我不知道nlist应该多少合适，选了和CPU差不多的一个数
# 我们用欧范衡量距离就可以了, 也应该尝试下别的
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
index.train(vecs)
index.add(vecs)

# 探查次数越多就越准确，时间也翻倍
index.nprobe = nprobe
_, i = index.search(vecs, k)  # 搜索所有视频的前10个相似向量
# 因为我们只用了20K个enable的橙子视频，所以整个过程还很快，分钟内的事儿
# 2M个短时频可能会慢不少

# 存下结果来
to = Path(name).with_suffix(f'.{k}-search').resolve()
np.savetxt(to, i, fmt = '%d')

# 肉眼观察一些数据
samples = np.random.choice(len(i), sample_n)
for sample in i[samples]:
  g = names.iloc[sample]
  print(g)
```

当然也可以用`match.py`
```bash
python match.py --vec=orange-768.nd --name=orange.name
```
## FAQ

- `conda install faiss-gpu cuda92` 如果有GPU的话
- `pip install pytorch-pretrained-bert`
- 模型文件是转换过的google官方中文预先训练模型
- 可以在[pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT)获取
- 也可以在 http://10.1.7.198:8091/bert-features/ 下载
- 如果不能访问7.198，也可以使用 http://192.168.8.220:18091/bert-features/ 这个地址
- 如果有什么疑问欢迎交流
- 如果有什么好建议，希望带着`代码`来
