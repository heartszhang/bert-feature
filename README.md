# bert-feature

![Python: 3.7](https://img.shields.io/badge/Python-3.7-brightgreen.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)
![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)

## Requirements

- Anaconda or `Miniconda`
- Python >=3.6 (and Python 2 is NOT Supported)
- `PyTorch` nightly build
- `TensorboardX`
- pandas,  numpy...
- `fire`
- *pytorch-pretrained-BERT* (`pip install pytorch-pretrained-bert`)

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
id, vec = vecs[:,0], vecs[:,1:]
assert vec.shape[1]==768
```

