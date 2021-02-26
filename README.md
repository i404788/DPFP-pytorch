# DPFP-pytorch

Implementation of [DPFP](https://arxiv.org/pdf/2102.11174v2.pdf) in pytorch as provided in the paper.

## Install

```
$ pip install dpfp-pytorch
```

# Usage
```py3
from dpfp_pytorch import dpfp
import torch

features = torch.randn(4,3,256,256)

x = dpfp(features)
# torch.Size([4, 3, 256, 512])

x = dpfp(features, nu=2)
# torch.Size([4, 3, 256, 1024])

```


# Cite the authors
```bibtex
@misc{schlag2021linear,
      title={Linear Transformers Are Secretly Fast Weight Memory Systems}, 
      author={Imanol Schlag and Kazuki Irie and Jürgen Schmidhuber},
      year={2021},
      eprint={2102.11174},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
