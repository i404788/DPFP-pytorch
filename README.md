# DPFP-pytorch

Implementation of [DPFP](https://arxiv.org/pdf/2102.11174v2.pdf) in pytorch as provided in the paper.

Note that the "update rule" is naively implemented on CPU/TPU (CUDA is optimized), as mentioned by the paper:
> We note that a custom implementation of backward pass for the fast weight is crucial for language modelling. A naive backward computation generated by automatic differentiation would store the fast weights for each time step, which can quickly hit the GPU memory limit. 

## Install

```
$ pip install dpfp-pytorch
```

# Usage
## DPFP-v Transformer
```py3
import torch
from dpfp_pytorch import DPFPformer

v = 1
dim = 128
depth = 3
batch = 16

model = DPFPformer(dim, depth, nu=v)
x = torch.randn(batch, 16, dim) # b, seq, dim

x = m.forward(x)
```

## DPFP-v Multi-head Attention Module
```py3
import torch
from dpfp_pytorch import DPFPAttention

v = 1
dim = 128
batch = 1
m = DPFPAttention(dim, nu=v)
x = torch.randn(batch, 16, dim) # b, seq, dim
W = torch.zeros(m.get_memory_shape(batch), device=x.device)

x, W = m.forward(x, W)
```

## DPFP-v 'Cell'
This is the DPFP-v Cell is directly translated from the paper, so no multi-head attention or anything fancy.

This variant is also not optimized for auto-differentiation unlike `DPFPformer` and `DPFPAttention`.

```py3
import torch
from dpfp_pytorch import DPFPCell

v = 1
dim = 128

m = DPFPCell(dim, nu=v)
W = torch.randn(1, 2 * nu * dim, 2 * nu * dim)
x = torch.randn(1, dim)

x, W = m.forward(x, W)
```

## Projection Mechanism
```py3
from dpfp_pytorch import dpfp
import torch

features = torch.randn(4,3,256,256)

x = dpfp(features)
# torch.Size([4, 3, 256, 512])

x = dpfp(features, nu=2)
# torch.Size([4, 3, 256, 1024])
```


# Citations
The cuda module is adapted from the original research repository: https://github.com/IDSIA/lmtool-fwms

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
