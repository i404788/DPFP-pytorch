import torch
from dpfp_pytorch import DPFPCell, DPFPAttention, DPFPformer

# Single DPFP Recurrent cell
nu = 1
dim = 128
m = DPFPCell(dim, nu=nu)
W = torch.randn(1, 2 * nu * dim, 2 * nu * dim)
x = torch.randn(1, dim)

x, W = m.forward(x, W)

# Single multi-head attntention
m = DPFPAttention(dim)
x = torch.randn(1, 16, dim)
W = torch.zeros(m.get_memory_shape(1), device=x.device)
m.forward(x, W)


# Full transformer model with DPFP attention
m = DPFPformer(dim, 3)
x = torch.randn(1, 16, dim)
x = m.forward(x)

