import torch
from dpfp_pytorch import DPFPCell

nu = 1
dim = 128
m = DPFPCell(dim, nu=nu)
W = torch.randn(1, 2 * nu * dim, 2 * nu * dim)
x = torch.randn(1, dim)

x, W = m.forward(x, W)
x, W = m.forward(x, W)

print(x.shape, W.shape)
