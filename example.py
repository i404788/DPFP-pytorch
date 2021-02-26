import torch
from dpfp_pytorch import SelfAttention

m = SelfAttention(128)
W = torch.randn(1, 256, 256)
x = torch.randn(1, 128)


print(x.shape, W.shape)
x, W = m.forward(x, W)
x, W = m.forward(x, W)

print(x.shape, W.shape)
