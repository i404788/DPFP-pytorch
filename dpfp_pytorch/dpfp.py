import torch
from torch import cat
from torch.nn.functional import relu as r

def dpfp(x, nu=1):
    x = cat([r(x), r(-x)], dim=-1)
    x_rolled = cat([x.roll(shifts=j, dims=-1)
        for j in range(1,nu+1)], dim=-1)
    x_repeat = cat([x]*nu, dim=-1)
    return x_repeat*x_rolled
