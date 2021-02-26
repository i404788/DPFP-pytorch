import torch
import torch.nn as nn
from torch import cat
from torch.nn.functional import relu, sigmoid

def dpfp(x, nu=1):
    x = cat([relu(x), relu(-x)], dim=-1)
    x_rolled = cat([x.roll(shifts=j, dims=-1) for j in range(1, nu+1)], dim=-1)
    x_repeat = cat([x]*nu, dim=-1)
    return x_repeat*x_rolled


class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head=64, dropout=0.):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.to_beta = nn.Linear(inner_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None)
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        context = context or x

        # k(i),v(i),q(i)=Wkx(i),Wvx(i),Wqx(i)
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        # β(i)=σ(Wβx(i))
        beta = sigmoid(self.to_beta(x))

        # v(i)=W(i−1)φ(k(i))
        # v =   

