import torch
import torch.nn as nn
from torch import cat
from torch.nn.functional import relu

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

def dpfp(x, nu=1):
    x = cat([relu(x), relu(-x)], dim=-1)
    x_rolled = cat([x.roll(shifts=j, dims=-1) for j in range(1, nu+1)], dim=-1)
    x_repeat = cat([x]*nu, dim=-1)
    return x_repeat*x_rolled

class DPFPCell(nn.Module):
    def __init__(self, dim, heads = 8, dim_head=64, nu=1, dropout=0.):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = 2 * dim * nu

        self.nu = nu
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.to_beta = nn.Linear(dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, W, context=None):
        b, n = x.shape
        context = context or x

        # W = torch.zeros(b, ddot, ddot)

        # k(i),v(i),q(i)=Wkx(i),Wvx(i),Wqx(i)
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        # β(i)=σ(Wβx(i))
        beta = torch.sigmoid(self.to_beta(x))

        #  ̄v(i)=W(i−1)φ(k(i)i)
        vo = torch.einsum('bij, bj -> bj', W, dpfp(k, nu=self.nu))

        # W(i−1)+β(i)(v(i)− ̄v(i))⊗ φ(k(i))
        dv = beta * (v - vo)
        kp = dpfp(k, nu=self.nu)
        W = W + torch.einsum('bp, bq->bpq', (dv, kp))

        # y(i)=W(i)φ(q(i)
        return self.to_out(torch.einsum('bij, bi -> bi', W, dpfp(q, nu=self.nu))), W
            


