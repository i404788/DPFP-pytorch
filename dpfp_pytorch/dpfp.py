import torch
import torch.nn as nn
from torch import cat
from torch.nn.functional import relu
from einops import rearrange
from functools import partial

identity = lambda x: x

def dpfp(x, nu=1):
    x = cat([relu(x), relu(-x)], dim=-1)
    x_rolled = cat([x.roll(shifts=j, dims=-1) for j in range(1, nu+1)], dim=-1)
    x_repeat = cat([x]*nu, dim=-1)
    return x_repeat*x_rolled

class DPFPCell(nn.Module):
    def __init__(self, dim, nu=1, dropout=0.):
        super().__init__()
        inner_dim = 2 * dim * nu

        self.nu = nu
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.to_beta = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, W, context=None):
        context = context or x

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

        # y(i)=W(i)φ(q(i))
        out = torch.einsum('bij, bi -> bi', W, dpfp(q, nu=self.nu))
        out = self.to_out(out)
        out = self.dropout(out)
        return out, W
            

class DPFPAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, nu=1, dropout=0., layernorm=False, prenorm=False):
        super().__init__()
        self.ln = nn.LayerNorm(dim) if layernorm else identity
        self.prenorm = prenorm
        self.heads = heads
        self.dim_head = dim_head
        self.nu = nu

        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_beta = nn.Linear(dim, heads)
        self.to_out = nn.Linear(inner_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.dpfp = partial(dpfp, nu=nu)
        self.scale = 1 / (dim_head ** 0.5)

    def forward(self, x, W=None, mask=None):
        b, n, _ = x.shape
        h, nu = self.heads, self.nu
        
        if self.prenorm:
            x = self.ln(x) 

        # Generate multi-head q,k,v,beta by linear projection
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        beta = self.to_beta(x)
        q, k, v, beta = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v, beta))

        if mask:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # Apply dpfp & normalization to q, k
        q, k = map(self.dpfp, (q, k))
        q, k = map(lambda t: t / t.sum(-1, keepdim=True), (q, k)) # Normalize (from reference impl)
        
        if not W:
            W = torch.zeros(b, self.heads, 2 * self.nu * self.dim_head, self.dim_head, device=x.device)

        # Apply update function
        if 'cuda' in str(x.device):
            raise NotImplementedError("CUDA coming soon")
            # TODO: copy fast_weight_memory cuda
            pass
        else:
            # Autograd-naive version
            vo = torch.einsum('bhsd, bhns -> bhnd', W, k)
            dv = beta * (v - vo)
            W = W + torch.einsum('bhnd, bhns -> bhsd', (dv, k))
            out = torch.einsum('bhsd, bhns -> bhnd', W, q)

        # Standard multi-head attention post processing
        out = out * self.scale
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = self.drop(out)

        out = out + x
        if not self.prenorm:
            out = self.ln(out)

        return out




