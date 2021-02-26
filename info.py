import torch
from dpfp_pytorch import DPFPCell

nu = 1
dim = 128
m = DPFPCell(dim, nu=nu)
W = torch.randn(1, 2 * nu * dim, 2 * nu * dim)
x = torch.randn(1, dim)

if __name__ == "__main__":
    import torch_inspect as ti
    from thop import profile, clever_format
    from torchprofile import profile_macs

    true_macs = profile_macs(m, (x, W)) 
#    ti.summary(m, input_size=(x.shape, W.shape), batch_size=1)
    print('\n')
    macs, params = profile(m, inputs=(x,W))
    true_macs, macs, params = clever_format([true_macs, macs, params], "%.3f")

    print(f'For DPFPCell-{nu}, dim=128:')
    print(f'Input shapes: x={tuple(x.shape)}, W={tuple(W.shape)}')
    print("Macs: %s (alt. %s), Params: %s" % (true_macs, macs, params))

    def print_params(model):
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))


    print_params(m)
