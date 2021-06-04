import torch

from bitsandbytes import ops

def setup():
    pass

def teardown():
    pass

def test_scalar():
    a = torch.rand(10,10).cuda()
    torch.optim.Adam
    out = torch.rand(10,10).cuda()
    ops.testmul(a, 3.0, out)
    torch.testing.assert_allclose(a*3.0, out)


def test_estimate_quantiles():

    A = torch.rand(1024, 1024, device='cuda')
    code = ops.estimate_quantiles(A)

    percs = torch.linspace(1/512, 511/512, 256, device=A.device)
    torch.testing.assert_allclose(percs, code, atol=1e-3, rtol=1e-2)

    A = torch.randn(1024, 1024, device='cuda')
    code = ops.estimate_quantiles(A)

    quantiles = torch.quantile(A, percs)
    diff = torch.abs(code-quantiles)
    assert (diff > 5e-02).sum().item() == 0
