import pytest
import torch

from itertools import product

from bitsandbytes import functional as F

def setup():
    pass

def teardown():
    pass

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=['float', 'half'])
def test_estimate_quantiles(dtype):
    A = torch.rand(1024, 1024, device='cuda')
    A = A.to(dtype)
    code = F.estimate_quantiles(A)

    percs = torch.linspace(1/512, 511/512, 256, device=A.device)
    torch.testing.assert_allclose(percs, code, atol=1e-3, rtol=1e-2)

    A = torch.randn(1024, 1024, device='cuda')
    A = A.to(dtype)
    code = F.estimate_quantiles(A)

    quantiles = torch.quantile(A.float(), percs)
    diff = torch.abs(code-quantiles)
    assert (diff > 5e-02).sum().item() == 0


def test_quantization():
    A1 = torch.rand(1024, 1024, device='cuda')
    code = F.estimate_quantiles(A1)
    C = F.quantize(code, A1)
    A2 = F.dequantize(code, C)
    diff = torch.abs(A1-A2).mean().item()
    assert diff < 0.001
    torch.testing.assert_allclose(A1, A2, atol=5e-3, rtol=0)
