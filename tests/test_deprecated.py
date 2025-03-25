import numpy as np
import pytest
from scipy.stats import norm
import torch

from bitsandbytes import functional as F


@pytest.mark.deprecated
def test_kbit_quantile_estimation():
    for i in range(100):
        data = torch.randn(1024, 1024, device="cuda")
        for bits in range(2, 9):
            p = np.linspace(1.3e-4, 1 - 1.3e-4, 2**bits)
            val1 = torch.Tensor(norm.ppf(p)).cuda()
            val2 = F.estimate_quantiles(data, offset=0, num_quantiles=2**bits)
            err = torch.abs(val1 - val2).mean()
            assert err < 0.038

    for i in range(100):
        data = torch.randn(1024, 1024, device="cuda")
        for bits in range(2, 4):
            total_values = 2**bits - 1
            p = np.linspace(0, 1, 2 * total_values + 1)
            idx = np.arange(1, 2 * total_values + 1, 2)
            p = p[idx]
            offset = 1 / (2 * total_values)
            p = np.linspace(offset, 1 - offset, total_values)
            val1 = torch.Tensor(norm.ppf(p)).cuda()
            val2 = F.estimate_quantiles(data, num_quantiles=2**bits - 1)
            err = torch.abs(val1 - val2).mean()
            assert err < 0.035


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["float", "half"])
@pytest.mark.deprecated
def test_estimate_quantiles(dtype):
    A = torch.rand(1024, 1024, device="cuda")
    A = A.to(dtype)
    code = F.estimate_quantiles(A)

    percs = torch.linspace(1 / 512, 511 / 512, 256, device=A.device)
    torch.testing.assert_close(percs, code, atol=1e-3, rtol=1e-2)

    A = torch.randn(1024, 1024, device="cuda")
    A = A.to(dtype)
    code = F.estimate_quantiles(A)

    quantiles = torch.quantile(A.float(), percs)
    diff = torch.abs(code - quantiles)
    assert (diff > 5e-02).sum().item() == 0


@pytest.mark.deprecated
def test_quantile_quantization():
    for i in range(100):
        A1 = torch.randn(1024, 1024, device="cuda")
        code = F.estimate_quantiles(A1)
        C = F.quantize_no_absmax(A1, code)
        A2 = F.dequantize_no_absmax(C, code)
        diff = torch.abs(A1 - A2).mean().item()
        assert diff < 0.0075

        A1 = torch.rand(1024, 1024, device="cuda")
        code = F.estimate_quantiles(A1)
        C = F.quantize_no_absmax(A1, code)
        A2 = F.dequantize_no_absmax(C, code)
        diff = torch.abs(A1 - A2).mean().item()
        torch.testing.assert_close(A1, A2, atol=5e-3, rtol=0)
        assert diff < 0.001


@pytest.mark.deprecated
def test_dynamic_quantization():
    diffs = []
    reldiffs = []
    for i in range(100):
        A1 = torch.randn(1024, 1024, device="cuda")
        C, S = F.quantize(A1)
        A2 = F.dequantize(C, S)
        diff = torch.abs(A1 - A2)
        reldiff = diff / torch.abs(A1 + 1e-8)
        diffs.append(diff.mean().item())
        reldiffs.append(reldiff.mean().item())
        assert diff.mean().item() < 0.0135
    print(sum(diffs) / len(diffs))
    print(sum(reldiffs) / len(reldiffs))

    for i in range(100):
        A1 = torch.rand(1024, 1024, device="cuda")
        C, S = F.quantize(A1)
        A2 = F.dequantize(C, S)
        diff = torch.abs(A1 - A2).mean().item()
        torch.testing.assert_close(A1, A2, atol=1e-2, rtol=0)
        assert diff < 0.004


@pytest.mark.parametrize("gtype", [torch.float32, torch.float16], ids=["float", "half"])
@pytest.mark.deprecated
def test_percentile_clipping(gtype):
    gnorm_vec1 = torch.zeros(100, device="cuda")
    gnorm_vec2 = torch.zeros(100, device="cuda")
    n = 4
    step = 0
    percentile = 5
    for i in range(20):
        step += 1
        g = torch.randn(n, n, dtype=gtype, device="cuda")
        gnorm1, clip2, gnorm_scale = F.percentile_clipping(g, gnorm_vec2, step, percentile=percentile)
        assert gnorm_scale == 1.0 if gnorm1 < clip2 else clip2 / gnorm1

        gnorm2 = torch.norm(g.float())
        if step == 1:
            gnorm_vec1[:] = gnorm2
        else:
            gnorm_vec1[step % 100] = gnorm2

        vals, idx = torch.sort(gnorm_vec1)
        clip1 = vals[percentile]

        torch.testing.assert_close(gnorm_vec1, torch.sqrt(gnorm_vec2))
        torch.testing.assert_close(clip1, clip2)
        torch.testing.assert_close(gnorm1, gnorm2)
