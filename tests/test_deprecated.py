from typing import Tuple

import numpy as np
import pytest
from scipy.stats import norm
import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
    BOOLEAN_TUPLES,
    describe_dtype,
    get_test_dims,
    id_formatter,
)


@pytest.mark.parametrize("dim1", get_test_dims(16, 64, n=1), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(32, 96, n=1), ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 96, n=1), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", get_test_dims(32, 96, n=1), ids=id_formatter("dim4"))
@pytest.mark.parametrize(
    "funcs",
    [(torch.bmm, bnb.bmm_cublas), (torch.matmul, bnb.matmul_cublas)],
    ids=["func=bmm", "func=matmul"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=describe_dtype)
@pytest.mark.parametrize("req_grad", BOOLEAN_TUPLES, ids=id_formatter("req_grad"))
@pytest.mark.parametrize("transpose", BOOLEAN_TUPLES, ids=id_formatter("transpose"))
@pytest.mark.deprecated
def test_matmul(dim1, dim2, dim3, dim4, funcs, dtype, req_grad: Tuple[bool, bool], transpose: Tuple[bool, bool]):
    if dim2 > 0:
        dim2 = dim2 - (dim2 % 16)
    dim3 = dim3 - (dim3 % 16)
    dim4 = dim4 - (dim4 % 16)
    for i in range(25):
        # normal multiply
        if funcs[0] in [torch.mm, torch.matmul]:
            dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
            dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
            A = torch.randn(size=dimA, device="cuda", requires_grad=req_grad[0])
            B = torch.randn(size=dimB, device="cuda", requires_grad=req_grad[1])
            target = torch.randn(size=(dim2, dim4), device="cuda", requires_grad=req_grad[1])
            torch.nn.init.xavier_uniform_(B)

            if not transpose[0] and not transpose[1]:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B)
            elif not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B.t())
            elif transpose[0] and not transpose[1]:
                out_torch = funcs[0](A.t(), B)
                out_bnb = funcs[1](A.t(), B)
            elif transpose[0] and transpose[1]:
                out_torch = funcs[0](A.t(), B.t())
                out_bnb = funcs[1](A.t(), B.t())

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() < n * 0.0175
            idx = torch.isclose(out_bnb, out_torch, atol=0.035, rtol=0.2)
            assert (idx == 0).sum().item() < n * 0.001

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.02
                torch.testing.assert_close(gradB1, gradB2, atol=0.18, rtol=0.3)

        # batched matrix multiply
        if funcs[0] in [torch.bmm, torch.matmul]:
            A = torch.randn(
                size=(dim1, dim2, dim3),
                device="cuda",
                requires_grad=req_grad[0],
            )
            B = torch.randn(
                size=(dim1, dim3, dim4),
                device="cuda",
                requires_grad=req_grad[1],
            )
            target = torch.randn(
                size=(dim1, dim2, dim4),
                device="cuda",
                requires_grad=req_grad[1],
            )
            torch.nn.init.xavier_uniform_(B)

            out_torch = funcs[0](A, B)
            out_bnb = funcs[1](A, B)

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() < n * 0.01
            torch.testing.assert_close(out_bnb, out_torch, atol=0.027, rtol=0.2)

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.02

        if funcs[0] in [torch.matmul]:
            dim1 = dim1 - (dim1 % 16)
            A = torch.randn(
                size=(dim1, dim2, dim3),
                device="cuda",
                requires_grad=req_grad[0],
            )
            dimB = (dim4, dim3) if transpose[1] else (dim3, dim4)
            B = torch.randn(size=dimB, device="cuda", requires_grad=req_grad[1])
            target = torch.randn(
                size=(dim1, dim2, dim4),
                device="cuda",
                requires_grad=req_grad[1],
            )
            torch.nn.init.xavier_uniform_(B)

            if transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B.t())
            else:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B)

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() < n * 0.0175
            idx = torch.isclose(out_bnb, out_torch, atol=0.035, rtol=0.2)
            assert (idx == 0).sum().item() < n * 0.001

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.02


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
