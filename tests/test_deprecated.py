import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import BOOLEAN_TRIPLES, describe_dtype, get_test_dims, id_formatter
from tests.test_autograd import TRANSPOSE_VALS


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


@pytest.mark.parametrize("dim1", get_test_dims(16, 64, n=1), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [*get_test_dims(32, 96, n=1), 0], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 96, n=1), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", get_test_dims(32, 96, n=1), ids=id_formatter("dim4"))
@pytest.mark.parametrize("req_grad", BOOLEAN_TRIPLES, ids=id_formatter("req_grad"))
@pytest.mark.parametrize("transpose", TRANSPOSE_VALS, ids=id_formatter("transpose"))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize(
    "funcs",
    [(torch.matmul, bnb.research.matmul_fp8_mixed), (torch.matmul, bnb.research.matmul_fp8_global)],
    ids=["matmul_fp8_mixed", "matmul_fp8_global"],
)
@pytest.mark.deprecated
@pytest.mark.skip("Deprecated functionality, to be removed.")
def test_matmul_fp8(dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose):
    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    req_grad = list(req_grad)
    req_grad[2] = False

    for i in range(3):
        # normal multiply
        if funcs[0] in [torch.mm, torch.matmul]:
            A = torch.randn(size=dimA, device="cuda", requires_grad=req_grad[0], dtype=dtype)
            B = torch.randn(size=dimB, device="cuda", requires_grad=req_grad[1], dtype=dtype)
            target = torch.randn(size=(dim2, dim4), device="cuda", requires_grad=req_grad[1], dtype=dtype)

            torch.nn.init.xavier_uniform_(B)

            fw_code = bnb.functional.create_fp8_map(True, 4, 3, 8).to(A.device)
            bw_code = bnb.functional.create_fp8_map(True, 5, 2, 8).to(A.device)

            if not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B.t(), fw_code, bw_code)
            elif not transpose[0] and not transpose[1]:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B, fw_code, bw_code)

            assert out_bnb.dtype == A.dtype, f"bnb matmullt received {A.dtype} but returned {out_bnb.dtype}"

            n = out_bnb.numel()
            err = torch.abs(out_bnb - out_torch).float().mean().item()
            if n > 0:
                assert err < 0.115
                # assert err < 0.20
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
                    if dim2 > 0:
                        assert torch.abs(gradB1).sum() > 0.0
                        assert torch.abs(gradB2).sum() > 0.0
                    else:
                        assert torch.abs(gradB1).sum() == 0.0
                        assert torch.abs(gradB2).sum() == 0.0
                    idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)

                    assert (idx == 0).sum().item() <= n * 0.1
                    idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                    assert (idx == 0).sum().item() <= n * 0.02
                    grad_err = (gradB1 - gradB2).abs().mean()
                    assert grad_err.item() < 0.003
                    torch.testing.assert_close(gradB1, gradB2, atol=0.18, rtol=0.3)


@pytest.mark.deprecated
def test_fp8linear():
    b = 10
    h = 1024
    inp = torch.randn(b, h).cuda()
    fp32 = torch.nn.Linear(h, h * 2).cuda()
    fp8 = bnb.research.nn.LinearFP8Mixed(h, h * 2).cuda()
    fp32b = torch.nn.Linear(h * 2, h).cuda()
    fp8b = bnb.research.nn.LinearFP8Mixed(h * 2, h).cuda()

    fp8.weight.data.copy_(fp32.weight.data)
    fp8.bias.data.copy_(fp32.bias.data)
    fp8b.weight.data.copy_(fp32b.weight.data)
    fp8b.bias.data.copy_(fp32b.bias.data)

    a = fp32b(torch.nn.functional.gelu(fp32(inp)))
    b = fp8b(torch.nn.functional.gelu(fp8(inp)))

    err = (a - b).abs().mean()

    a.mean().backward()
    b.mean().backward()

    graderr = (fp8.weight.grad - fp32.weight.grad).abs().mean()
    bgraderr = (fp8.bias.grad - fp32.bias.grad).abs().mean()

    assert err < 0.05
    assert graderr < 0.00002
    assert bgraderr < 0.00002
