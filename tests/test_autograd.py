import pytest
import torch

import bitsandbytes as bnb
from tests.helpers import (
    BOOLEAN_TRIPLES,
    TRUE_FALSE,
    describe_dtype,
    get_available_devices,
    id_formatter,
    is_supported_on_hpu,
)

TRANSPOSE_VALS = [(False, True), (False, False)]


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dim1", [40], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [64, 0], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", [32], ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", [48], ids=id_formatter("dim4"))
@pytest.mark.parametrize("decomp", [0.0, 6.0], ids=id_formatter("decomp"))
@pytest.mark.parametrize(
    "funcs",
    [(torch.matmul, bnb.matmul)],
    ids=["func=matmul"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("req_grad", BOOLEAN_TRIPLES, ids=id_formatter("req_grad"))
@pytest.mark.parametrize("transpose", TRANSPOSE_VALS, ids=id_formatter("transpose"))
@pytest.mark.parametrize("has_fp16_weights", TRUE_FALSE, ids=id_formatter("has_fp16_weights"))
@pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
def test_matmullt(
    device, dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose, decomp, has_fp16_weights, has_bias
):
    if device != "cuda":
        if req_grad[1]:
            # This will be deprecated for CUDA in the future. We don't expect
            # this to work on any other device.
            pytest.skip("Deprecated feature with CUDA support only.")

    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    outlier_dim = torch.randint(0, dimA[1], size=(dimA[1] // 8,), device=device)
    if has_bias == False:
        req_grad = list(req_grad)
        req_grad[2] = False

    if device == "cpu" and dtype != torch.float32 and has_fp16_weights and any(req_grad):
        if torch.__version__ < (2, 6):
            pytest.xfail("mse_loss bf16/fp16 on CPU is not supported in torch < 2.6")

    for i in range(3):
        # normal multiply
        if funcs[0] in [torch.mm, torch.matmul]:
            A = torch.randn(size=dimA, device=device, requires_grad=req_grad[0], dtype=dtype)
            if decomp == 6.0:
                with torch.no_grad():
                    A[:, outlier_dim] = 6.0
            B = torch.randn(size=dimB, device=device, requires_grad=req_grad[1], dtype=dtype)
            target = torch.randn(
                size=(dim2, dim4),
                device=device,
                requires_grad=req_grad[1],
                dtype=dtype,
            )
            bias = None
            bias2 = None
            if has_bias:
                bias = torch.randn(dim4, device=device, dtype=dtype, requires_grad=req_grad[2])
                bias2 = bias.clone()
            torch.nn.init.xavier_uniform_(B)
            B2 = B.clone()

            state = bnb.MatmulLtState()
            state.threshold = decomp
            state.has_fp16_weights = has_fp16_weights
            if not has_fp16_weights:
                if not transpose[0] and not transpose[1]:
                    B2 = B2.t().contiguous()

                state.CB, state.SCB, _ = bnb.functional.int8_vectorwise_quant(B2.to(torch.float16))
                B2 = state.CB

            if not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B2, state=state, bias=bias2)
            elif not transpose[0] and not transpose[1]:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B2.t(), state=state, bias=bias2)

            if has_bias:
                out_torch += bias

            assert out_bnb.dtype == A.dtype, f"bnb matmullt received {A.dtype} but returned {out_bnb.dtype}"

            n = out_bnb.numel()
            err = torch.abs(out_bnb - out_torch).mean().item()
            # print(f'abs error {err:.4f}')

            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() <= n * (0.0175 if dtype == torch.float16 else 0.021)
            idx = torch.isclose(out_bnb, out_torch, atol=0.035, rtol=0.2)
            assert (idx == 0).sum().item() <= n * 0.001

            if has_fp16_weights:
                if any(req_grad):
                    out_bnb.data.copy_(out_torch)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                    loss_bnb.backward()
                    gradA1 = A.grad
                    gradB1 = B.grad
                    A.grad = None
                    B.grad = None
                    if has_bias:
                        gradBias1 = bias.grad
                        bias.grad = None

                    loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                    loss_torch.backward()
                    gradA2 = A.grad
                    gradB2 = B.grad
                    A.grad = None
                    B.grad = None
                    if has_bias:
                        gradBias2 = bias.grad
                        bias.grad = None

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
                    assert (idx == 0).sum().item() <= n * 0.10

                    idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                    assert (idx == 0).sum().item() <= n * 0.02

                    torch.testing.assert_close(gradB1, gradB2, atol=0.18, rtol=0.3)

                if req_grad[2]:
                    torch.testing.assert_close(gradBias1, gradBias2)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dim1", [48], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [64, 0], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", [64], ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", [96], ids=id_formatter("dim4"))
@pytest.mark.parametrize("funcs", [(torch.matmul, bnb.matmul_4bit)], ids=["func=matmul"])
@pytest.mark.parametrize("req_grad", BOOLEAN_TRIPLES, ids=id_formatter("req_grad"))
@pytest.mark.parametrize("transpose", TRANSPOSE_VALS, ids=id_formatter("transpose"))
@pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
@pytest.mark.parametrize("quant_type", ["fp4", "nf4"], ids=id_formatter("quant_type"))
def test_matmul_4bit(
    device,
    dim1,
    dim2,
    dim3,
    dim4,
    funcs,
    dtype,
    req_grad,
    transpose,
    has_bias,
    compress_statistics,
    quant_type,
):
    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    if has_bias == False:
        req_grad = list(req_grad)
        req_grad[2] = False

    if device == "cpu" and dtype != torch.float32 and any(req_grad) and torch.__version__ < (2, 6):
        pytest.xfail("mse_loss fp16 on CPU is not supported in torch < 2.6")

    if device == "hpu" and not is_supported_on_hpu(quant_type, dtype):
        pytest.skip("This configuration is not supported on HPU.")

    for i in range(3):
        # normal multiply
        if funcs[0] in [torch.mm, torch.matmul]:
            A = torch.randn(size=dimA, device=device, requires_grad=req_grad[0], dtype=dtype)
            B = torch.randn(size=dimB, device=device, requires_grad=req_grad[1], dtype=dtype)
            target = torch.randn(size=(dim2, dim4), device=device, requires_grad=req_grad[1], dtype=dtype)
            bias = None
            bias2 = None
            if has_bias:
                bias = torch.randn(dim4, device=device, dtype=dtype, requires_grad=req_grad[2])
                bias2 = bias.clone()
            torch.nn.init.xavier_uniform_(B)

            B2, quant_state = bnb.functional.quantize_4bit(
                B,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
            )

            if not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B2.t(), quant_state, bias=bias2)
            elif not transpose[0] and not transpose[1]:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B2, quant_state, bias=bias2)

            if has_bias:
                out_torch += bias

            assert out_bnb.dtype == A.dtype, f"bnb matmullt received {A.dtype} but returned {out_bnb.dtype}"

            n = out_bnb.numel()
            err = torch.abs(out_bnb - out_torch).float().mean().item()
            if n > 0:
                assert err < 0.115

                # assert err < 0.20
            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                if device == "cuda":
                    torch.cuda.synchronize()
                elif device == "hpu":
                    torch.hpu.synchronize()

                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None
                if has_bias:
                    gradBias1 = bias.grad
                    bias.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None
                if has_bias:
                    gradBias2 = bias.grad
                    bias.grad = None

                if req_grad[0]:
                    torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)

                if req_grad[2]:
                    torch.testing.assert_close(gradBias1, gradBias2)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"], ids=id_formatter("quant_type"))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
def test_matmul_4bit_out_parameter(device, quant_type, dtype, has_bias):
    """Test that matmul_4bit(A, B, out=output) writes the result into output (issue #1235)."""
    M, K, N = 32, 64, 48

    # Create weight matrix (K, N) and quantize — matmul_4bit computes A @ dequant(B)
    W = torch.randn(K, N, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W)
    B_quant, quant_state = bnb.functional.quantize_4bit(W, quant_type=quant_type)

    bias = None
    if has_bias:
        bias = torch.randn(N, device=device, dtype=dtype)

    # --- Test 2D input (matrix path through MatMul4Bit) ---
    A_2d = torch.randn(M, K, device=device, dtype=dtype)
    expected = bnb.matmul_4bit(A_2d, B_quant, quant_state, bias=bias)

    out_2d = torch.zeros(M, N, device=device, dtype=dtype)
    returned = bnb.matmul_4bit(A_2d, B_quant, quant_state, out=out_2d, bias=bias)

    # out tensor should contain the result
    torch.testing.assert_close(out_2d, expected)
    # returned value should be the same object as out
    assert returned.data_ptr() == out_2d.data_ptr(), "returned tensor should share storage with out"

    # --- Test 1D input (gemv path) if on CUDA and blocksize divides K ---
    # Skip bias for 1D: the gemv path has a pre-existing shape bug with bias when K != N.
    if device == "cuda" and K % quant_state.blocksize == 0 and not has_bias:
        A_1d = torch.randn(K, device=device, dtype=dtype)
        expected_1d = bnb.matmul_4bit(A_1d, B_quant, quant_state)

        out_1d = torch.zeros_like(expected_1d)
        returned_1d = bnb.matmul_4bit(A_1d, B_quant, quant_state, out=out_1d)

        torch.testing.assert_close(out_1d, expected_1d)
        assert returned_1d.data_ptr() == out_1d.data_ptr(), "returned tensor should share storage with out"
