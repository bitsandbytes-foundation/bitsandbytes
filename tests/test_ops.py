import pytest
import torch

import bitsandbytes


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_int8_linear_matmul(device):
    A = torch.randint(-128, 127, (10, 20), dtype=torch.int8, device=device)
    B = torch.randint(-128, 127, (30, 20), dtype=torch.int8, device=device)
    out = torch.ops.bitsandbytes.int8_linear_matmul(A, B)

    assert out.shape == (10, 30)
    assert out.dtype == torch.int32
    assert out.device == A.device

    torch.library.opcheck(torch.ops.bitsandbytes.int8_linear_matmul, (A, B))


@pytest.mark.parametrize("threshold", [0.0, 6.0])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_int8_vectorwise_quant(threshold, device):
    if device == "cpu":
        pytest.skip("CPU implementation is not available")

    A = torch.randn(10, 20, dtype=torch.float16, device=device)
    A[1][0] = 1000.0

    out_row, row_stats, outlier_cols = torch.ops.bitsandbytes.int8_vectorwise_quant(A, threshold=threshold)

    assert out_row.shape == (10, 20)
    assert out_row.dtype == torch.int8
    assert out_row.device == A.device
    assert row_stats.shape == (10,)
    assert row_stats.dtype == torch.float32
    assert row_stats.device == A.device

    if threshold > 0.0:
        assert outlier_cols is not None
        assert outlier_cols.dim() == 1
        assert outlier_cols.shape[0] <= A.shape[1]
        assert outlier_cols.device == A.device
    else:
        assert outlier_cols is None

    torch.library.opcheck(torch.ops.bitsandbytes.int8_vectorwise_quant, (A,))

    torch.library.opcheck(torch.ops.bitsandbytes.int8_vectorwise_quant, (A, threshold))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_int8_mm_dequant(device):
    if device == "cpu":
        pytest.skip("CPU implementation is not available")

    A = torch.randint(-128, 127, (10, 20), dtype=torch.int32, device=device)
    row_stats = torch.randn(10, dtype=torch.float16, device=device)
    col_stats = torch.randn(20, dtype=torch.float16, device=device)
    out = torch.ops.bitsandbytes.int8_mm_dequant(A, row_stats, col_stats)

    assert out.shape == A.shape
    assert out.dtype == torch.float16
    assert out.device == A.device

    torch.library.opcheck(torch.ops.bitsandbytes.int8_mm_dequant, (A, row_stats, col_stats))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quantize_blockwise(device):
    # if device == "cpu":
    #     pytest.skip("CPU implementation is not available")
    blocksize = 256

    code = bitsandbytes.functional.create_dynamic_map().to(device)
    A = torch.randn(1024, 1024, dtype=torch.float16, device=device)
    out, absmax = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)

    assert out.shape == A.shape
    assert out.dtype == torch.uint8
    assert out.device == A.device

    assert absmax.device == A.device
    assert absmax.dtype == torch.float32

    torch.library.opcheck(torch.ops.bitsandbytes.quantize_blockwise, (A, code, blocksize))
