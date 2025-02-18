from math import prod

import pytest
import torch

import bitsandbytes
from tests.helpers import id_formatter


class TestLLMInt8Ops:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_int8_linear_matmul(self, device):
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8, device=device)
        B = torch.randint(-128, 127, (30, 20), dtype=torch.int8, device=device)
        out = torch.ops.bitsandbytes.int8_linear_matmul(A, B)

        assert out.shape == (10, 30)
        assert out.dtype == torch.int32
        assert out.device == A.device

        torch.library.opcheck(torch.ops.bitsandbytes.int8_linear_matmul, (A, B))

    @pytest.mark.parametrize("threshold", [0.0, 6.0])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_int8_vectorwise_quant(self, threshold, device):
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
    def test_int8_mm_dequant(self, device):
        if device == "cpu":
            pytest.skip("CPU implementation is not available")

        A = torch.randint(-128, 127, (256, 256), dtype=torch.int32, device=device)
        row_stats = torch.randn(256, dtype=torch.float32, device=device)
        col_stats = torch.randn(256, dtype=torch.float32, device=device)
        out = torch.ops.bitsandbytes.int8_mm_dequant(A, row_stats, col_stats)

        assert out.shape == A.shape
        assert out.dtype == torch.float16
        assert out.device == A.device

        torch.library.opcheck(torch.ops.bitsandbytes.int8_mm_dequant, (A, row_stats, col_stats))

    def test_int8_double_quant():
        pass


class TestInt8BlockwiseQuantOps:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512])
    def test_quantize_blockwise(self, device, dtype, blocksize):
        if device == "cpu" and dtype != torch.float32:
            pytest.skip("CPU implementation is only available for float32")

        code = bitsandbytes.functional.create_dynamic_map().to(device)
        A = torch.randn(1024, 1024, dtype=dtype, device=device)
        out, absmax = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)

        assert out.shape == A.shape
        assert out.dtype == torch.uint8
        assert out.device == A.device

        assert absmax.device == A.device
        assert absmax.dtype == torch.float32

        torch.library.opcheck(torch.ops.bitsandbytes.quantize_blockwise, (A, code, blocksize))

    def test_dequantize_blockwise():
        pass


class Test4bitBlockwiseQuantOps:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage_dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512])
    def test_quantize_4bit(self, device, dtype, storage_dtype, quant_type, blocksize):
        if device == "cpu":
            pytest.skip("CPU implementation is not available")

        A = torch.randn(1024, 1024, dtype=dtype, device=device)

        out, absmax = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, storage_dtype)

        assert out.device == A.device
        assert out.dtype == storage_dtype

        assert absmax.device == A.device
        assert absmax.dtype == torch.float32

        torch.library.opcheck(torch.ops.bitsandbytes.quantize_4bit, (A, blocksize, quant_type, storage_dtype))

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage_dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512])
    def test_dequantize_4bit(self, device, dtype, storage_dtype, quant_type, blocksize):
        if device == "cpu":
            pytest.skip("CPU implementation is not available")

        shape = (128, 128)

        n = prod(shape)
        blocks = -(n // -blocksize)
        quantized_shape = ((n + 1) // (storage_dtype.itemsize * 2), 1)

        A = (
            torch.randint(0, 255, ((n + 1) // 2,), dtype=torch.uint8, device=device)
            .view(storage_dtype)
            .reshape(quantized_shape)
        )

        absmax = torch.randn((blocks,), dtype=torch.float32, device=device)

        out = torch.ops.bitsandbytes.dequantize_4bit(A, absmax, blocksize, quant_type, shape, dtype)

        assert out.device == A.device
        assert out.shape == shape

        torch.library.opcheck(torch.ops.bitsandbytes.dequantize_4bit, (A, absmax, blocksize, quant_type, shape, dtype))
