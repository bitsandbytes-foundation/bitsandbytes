from math import prod

import pytest
import torch

import bitsandbytes
from bitsandbytes.cextension import HIP_ENVIRONMENT
from bitsandbytes.functional import ipex_xpu
from tests.helpers import TRUE_FALSE, get_available_devices, id_formatter, is_supported_on_hpu

# torch.library.opcheck is only available in torch 2.4 and later.
# When testing with older versions, we will skip it as a no-op.
if torch.__version__ >= (2, 4):
    opcheck = torch.library.opcheck
else:
    opcheck = lambda *args, **kwargs: None


class TestLLMInt8Ops:
    @pytest.mark.parametrize("device", get_available_devices())
    def test_int8_linear_matmul(self, device):
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8, device=device)
        B = torch.randint(-128, 127, (30, 20), dtype=torch.int8, device=device)
        out = torch.ops.bitsandbytes.int8_linear_matmul.default(A, B)

        assert out.shape == (10, 30)
        assert out.dtype == torch.int32
        assert out.device == A.device

        opcheck(torch.ops.bitsandbytes.int8_linear_matmul.default, (A, B))

    @pytest.mark.parametrize("device", get_available_devices())
    def test_int8_linear_matmul_out(self, device):
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8, device=device)
        B = torch.randint(-128, 127, (30, 20), dtype=torch.int8, device=device)

        out = torch.empty((10, 30), dtype=torch.int32, device=device)
        torch.ops.bitsandbytes.int8_linear_matmul.out(A, B, out)

        assert out.shape == (10, 30)
        assert out.dtype == torch.int32
        assert out.device == A.device

        opcheck(torch.ops.bitsandbytes.int8_linear_matmul.out, (A, B, out))

    @pytest.mark.parametrize("threshold", [0.0, 6.0])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_int8_vectorwise_quant(self, threshold, device):
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

        opcheck(torch.ops.bitsandbytes.int8_vectorwise_quant, (A,))
        opcheck(torch.ops.bitsandbytes.int8_vectorwise_quant, (A, threshold))

    @pytest.mark.parametrize("device", get_available_devices())
    def test_int8_mm_dequant(self, device):
        A = torch.randint(-128, 127, (256, 256), dtype=torch.int32, device=device)
        row_stats = torch.randn(256, dtype=torch.float32, device=device)
        col_stats = torch.randn(256, dtype=torch.float32, device=device)
        out = torch.ops.bitsandbytes.int8_mm_dequant(A, row_stats, col_stats)

        assert out.shape == A.shape
        assert out.dtype == torch.float16
        assert out.device == A.device

        opcheck(torch.ops.bitsandbytes.int8_mm_dequant, (A, row_stats, col_stats))

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("has_bias", TRUE_FALSE)
    def test_int8_scaled_mm(self, device, dtype, has_bias):
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8, device=device)
        B = torch.randint(-128, 127, (30, 20), dtype=torch.int8, device=device)
        row_stats = torch.randn(10, dtype=torch.float32, device=device)
        col_stats = torch.randn(30, dtype=torch.float32, device=device)
        bias = torch.randn(30, dtype=dtype, device=device) if has_bias else None
        out = torch.ops.bitsandbytes.int8_scaled_mm(A, B, row_stats, col_stats, bias=bias, dtype=dtype)

        assert out.shape == (10, 30)
        assert out.dtype == dtype
        assert out.device == A.device

        opcheck(torch.ops.bitsandbytes.int8_scaled_mm, (A, B, row_stats, col_stats, bias, dtype))


class TestInt8BlockwiseQuantOps:
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512] if not HIP_ENVIRONMENT else [128, 256, 512])
    def test_quantize_blockwise(self, device, dtype, blocksize):
        if device == "cpu":
            if dtype != torch.float32:
                pytest.skip("CPU implementation is only available for float32")

            if blocksize != 256:
                pytest.skip("CPU implementation is slow; only test blocksize=256")

        code = bitsandbytes.functional.create_dynamic_map().to(device)
        A = torch.randn(1024, 1024, dtype=dtype, device=device)
        out, absmax = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)

        assert out.shape == A.shape
        assert out.dtype == torch.uint8
        assert out.device == A.device

        assert absmax.device == A.device
        assert absmax.dtype == torch.float32

        opcheck(torch.ops.bitsandbytes.quantize_blockwise, (A, code, blocksize))

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512] if not HIP_ENVIRONMENT else [128, 256, 512])
    def test_dequantize_blockwise(self, device, dtype, blocksize):
        if device == "cpu" and dtype != torch.float32:
            pytest.skip("CPU implementation is only available for float32")

        A = torch.randint(0, 255, (1024, 1024), dtype=torch.uint8, device=device)
        code = bitsandbytes.functional.create_dynamic_map().to(device, dtype=torch.float32)

        n = A.numel()
        blocks = -(n // -blocksize)
        absmax = torch.randn((blocks,), device=device, dtype=torch.float32)

        out = torch.ops.bitsandbytes.dequantize_blockwise.default(A, absmax, code, blocksize, dtype)

        assert out.shape == A.shape
        assert out.dtype == dtype
        assert out.device == A.device

        # TODO: Enable it
        if device == "xpu" and ipex_xpu:
            pytest.skip("XPU implementation have torch.op inside torch.op, it will fail on op check")

        opcheck(torch.ops.bitsandbytes.dequantize_blockwise.default, (A, absmax, code, blocksize, dtype))


class Test4bitBlockwiseQuantOps:
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage_dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512] if not HIP_ENVIRONMENT else [128, 256, 512])
    def test_quantize_4bit(self, device, dtype, storage_dtype, quant_type, blocksize):
        if device == "hpu" and not is_supported_on_hpu(quant_type, dtype, storage_dtype):
            pytest.skip("This configuration is not supported on HPU.")

        A = torch.randn(1024, 1024, dtype=dtype, device=device)

        out, absmax = torch.ops.bitsandbytes.quantize_4bit.default(A, blocksize, quant_type, storage_dtype)

        assert out.device == A.device
        assert out.dtype == storage_dtype

        assert absmax.device == A.device
        assert absmax.dtype == torch.float32

        if storage_dtype != torch.uint8:
            pytest.xfail("opcheck fails for storage_dtype != torch.uint8")

        opcheck(torch.ops.bitsandbytes.quantize_4bit.default, (A, blocksize, quant_type, storage_dtype))

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage_dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512] if not HIP_ENVIRONMENT else [128, 256, 512])
    def test_dequantize_4bit(self, device, dtype, storage_dtype, quant_type, blocksize):
        if device == "hpu" and not is_supported_on_hpu(quant_type, dtype, storage_dtype):
            pytest.skip("This configuration is not supported on HPU.")

        shape = (128, 128)

        n = prod(shape)
        blocks = -(n // -blocksize)
        quantized_shape = ((n + 1) // (storage_dtype.itemsize * 2), 1)

        A = (
            torch.randint(0, 255, ((n + 1) // 2,), dtype=torch.uint8, device=device)
            .view(storage_dtype)
            .reshape(quantized_shape)
            .contiguous()
        )

        absmax = torch.randn((blocks,), dtype=torch.float32, device=device)

        out = torch.ops.bitsandbytes.dequantize_4bit.default(A, absmax, blocksize, quant_type, shape, dtype)

        assert out.device == A.device
        assert out.shape == shape

        opcheck(
            torch.ops.bitsandbytes.dequantize_4bit.default,
            (A, absmax, blocksize, quant_type, shape, dtype),
        )

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage_dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512] if not HIP_ENVIRONMENT else [128, 256, 512])
    def test_gemv_4bit(self, device, dtype, storage_dtype, quant_type, blocksize):
        if device == "hpu" and not is_supported_on_hpu(quant_type, dtype, storage_dtype):
            pytest.skip("This configuration is not supported on HPU.")

        out_features = 1024
        in_features = 256

        A = torch.randn((1, 1, in_features), dtype=dtype, device=device)
        B = torch.randn((out_features, in_features), dtype=dtype, device=A.device)
        B_q, absmax = torch.ops.bitsandbytes.quantize_4bit(B, blocksize, quant_type, storage_dtype)
        code = bitsandbytes.functional.get_4bit_type(quant_type, device=A.device, blocksize=blocksize)

        out = torch.ops.bitsandbytes.gemv_4bit.default(A, B_q, B.shape, absmax, code, blocksize)

        assert out.device == A.device
        assert out.dtype == dtype
        assert out.shape == (1, 1, out_features)
        assert out.isreal().all()

        opcheck(torch.ops.bitsandbytes.gemv_4bit.default, (A, B_q, B.shape, absmax, code, blocksize))
