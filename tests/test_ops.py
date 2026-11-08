from math import prod
from unittest.mock import Mock

import pytest
import torch

import bitsandbytes
from tests.helpers import TRUE_FALSE, describe_dtype, get_available_devices, id_formatter, is_supported_on_hpu

opcheck = torch.library.opcheck


def _dequantize_4bit_reference(packed, absmax, blocksize, quant_type, shape, dtype):
    # Decode on CPU independently of the quantizer and all backend dequantizers.
    raw = packed.cpu().contiguous().view(torch.uint8).reshape(-1)
    codes = torch.stack((raw >> 4, raw & 0xF), dim=1).flatten()[: prod(shape)].long()
    codebook = bitsandbytes.functional.get_4bit_type(quant_type, device="cpu")
    scales = absmax.cpu().repeat_interleave(blocksize)[: prod(shape)]
    return (codebook[codes] * scales).reshape(shape).to(dtype)


@pytest.fixture(params=["hub", "fallback"])
def mps_dequantize_backend(request, monkeypatch):
    if "mps" not in get_available_devices() or not torch.backends.mps.is_available():
        pytest.skip("MPS is not available or not selected for this run")

    from bitsandbytes.backends.mps import ops

    # Restore loader state too: _get_kernel() can mutate both globals.
    monkeypatch.setattr(ops, "_kernel", ops._kernel)
    monkeypatch.setattr(ops, "_kernel_load_failed", ops._kernel_load_failed)
    kernel = ops._get_kernel()
    if request.param == "hub" and kernel is None:
        if request.config.getoption("--require-mps-hub"):
            pytest.fail("The required MPS Hub kernel is unavailable")
        pytest.skip("The bitsandbytes-mps Hub kernel is unavailable (requires macOS 26+ and kernels)")

    if kernel is not None:
        calls = Mock(wraps=kernel.dequantize_4bit)
        monkeypatch.setattr(kernel, "dequantize_4bit", calls)
    else:
        calls = Mock(side_effect=AssertionError("Hub dequantization must not run in fallback mode"))
        monkeypatch.setattr(ops, "_kernel", Mock(dequantize_4bit=calls))
    if request.param == "fallback":
        monkeypatch.setattr(ops, "_kernel_load_failed", True)
    return request.param, calls


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
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512])
    def test_quantize_blockwise(self, device, dtype, blocksize):
        if device == "cpu" and blocksize != 256:
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
    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512])
    def test_dequantize_blockwise(self, device, dtype, blocksize):
        A = torch.randint(0, 255, (1024, 1024), dtype=torch.uint8, device=device)
        code = bitsandbytes.functional.create_dynamic_map().to(device, dtype=torch.float32)

        n = A.numel()
        blocks = -(n // -blocksize)
        absmax = torch.randn((blocks,), device=device, dtype=torch.float32)

        out = torch.ops.bitsandbytes.dequantize_blockwise.default(A, absmax, code, blocksize, dtype)

        assert out.shape == A.shape
        assert out.dtype == dtype
        assert out.device == A.device

        opcheck(torch.ops.bitsandbytes.dequantize_blockwise.default, (A, absmax, code, blocksize, dtype))


class Test4bitBlockwiseQuantOps:
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage_dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [32, 64, 128, 256, 512])
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
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256])
    def test_quantize_4bit_not_divisible_by_blocksize(self, device, dtype, quant_type, blocksize):
        """Test quantize/dequantize roundtrip when n_elements is not divisible by blocksize."""
        # Shape chosen so numel is NOT divisible by blocksize
        shape = (7, blocksize - 1)
        A = torch.randn(shape, dtype=dtype, device=device)
        storage_dtype = torch.uint8

        # Should not raise
        packed, absmax = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, storage_dtype)

        assert packed.device == A.device
        assert absmax.device == A.device

        # Dequantize back and verify shape is preserved
        out = torch.ops.bitsandbytes.dequantize_4bit(packed, absmax, blocksize, quant_type, shape, dtype)

        assert out.shape == shape
        assert out.dtype == dtype

        # Verify output is finite (no NaN/Inf)
        assert torch.isfinite(out).all(), "Dequantized output contains NaN or Inf"

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage_dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [32, 64, 128, 256, 512])
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
        assert out.dtype == dtype
        expected = _dequantize_4bit_reference(A, absmax, blocksize, quant_type, shape, dtype)
        torch.testing.assert_close(out.cpu(), expected)

        opcheck(
            torch.ops.bitsandbytes.dequantize_4bit.default,
            (A, absmax, blocksize, quant_type, shape, dtype),
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize(
        ("blocksize", "shape", "storage_dtype"),
        [
            (64, (4, 64), torch.uint8),
            (128, (4, 128), torch.bfloat16),
            (256, (3, 257), torch.bfloat16),
            (512, (3, 513), torch.uint8),
            (32, (3, 33), torch.bfloat16),
        ],
        ids=["block64-aligned", "block128-aligned", "block256-tail", "block512-tail", "block32-fallback"],
    )
    def test_dequantize_4bit_mps(
        self, mps_dequantize_backend, record_property, dtype, quant_type, blocksize, shape, storage_dtype
    ):
        mode, hub_calls = mps_dequantize_backend
        hub_calls.reset_mock()
        indices = torch.arange((prod(shape) + 1) // 2)
        # Exercise every nibble code, including different high/low nibbles.
        raw = (((indices % 16) << 4) | ((indices * 7 + 3) % 16)).to(torch.uint8)
        packed = raw.view(storage_dtype).reshape(-1, 1).to("mps")
        absmax = torch.tensor([0.0, 0.37, 1.125, 2.75], device="mps")
        expected = _dequantize_4bit_reference(packed, absmax, blocksize, quant_type, shape, dtype)

        actual = torch.ops.bitsandbytes.dequantize_4bit.default(packed, absmax, blocksize, quant_type, shape, dtype)
        assert actual.device == packed.device
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

        out = torch.empty((shape[1], shape[0]), device="mps", dtype=dtype).T
        pointer = out.data_ptr()
        torch.ops.bitsandbytes.dequantize_4bit.out(packed, absmax, blocksize, quant_type, shape, dtype, out=out)
        assert out.data_ptr() == pointer
        assert not out.is_contiguous()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        assert hub_calls.call_count == (2 if mode == "hub" and blocksize != 32 else 0)
        if mode == "hub" and blocksize != 32:
            record_property("mps_hub_verified", True)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage_dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [32, 64, 128, 256, 512])
    def test_gemv_4bit(self, device, dtype, storage_dtype, quant_type, blocksize):
        if device == "hpu" and not is_supported_on_hpu(quant_type, dtype, storage_dtype):
            pytest.skip("This configuration is not supported on HPU.")

        out_features = 1024
        in_features = 256

        if device in ("cpu", "mps") and blocksize > in_features:
            pytest.skip("CPU/MPS implementation only supports blocksize <= in_features")

        A = torch.randn((1, 1, in_features), dtype=dtype, device=device)
        B = torch.randn((out_features, in_features), dtype=dtype, device=A.device)
        B_q, absmax = torch.ops.bitsandbytes.quantize_4bit(B, blocksize, quant_type, storage_dtype)
        code = bitsandbytes.functional.get_4bit_type(quant_type, device=A.device, blocksize=blocksize)

        if device == "cpu" and bitsandbytes.functional.has_avx512bf16():
            state = bitsandbytes.functional.QuantState(
                absmax=absmax,
                shape=B.shape,
                dtype=A.dtype,
                blocksize=blocksize,
                code=code,
                quant_type=quant_type,
            )
            B_q, state = bitsandbytes.functional._convert_weight_packed_for_cpu(B_q, state)
            absmax = state.absmax
        out = torch.ops.bitsandbytes.gemv_4bit.default(A, B_q, B.shape, absmax, code, blocksize)

        assert out.device == A.device
        assert out.dtype == dtype
        assert out.shape == (1, 1, out_features)
        assert out.isreal().all()

        opcheck(torch.ops.bitsandbytes.gemv_4bit.default, (A, B_q, B.shape, absmax, code, blocksize))

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("requires_grad", TRUE_FALSE, ids=id_formatter("requires_grad"))
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage_dtype"))
    @pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
    @pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
    def test_gemm_4bit(self, device, dtype, quant_type, compress_statistics, has_bias, storage_dtype, requires_grad):
        if device == "hpu" and not is_supported_on_hpu(quant_type, dtype, storage_dtype):
            pytest.skip("This configuration is not supported on HPU.")

        N, K, blocksize = 64, 64, 64

        A = torch.randn(2, 2, K, dtype=dtype, device=device, requires_grad=requires_grad)
        B = torch.randn(N, K, dtype=dtype, device=device)
        B_q, qs = bitsandbytes.functional.quantize_4bit(
            B,
            blocksize=blocksize,
            quant_type=quant_type,
            compress_statistics=compress_statistics,
            quant_storage=storage_dtype,
        )
        bias = torch.randn(N, dtype=dtype, device=device) if has_bias else None

        if compress_statistics:
            out = torch.ops.bitsandbytes.gemm_4bit.default(
                A,
                B_q,
                list(B.shape),
                qs.state2.absmax,
                blocksize,
                quant_type,
                bias=bias,
                absmax_8bit=qs.absmax,
                absmax_code=qs.state2.code,
                absmax_offset=qs.offset,
            )
        else:
            out = torch.ops.bitsandbytes.gemm_4bit.default(
                A,
                B_q,
                list(B.shape),
                qs.absmax,
                blocksize,
                quant_type,
                bias=bias,
            )

        assert out.shape == (2, 2, N)
        assert out.dtype == dtype
        assert out.device.type == A.device.type
        assert out.isreal().all()

        # TODO: remove detach when register_autograd is added for gemm_4bit.
        # opcheck requires no autograd; detach A to skip the registration check.
        A_op = A.detach()
        if compress_statistics:
            opcheck(
                torch.ops.bitsandbytes.gemm_4bit.default,
                (A_op, B_q, list(B.shape), qs.state2.absmax, blocksize, quant_type),
                kwargs={
                    "bias": bias,
                    "absmax_8bit": qs.absmax,
                    "absmax_code": qs.state2.code,
                    "absmax_offset": qs.offset,
                },
            )
        else:
            opcheck(
                torch.ops.bitsandbytes.gemm_4bit.default,
                (A_op, B_q, list(B.shape), qs.absmax, blocksize, quant_type),
                kwargs={"bias": bias},
            )

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=describe_dtype)
    @pytest.mark.parametrize("offset_dtype", [torch.float16, torch.bfloat16], ids=describe_dtype)
    def test_gemm_4bit_non_float32_offset(self, device, dtype, offset_dtype):
        """Regression test: offset tensors not in float32 must still produce correct results."""
        N, K, blocksize = 64, 64, 64
        A = torch.randn(4, K, dtype=dtype, device=device)
        B = torch.randn(N, K, dtype=dtype, device=device)
        B_q, qs = bitsandbytes.functional.quantize_4bit(
            B, blocksize=blocksize, quant_type="nf4", compress_statistics=True
        )

        # Simulate a pre-quantized model where offset may not be float32.
        offset_non_f32 = qs.offset.to(dtype=offset_dtype)

        # Reference: explicitly use the rounded float32 value.
        offset_as_f32 = offset_non_f32.to(dtype=torch.float32)
        ref = torch.ops.bitsandbytes.gemm_4bit.default(
            A,
            B_q,
            list(B.shape),
            qs.state2.absmax,
            blocksize,
            "nf4",
            absmax_8bit=qs.absmax,
            absmax_code=qs.state2.code,
            absmax_offset=offset_as_f32,
        )
        out = torch.ops.bitsandbytes.gemm_4bit.default(
            A,
            B_q,
            list(B.shape),
            qs.state2.absmax,
            blocksize,
            "nf4",
            absmax_8bit=qs.absmax,
            absmax_code=qs.state2.code,
            absmax_offset=offset_non_f32,
        )
        torch.testing.assert_close(out, ref)


class TestNonContiguousInputs:
    """Regression tests for #1342 and #1690: quantization must handle non-contiguous tensors correctly."""

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("blocksize", [64, 128, 256])
    def test_quantize_blockwise_non_contiguous(self, device, dtype, blocksize):
        code = bitsandbytes.functional.create_dynamic_map().to(device)

        # Create non-contiguous tensor via slicing
        A_full = torch.randn(3, 4, 6, 256, dtype=dtype, device=device)
        A_noncontig = A_full[:, ::2, :, :]
        assert not A_noncontig.is_contiguous()

        A_contig = A_noncontig.contiguous()

        out_nc, absmax_nc = torch.ops.bitsandbytes.quantize_blockwise(A_noncontig, code, blocksize)
        out_c, absmax_c = torch.ops.bitsandbytes.quantize_blockwise(A_contig, code, blocksize)

        torch.testing.assert_close(absmax_nc, absmax_c)
        torch.testing.assert_close(out_nc, out_c)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("blocksize", [64, 128, 256])
    def test_dequantize_blockwise_non_contiguous(self, device, dtype, blocksize):
        code = bitsandbytes.functional.create_dynamic_map().to(device, dtype=torch.float32)

        # Quantize a contiguous tensor, then create non-contiguous uint8 via transpose
        A = torch.randn(1024, 1024, dtype=dtype, device=device)
        quantized, absmax = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)

        # Create non-contiguous uint8 tensor by transposing and transposing back
        q_noncontig = quantized.t().t()
        # If that's still contiguous, use a different approach
        if q_noncontig.is_contiguous():
            # Pad and slice to force non-contiguity
            q_padded = torch.zeros(1024, 1025, dtype=torch.uint8, device=device)
            q_padded[:, :1024] = quantized
            q_noncontig = q_padded[:, :1024]

        assert not q_noncontig.is_contiguous()
        q_contig = q_noncontig.contiguous()

        out_nc = torch.ops.bitsandbytes.dequantize_blockwise(q_noncontig, absmax, code, blocksize, dtype)
        out_c = torch.ops.bitsandbytes.dequantize_blockwise(q_contig, absmax, code, blocksize, dtype)

        torch.testing.assert_close(out_nc, out_c)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256])
    def test_quantize_4bit_non_contiguous(self, device, dtype, quant_type, blocksize):
        if device not in ("cuda", "mps"):
            pytest.skip("Non-contiguous input handling not implemented for this backend")

        # Reproduce issue #1342: non-contiguous tensor from slicing
        A_full = torch.randn(3, 4, 6, 256, dtype=dtype, device=device)
        A_noncontig = A_full[:, ::2, :, :]
        assert not A_noncontig.is_contiguous()

        A_contig = A_noncontig.contiguous()
        storage_dtype = torch.uint8

        out_nc, absmax_nc = torch.ops.bitsandbytes.quantize_4bit(A_noncontig, blocksize, quant_type, storage_dtype)
        out_c, absmax_c = torch.ops.bitsandbytes.quantize_4bit(A_contig, blocksize, quant_type, storage_dtype)

        torch.testing.assert_close(absmax_nc, absmax_c)
        torch.testing.assert_close(out_nc, out_c)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=id_formatter("dtype"))
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256])
    def test_quantize_4bit_roundtrip_non_contiguous(self, device, dtype, quant_type, blocksize):
        """End-to-end test: quantize non-contiguous, dequantize, compare with contiguous path."""
        if device not in ("cuda", "mps"):
            pytest.skip("Non-contiguous input handling not implemented for this backend")

        A_full = torch.randn(3, 4, 6, 256, dtype=dtype, device=device)
        A_noncontig = A_full[:, ::2, :, :]
        assert not A_noncontig.is_contiguous()

        A_contig = A_noncontig.contiguous()
        storage_dtype = torch.uint8

        # Quantize both
        q_nc, absmax_nc = torch.ops.bitsandbytes.quantize_4bit(A_noncontig, blocksize, quant_type, storage_dtype)
        q_c, absmax_c = torch.ops.bitsandbytes.quantize_4bit(A_contig, blocksize, quant_type, storage_dtype)

        # Dequantize both
        shape = A_contig.shape
        deq_nc = torch.ops.bitsandbytes.dequantize_4bit(q_nc, absmax_nc, blocksize, quant_type, shape, dtype)
        deq_c = torch.ops.bitsandbytes.dequantize_4bit(q_c, absmax_c, blocksize, quant_type, shape, dtype)

        torch.testing.assert_close(deq_nc, deq_c)
