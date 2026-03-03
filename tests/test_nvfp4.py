"""Tests for NVFP4 (E2M1) quantization kernels.

Tests the NVFP4 quantize/dequantize, Hadamard rotation, and fused
rotate+quantize kernels via ctypes calls to the C library.
"""

import ctypes
import os

import pytest
import torch


def get_lib():
    """Load the bitsandbytes CUDA library."""
    lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bitsandbytes")
    # Try cuda131 first (built from nvcc 13.1), fall back to cuda130
    for suffix in ["cuda131", "cuda130"]:
        lib_path = os.path.join(lib_dir, f"libbitsandbytes_{suffix}.so")
        if os.path.exists(lib_path):
            return ctypes.cdll.LoadLibrary(lib_path)
    raise RuntimeError(f"Could not find bitsandbytes CUDA library in {lib_dir}")


def quantize_nvfp4(x, tensor_scale=None):
    """Quantize a FP16/BF16/FP32 tensor to NVFP4 using the C kernel."""
    lib = get_lib()
    n = x.numel()
    assert n % 16 == 0, "NVFP4 requires tensor size divisible by 16"

    if tensor_scale is None:
        tensor_scale = x.abs().max().item()

    packed = torch.zeros(n // 2, dtype=torch.uint8, device=x.device)
    block_scales = torch.zeros(n // 16, dtype=torch.uint8, device=x.device)

    if x.dtype == torch.float16:
        func = lib.cquantize_nvfp4_fp16
    elif x.dtype == torch.bfloat16:
        func = lib.cquantize_nvfp4_bf16
    elif x.dtype == torch.float32:
        func = lib.cquantize_nvfp4_fp32
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    func(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(block_scales.data_ptr()),
        ctypes.c_float(tensor_scale),
        ctypes.c_int(n),
    )
    torch.cuda.synchronize()
    return packed, block_scales, tensor_scale


def dequantize_nvfp4(packed, block_scales, tensor_scale, n, dtype=torch.float16):
    """Dequantize NVFP4 packed data back to FP16/BF16/FP32."""
    lib = get_lib()
    output = torch.zeros(n, dtype=dtype, device=packed.device)

    if dtype == torch.float16:
        func = lib.cdequantize_nvfp4_fp16
    elif dtype == torch.bfloat16:
        func = lib.cdequantize_nvfp4_bf16
    elif dtype == torch.float32:
        func = lib.cdequantize_nvfp4_fp32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    func(
        ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(block_scales.data_ptr()),
        ctypes.c_float(tensor_scale),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(n),
        ctypes.c_void_p(0),  # default stream
    )
    torch.cuda.synchronize()
    return output


def hadamard_rotate16(x):
    """Apply block-diagonal Had16 rotation in-place."""
    lib = get_lib()
    n = x.numel()
    assert n % 16 == 0, "Hadamard rotation requires size divisible by 16"

    if x.dtype == torch.float16:
        func = lib.chadamard_rotate16_fp16
    elif x.dtype == torch.bfloat16:
        func = lib.chadamard_rotate16_bf16
    elif x.dtype == torch.float32:
        func = lib.chadamard_rotate16_fp32
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    func(ctypes.c_void_p(x.data_ptr()), ctypes.c_int(n))
    torch.cuda.synchronize()


def fused_hadamard_quantize_nvfp4(x, tensor_scale=None):
    """Fused Hadamard rotation + NVFP4 quantization."""
    lib = get_lib()
    n = x.numel()
    assert n % 16 == 0

    if tensor_scale is None:
        # Need to compute tensor_scale on rotated data
        # Apply rotation to a copy to get the scale
        x_copy = x.clone()
        hadamard_rotate16(x_copy)
        tensor_scale = x_copy.abs().max().item()

    packed = torch.zeros(n // 2, dtype=torch.uint8, device=x.device)
    block_scales = torch.zeros(n // 16, dtype=torch.uint8, device=x.device)

    if x.dtype == torch.float16:
        func = lib.cfused_hadamard_quantize_nvfp4_fp16
    elif x.dtype == torch.bfloat16:
        func = lib.cfused_hadamard_quantize_nvfp4_bf16
    elif x.dtype == torch.float32:
        func = lib.cfused_hadamard_quantize_nvfp4_fp32
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    func(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(block_scales.data_ptr()),
        ctypes.c_float(tensor_scale),
        ctypes.c_int(n),
    )
    torch.cuda.synchronize()
    return packed, block_scales, tensor_scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestNVFP4Encoding:
    """Test the E2M1 encoding table and basic quantization."""

    def test_nvfp4_encoding_table(self):
        """Verify all 16 E2M1 codes produce correct values via round-trip."""
        # E2M1 representable magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
        test_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        x = torch.tensor(test_vals, dtype=torch.float16, device="cuda")

        # tensor_scale = 1.0 so block_scale = max(6)/6 = 1.0 (exactly E4M3)
        packed, scales, ts = quantize_nvfp4(x, tensor_scale=1.0)
        y = dequantize_nvfp4(packed, scales, ts, len(test_vals))

        for i, (inp, out) in enumerate(zip(test_vals, y.tolist())):
            assert abs(inp - out) < 0.01, f"E2M1 code {i}: expected {inp}, got {out}"

    def test_nvfp4_round_trip_error(self):
        """Verify round-trip error is within expected E2M1 bounds."""
        torch.manual_seed(42)
        n = 1024 * 16  # Multiple of 16
        x = torch.randn(n, dtype=torch.float16, device="cuda")

        packed, scales, ts = quantize_nvfp4(x)
        y = dequantize_nvfp4(packed, scales, ts, n)

        err = (x.float() - y.float()).abs()
        mean_err = err.mean().item()
        # E2M1 with blocksize 16 on standard normal data should have
        # mean absolute error roughly 0.05-0.10
        assert mean_err < 0.15, f"Mean abs error {mean_err:.4f} exceeds bound 0.15"
        assert mean_err > 0.01, f"Mean abs error {mean_err:.4f} suspiciously low"

    def test_nvfp4_two_level_scaling(self):
        """Verify tensor scale + block scale correctly recovers large values."""
        # Create data with values outside [-6, 6]
        torch.manual_seed(42)
        n = 256
        x = torch.randn(n, dtype=torch.float16, device="cuda") * 100.0

        packed, scales, ts = quantize_nvfp4(x)
        y = dequantize_nvfp4(packed, scales, ts, n)

        # Output should have roughly the same range as input
        assert y.abs().max().item() > 50.0, "Two-level scaling failed to preserve large magnitudes"

        # Relative error should be bounded
        mask = x.abs() > 10.0
        if mask.sum() > 0:
            rel_err = ((x[mask].float() - y[mask].float()).abs() / x[mask].abs().float()).mean().item()
            assert rel_err < 0.5, f"Relative error on large values: {rel_err:.4f}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_nvfp4_dtypes(self, dtype):
        """Verify quantization works for FP16 and BF16."""
        torch.manual_seed(42)
        n = 1024
        x = torch.randn(n, dtype=dtype, device="cuda")

        packed, scales, ts = quantize_nvfp4(x)
        y = dequantize_nvfp4(packed, scales, ts, n, dtype=dtype)

        assert y.dtype == dtype
        err = (x.float() - y.float()).abs().mean().item()
        assert err < 0.15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestHadamardRotation:
    """Test the block-diagonal Had16 rotation kernel."""

    def test_hadamard_orthogonality(self):
        """Applying Hadamard twice should return the original (H*H^T = I)."""
        torch.manual_seed(42)
        n = 1024
        x = torch.randn(n, dtype=torch.float16, device="cuda")
        x_orig = x.clone()

        hadamard_rotate16(x)
        hadamard_rotate16(x)

        err = (x.float() - x_orig.float()).abs().max().item()
        assert err < 0.01, f"Double rotation max error {err:.6f} exceeds FP16 tolerance"

    def test_hadamard_reduces_kurtosis(self):
        """Hadamard rotation should make Laplace-distributed data more Gaussian."""
        torch.manual_seed(123)
        n = 4096

        # Generate Laplace distribution (kurtosis ~6)
        e1 = torch.empty(n, device="cuda").exponential_(1.0)
        e2 = torch.empty(n, device="cuda").exponential_(1.0)
        lap = (e1 - e2).half()

        def kurtosis(t):
            t = t.float()
            m = t.mean()
            return ((t - m) ** 4).mean() / ((t - m) ** 2).mean() ** 2

        kurt_before = kurtosis(lap).item()

        lap_rot = lap.clone()
        hadamard_rotate16(lap_rot)

        kurt_after = kurtosis(lap_rot).item()

        assert kurt_after < kurt_before, f"Kurtosis increased: {kurt_before:.2f} -> {kurt_after:.2f}"
        # After rotation, kurtosis should be closer to 3 (Gaussian)
        assert kurt_after < 4.0, f"Post-rotation kurtosis {kurt_after:.2f} too high (expected < 4.0)"

    def test_hadamard_preserves_norm(self):
        """Hadamard rotation should preserve L2 norm (orthogonal transform)."""
        torch.manual_seed(42)
        n = 1024
        x = torch.randn(n, dtype=torch.float32, device="cuda")
        norm_before = x.norm().item()

        hadamard_rotate16(x)
        norm_after = x.norm().item()

        rel_err = abs(norm_before - norm_after) / norm_before
        assert rel_err < 0.001, f"Norm changed by {rel_err:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFusedHadamardQuantize:
    """Test the fused Had16 + NVFP4 quantize kernel."""

    def test_fused_matches_sequential(self):
        """Fused kernel output should match sequential rotate+quantize."""
        torch.manual_seed(42)
        n = 1024
        x = torch.randn(n, dtype=torch.float16, device="cuda")

        # Sequential: rotate, then quantize
        x_seq = x.clone()
        hadamard_rotate16(x_seq)
        ts = x_seq.abs().max().item()
        packed_seq, scales_seq, _ = quantize_nvfp4(x_seq, tensor_scale=ts)

        # Fused: single kernel
        packed_fused, scales_fused, _ = fused_hadamard_quantize_nvfp4(x.clone(), tensor_scale=ts)

        assert torch.equal(packed_seq, packed_fused), "Packed data mismatch"
        assert torch.equal(scales_seq, scales_fused), "Block scales mismatch"

    def test_fused_quantization_error_bounded(self):
        """Fused rotation+quantization should produce bounded error."""
        torch.manual_seed(42)
        n = 4096

        # Laplace-distributed data (outlier-heavy)
        e1 = torch.empty(n, device="cuda").exponential_(1.0)
        e2 = torch.empty(n, device="cuda").exponential_(1.0)
        x = (e1 - e2).half()

        # With rotation (fused)
        packed_r, scales_r, ts_r = fused_hadamard_quantize_nvfp4(x)
        y_r = dequantize_nvfp4(packed_r, scales_r, ts_r, n)
        # Inverse rotation to get back to original domain
        hadamard_rotate16(y_r)
        err_rot = (x.float() - y_r.float()).abs().mean().item()

        # Error should be bounded (FP4 on Laplace data, including inverse rotation noise)
        assert err_rot < 0.2, f"Fused quantization error {err_rot:.4f} exceeds bound 0.2"
        assert err_rot > 0.01, f"Fused quantization error {err_rot:.4f} suspiciously low"
