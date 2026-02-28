"""Tests for CUTLASS-based fused quantize (QuTLASS integration).

Tests the fused quantize path that uses CUTLASS GEMM with always-on
randomized Hadamard rotation for NVFP4 quantization.
"""

import pytest
import torch

from bitsandbytes.functional import (
    _has_cutlass_fused_quantize,
    dequantize_nvfp4,
    quantize_nvfp4,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        not _has_cutlass_fused_quantize(),
        reason="CUTLASS fused quantize not available (requires SM_120+)",
    ),
]


class TestFusedQuantizeRoundTrip:
    """Test fused quantize with always-on Hadamard rotation."""

    def test_round_trip_error_bounded(self):
        """Fused quantize round-trip error should be bounded."""
        torch.manual_seed(42)
        A = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
        packed, state = quantize_nvfp4(A)
        deq = dequantize_nvfp4(packed, state)
        err = (deq - A).abs().mean() / A.abs().mean()
        assert err < 0.12, f"Round-trip error {err:.4f} exceeds 12%"
        assert err > 0.01, f"Round-trip error {err:.4f} suspiciously low"

    def test_output_shapes(self):
        """Verify output tensor shapes are correct."""
        torch.manual_seed(42)
        M, K = 128, 4096
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        packed, state = quantize_nvfp4(A)

        assert packed.shape == (M * K // 2,)
        assert state.block_scales.shape == (M * K // 16,)
        assert state.shape == (M, K)
        assert state.rotated

    def test_tensor_scale_computation(self):
        """Verify tensor_scale is computed correctly."""
        torch.manual_seed(42)
        A = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        _, state = quantize_nvfp4(A)
        expected_ts = A.abs().max().item()
        assert abs(state.tensor_scale - expected_ts) < 0.01

    def test_outlier_spreading(self):
        """Hadamard rotation should spread outliers, improving quantization."""
        torch.manual_seed(42)
        # Create data with extreme outliers (Laplace distribution)
        e1 = torch.empty(128, 4096, device="cuda").exponential_(1.0)
        e2 = torch.empty(128, 4096, device="cuda").exponential_(1.0)
        A = (e1 - e2).to(torch.bfloat16)

        packed, state = quantize_nvfp4(A)
        deq = dequantize_nvfp4(packed, state)
        err = (deq - A).abs().mean() / A.abs().mean()
        assert err < 0.15, f"Outlier data error {err:.4f} exceeds 15%"


class TestFusedQuantizePadding:
    """Test M padding (non-multiples of 128)."""

    @pytest.mark.parametrize("M", [1, 7, 33, 100, 127, 129, 255])
    def test_padding_round_trip(self, M):
        """M values not divisible by 128 should still produce correct output."""
        torch.manual_seed(42)
        K = 4096
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        packed, state = quantize_nvfp4(A)
        deq = dequantize_nvfp4(packed, state)
        err = (deq - A).abs().mean() / A.abs().mean()
        assert err < 0.12, f"Padding error for M={M}: {err:.4f} exceeds 12%"
        assert packed.shape == (M * K // 2,)
        assert state.block_scales.shape == (M * K // 16,)


class TestFusedQuantizeEndToEnd:
    """End-to-end tests: fused quantize -> CUTLASS GEMM."""

    def test_gemm_with_fused_quantize(self):
        """GEMM using fused-quantized inputs should match BF16 reference."""
        torch.manual_seed(42)
        M, N, K = 128, 256, 4096
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        ref = A @ B.T

        packed_a, state_a = quantize_nvfp4(A)
        packed_b, state_b = quantize_nvfp4(B)

        C = torch.ops.bitsandbytes.gemm_nvfp4(
            packed_a,
            packed_b,
            state_a.block_scales_blocked,
            state_b.block_scales_blocked,
            state_a.tensor_scale,
            state_b.tensor_scale,
            M,
            N,
            K,
        )

        err = (C - ref).abs().mean() / ref.abs().mean()
        assert err < 0.20, f"End-to-end GEMM error {err:.4f} exceeds 20%"

    def test_gemm_large_batch(self):
        """Large batch GEMM with fused quantize should work correctly."""
        torch.manual_seed(42)
        M, N, K = 4096, 4096, 4096
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        ref = A @ B.T

        packed_a, state_a = quantize_nvfp4(A)
        packed_b, state_b = quantize_nvfp4(B)

        C = torch.ops.bitsandbytes.gemm_nvfp4(
            packed_a,
            packed_b,
            state_a.block_scales_blocked,
            state_b.block_scales_blocked,
            state_a.tensor_scale,
            state_b.tensor_scale,
            M,
            N,
            K,
        )

        err = (C - ref).abs().mean() / ref.abs().mean()
        assert err < 0.20, f"Large batch GEMM error {err:.4f} exceeds 20%"


class TestFusedQuantizeFallback:
    """Test fallback to hand-written kernel when CUTLASS unavailable."""

    def test_fallback_detection(self):
        """_has_cutlass_fused_quantize should return True on SM_120+."""
        assert _has_cutlass_fused_quantize()

    def test_fallback_monkeypatch(self):
        """When fused quantize unavailable, fall back to hand-written kernel."""
        import bitsandbytes.functional as F

        original = F._has_cutlass_fused_quantize
        try:
            F._has_cutlass_fused_quantize = lambda: False

            torch.manual_seed(42)
            A = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
            packed, state = quantize_nvfp4(A)
            deq = dequantize_nvfp4(packed, state)
            err = (deq - A).abs().mean() / A.abs().mean()
            assert err < 0.12, f"Fallback error {err:.4f} exceeds 12%"
        finally:
            F._has_cutlass_fused_quantize = original


class TestFusedQuantizeDtypeConversion:
    """Test BF16 conversion for non-BF16 inputs."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_non_bf16_input(self, dtype):
        """Non-BF16 inputs should be converted to BF16 for fused quantize."""
        torch.manual_seed(42)
        A = torch.randn(128, 4096, dtype=dtype, device="cuda")
        packed, state = quantize_nvfp4(A)
        deq = dequantize_nvfp4(packed, state)
        err = (deq.to(dtype) - A).abs().mean() / A.abs().mean()
        assert err < 0.15, f"Non-BF16 input error for {dtype}: {err:.4f}"
