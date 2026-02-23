"""Tests for CUTLASS-based fused quantize (QuTLASS integration).

Tests the fused quantize path that uses CUTLASS GEMM for 7-9x faster
NVFP4 quantization with optional Hadamard rotation.
"""

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.functional import (
    NVFP4QuantState,
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


class TestFusedQuantizeAbsMax:
    """Test fused AbsMax quantize (no rotation)."""

    def test_round_trip_error_bounded(self):
        """Fused absmax quantize round-trip error should match old kernel."""
        torch.manual_seed(42)
        A = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
        packed, state = quantize_nvfp4(A, rotate=False)
        deq = dequantize_nvfp4(packed, state)
        err = (deq - A).abs().mean() / A.abs().mean()
        assert err < 0.12, f"Round-trip error {err:.4f} exceeds 12%"
        assert err > 0.01, f"Round-trip error {err:.4f} suspiciously low"

    def test_output_shapes(self):
        """Verify output tensor shapes are correct."""
        torch.manual_seed(42)
        M, K = 128, 4096
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        packed, state = quantize_nvfp4(A, rotate=False)

        assert packed.shape == (M * K // 2,)
        assert state.block_scales.shape == (M * K // 16,)
        assert state.shape == (M, K)
        assert not state.rotated

    def test_tensor_scale_computation(self):
        """Verify tensor_scale is computed correctly."""
        torch.manual_seed(42)
        A = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        _, state = quantize_nvfp4(A, rotate=False)
        expected_ts = A.abs().max().item()
        assert abs(state.tensor_scale - expected_ts) < 0.01


class TestFusedQuantizeQuest:
    """Test fused Quest quantize (with Hadamard rotation)."""

    def test_round_trip_error_bounded(self):
        """Fused quest quantize round-trip error should be reasonable."""
        torch.manual_seed(42)
        A = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
        packed, state = quantize_nvfp4(A, rotate=True)
        deq = dequantize_nvfp4(packed, state)
        err = (deq - A).abs().mean() / A.abs().mean()
        assert err < 0.12, f"Round-trip error {err:.4f} exceeds 12%"
        assert state.rotated

    def test_rotation_error_comparable(self):
        """Hadamard rotation error should be comparable to non-rotated."""
        torch.manual_seed(42)
        # Create data with outliers (Laplace distribution)
        e1 = torch.empty(128, 4096, device="cuda").exponential_(1.0)
        e2 = torch.empty(128, 4096, device="cuda").exponential_(1.0)
        A = (e1 - e2).to(torch.bfloat16)

        _, state_norot = quantize_nvfp4(A, rotate=False)
        deq_norot = dequantize_nvfp4(state_norot.packed_data, state_norot)
        err_norot = (deq_norot - A).abs().mean() / A.abs().mean()

        _, state_rot = quantize_nvfp4(A, rotate=True)
        deq_rot = dequantize_nvfp4(state_rot.packed_data, state_rot)
        err_rot = (deq_rot - A).abs().mean() / A.abs().mean()

        # Both should be bounded; rotation should not significantly degrade
        assert err_rot < 0.15, f"Rotation error {err_rot:.4f} exceeds 15%"
        assert err_norot < 0.15, f"Non-rotation error {err_norot:.4f} exceeds 15%"
        # Rotation should not be more than 50% worse than non-rotated
        assert err_rot < err_norot * 1.5, (
            f"Rotation error {err_rot:.4f} much worse than non-rotated {err_norot:.4f}"
        )


class TestFusedQuantizePadding:
    """Test M padding (non-multiples of 128)."""

    @pytest.mark.parametrize("M", [1, 7, 33, 100, 127, 129, 255])
    def test_padding_round_trip(self, M):
        """M values not divisible by 128 should still produce correct output."""
        torch.manual_seed(42)
        K = 4096
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        packed, state = quantize_nvfp4(A, rotate=False)
        deq = dequantize_nvfp4(packed, state)
        err = (deq - A).abs().mean() / A.abs().mean()
        assert err < 0.12, f"Padding error for M={M}: {err:.4f} exceeds 12%"
        assert packed.shape == (M * K // 2,)
        assert state.block_scales.shape == (M * K // 16,)

    @pytest.mark.parametrize("M", [1, 7, 100, 255])
    def test_padding_with_rotation(self, M):
        """Padded rotation round-trip should produce correct output."""
        torch.manual_seed(42)
        K = 4096
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        packed, state = quantize_nvfp4(A, rotate=True)
        deq = dequantize_nvfp4(packed, state)
        err = (deq - A).abs().mean() / A.abs().mean()
        assert err < 0.12, f"Padded rotation error for M={M}: {err:.4f}"


class TestFusedQuantizeEndToEnd:
    """End-to-end tests: fused quantize -> CUTLASS GEMM."""

    def test_gemm_with_fused_quantize(self):
        """GEMM using fused-quantized inputs should match BF16 reference."""
        torch.manual_seed(42)
        M, N, K = 128, 256, 4096
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        ref = A @ B.T

        packed_a, state_a = quantize_nvfp4(A, rotate=True)
        packed_b, state_b = quantize_nvfp4(B, rotate=True)

        C = torch.ops.bitsandbytes.gemm_nvfp4(
            packed_a, packed_b,
            state_a.block_scales_blocked, state_b.block_scales_blocked,
            state_a.tensor_scale, state_b.tensor_scale,
            M, N, K,
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

        packed_a, state_a = quantize_nvfp4(A, rotate=False)
        packed_b, state_b = quantize_nvfp4(B, rotate=False)

        C = torch.ops.bitsandbytes.gemm_nvfp4(
            packed_a, packed_b,
            state_a.block_scales_blocked, state_b.block_scales_blocked,
            state_a.tensor_scale, state_b.tensor_scale,
            M, N, K,
        )

        err = (C - ref).abs().mean() / ref.abs().mean()
        assert err < 0.20, f"Large batch GEMM error {err:.4f} exceeds 20%"


class TestFusedQuantizeFallback:
    """Test fallback to old kernel."""

    def test_fallback_detection(self):
        """_has_cutlass_fused_quantize should return True on SM_120+."""
        assert _has_cutlass_fused_quantize()

    def test_fallback_monkeypatch(self):
        """When fused quantize unavailable, fall back to old kernel."""
        import bitsandbytes.functional as F

        original = F._has_cutlass_fused_quantize
        try:
            # Monkeypatch to simulate non-Blackwell
            F._has_cutlass_fused_quantize = lambda: False

            torch.manual_seed(42)
            A = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
            packed, state = quantize_nvfp4(A, rotate=False)
            deq = dequantize_nvfp4(packed, state)
            err = (deq - A).abs().mean() / A.abs().mean()
            assert err < 0.12, f"Fallback error {err:.4f} exceeds 12%"
        finally:
            F._has_cutlass_fused_quantize = original

    def test_fallback_rotation(self):
        """Fallback with rotation should use old fused_hadamard_quantize."""
        import bitsandbytes.functional as F

        original = F._has_cutlass_fused_quantize
        try:
            F._has_cutlass_fused_quantize = lambda: False

            torch.manual_seed(42)
            A = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
            packed, state = quantize_nvfp4(A, rotate=True)
            deq = dequantize_nvfp4(packed, state)
            err = (deq - A).abs().mean() / A.abs().mean()
            assert err < 0.12, f"Fallback rotation error {err:.4f} exceeds 12%"
            assert state.rotated
        finally:
            F._has_cutlass_fused_quantize = original


class TestFusedQuantizeDtypeConversion:
    """Test BF16 conversion for non-BF16 inputs."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_non_bf16_input(self, dtype):
        """Non-BF16 inputs should be converted to BF16 for fused quantize."""
        torch.manual_seed(42)
        A = torch.randn(128, 4096, dtype=dtype, device="cuda")
        packed, state = quantize_nvfp4(A, rotate=False)
        deq = dequantize_nvfp4(packed, state)
        err = (deq.to(dtype) - A).abs().mean() / A.abs().mean()
        assert err < 0.15, f"Non-BF16 input error for {dtype}: {err:.4f}"
