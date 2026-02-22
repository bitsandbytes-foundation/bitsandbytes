"""Tests for CUDA training kernels: SwiGLU, RMSNorm, RoPE.

Tests compare CUDA kernel output against PyTorch reference implementations
to verify correctness within fp16/bf16 tolerance.
"""

import pytest
import torch

import bitsandbytes  # noqa: F401 — triggers op registration
from bitsandbytes.autograd.training_kernels import rmsnorm, rope, swiglu


def _ref_swiglu(gate, up):
    """PyTorch reference: silu(gate) * up."""
    return torch.nn.functional.silu(gate.float()) * up.float()


def _ref_rmsnorm(x, w, eps=1e-6, add_unit_offset=False):
    """PyTorch reference for RMSNorm."""
    x_f = x.float()
    rms = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_normed = x_f / rms
    w_eff = (w.float() + 1.0) if add_unit_offset else w.float()
    return x_normed * w_eff


def _ref_rope(q, cos_cache, sin_cache):
    """PyTorch reference for RoPE: rotation in pairs."""
    # q: [T, H, D], cos/sin: [T, D/2]
    half = q.shape[-1] // 2
    q_f = q.float()
    cos = cos_cache.float().unsqueeze(1)  # [T, 1, D/2]
    sin = sin_cache.float().unsqueeze(1)  # [T, 1, D/2]
    q_r = q_f[..., :half]
    q_i = q_f[..., half:]
    out = torch.cat([q_r * cos - q_i * sin, q_i * cos + q_r * sin], dim=-1)
    return out


# ============================================================================
# SwiGLU Tests
# ============================================================================


class TestSwiGLU:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_matches_reference(self, dtype):
        gate = torch.randn(128, 256, device="cuda", dtype=dtype)
        up = torch.randn(128, 256, device="cuda", dtype=dtype)

        out = torch.ops.bitsandbytes.swiglu_forward(gate, up)
        ref = _ref_swiglu(gate, up).to(dtype)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_gradcheck(self, dtype):
        """Verify backward pass produces correct gradients."""
        gate = torch.randn(16, 32, device="cuda", dtype=dtype, requires_grad=True)
        up = torch.randn(16, 32, device="cuda", dtype=dtype, requires_grad=True)

        out = swiglu(gate, up)
        loss = out.sum()
        loss.backward()

        # Compare with PyTorch reference gradients
        gate_ref = gate.detach().clone().requires_grad_(True)
        up_ref = up.detach().clone().requires_grad_(True)
        ref_out = _ref_swiglu(gate_ref, up_ref).to(dtype)
        ref_loss = ref_out.sum()
        ref_loss.backward()

        torch.testing.assert_close(gate.grad, gate_ref.grad.to(dtype), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(up.grad, up_ref.grad.to(dtype), atol=1e-2, rtol=1e-2)

    def test_autograd_function(self):
        """Test that the autograd Function wrapper works end-to-end."""
        gate = torch.randn(8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        up = torch.randn(8, 16, device="cuda", dtype=torch.float16, requires_grad=True)

        out = swiglu(gate, up)
        assert out.shape == gate.shape
        out.sum().backward()
        assert gate.grad is not None
        assert up.grad is not None

    def test_large_tensor(self):
        """Test with a large tensor to verify grid/block sizing."""
        gate = torch.randn(1024, 4096, device="cuda", dtype=torch.float16)
        up = torch.randn(1024, 4096, device="cuda", dtype=torch.float16)

        out = torch.ops.bitsandbytes.swiglu_forward(gate, up)
        ref = _ref_swiglu(gate, up).to(torch.float16)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# ============================================================================
# RMSNorm Tests
# ============================================================================


class TestRMSNorm:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_matches_reference(self, dtype):
        rows, cols = 64, 256
        x = torch.randn(rows, cols, device="cuda", dtype=dtype)
        w = torch.randn(cols, device="cuda", dtype=dtype)
        eps = 1e-6

        out, rrms = torch.ops.bitsandbytes.rmsnorm_forward(x, w, eps, False)
        ref = _ref_rmsnorm(x, w, eps).to(dtype)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_gemma_variant(self, dtype):
        """Test add_unit_offset (Gemma: uses w + 1)."""
        rows, cols = 32, 128
        x = torch.randn(rows, cols, device="cuda", dtype=dtype)
        w = torch.randn(cols, device="cuda", dtype=dtype)
        eps = 1e-6

        out, _ = torch.ops.bitsandbytes.rmsnorm_forward(x, w, eps, True)
        ref = _ref_rmsnorm(x, w, eps, add_unit_offset=True).to(dtype)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_rrms_correctness(self):
        """Verify the rrms output matches manual computation."""
        rows, cols = 16, 64
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
        w = torch.ones(cols, device="cuda", dtype=torch.float16)
        eps = 1e-6

        _, rrms = torch.ops.bitsandbytes.rmsnorm_forward(x, w, eps, False)

        # Manual: rrms = 1 / sqrt(mean(x^2) + eps)
        x_f = x.float()
        expected_rrms = torch.rsqrt(x_f.pow(2).mean(dim=-1) + eps)

        torch.testing.assert_close(rrms, expected_rrms, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_gradients(self, dtype):
        """Verify backward produces correct gradients via autograd wrapper."""
        rows, cols = 16, 64
        x = torch.randn(rows, cols, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.randn(cols, device="cuda", dtype=dtype, requires_grad=True)

        out = rmsnorm(x, w)
        out.sum().backward()

        # Reference
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = w.detach().clone().requires_grad_(True)
        ref = _ref_rmsnorm(x_ref, w_ref).to(dtype)
        ref.sum().backward()

        torch.testing.assert_close(x.grad, x_ref.grad.to(dtype), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(w.grad, w_ref.grad.to(dtype), atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("add_unit_offset", [False, True])
    def test_backward_gemma_variant(self, add_unit_offset):
        """Verify backward works with both standard and Gemma variants."""
        rows, cols = 8, 32
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16, requires_grad=True)
        w = torch.randn(cols, device="cuda", dtype=torch.float16, requires_grad=True)

        out = rmsnorm(x, w, add_unit_offset=add_unit_offset)
        out.sum().backward()

        assert x.grad is not None
        assert w.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(w.grad).any()

    def test_various_hidden_sizes(self):
        """Test with hidden sizes typical of LLMs."""
        for cols in [128, 256, 512, 1024, 2048, 4096]:
            x = torch.randn(32, cols, device="cuda", dtype=torch.float16)
            w = torch.randn(cols, device="cuda", dtype=torch.float16)
            out, _ = torch.ops.bitsandbytes.rmsnorm_forward(x, w, 1e-6, False)
            ref = _ref_rmsnorm(x, w).to(torch.float16)
            torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    def test_3d_input(self):
        """Test that autograd wrapper handles 3D input (batch, seq, hidden)."""
        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        w = torch.randn(64, device="cuda", dtype=torch.float16, requires_grad=True)

        out = rmsnorm(x, w)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad.shape == x.shape


# ============================================================================
# RoPE Tests
# ============================================================================


class TestRoPE:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_matches_reference(self, dtype):
        T, H, D = 32, 8, 64
        q = torch.randn(T, H, D, device="cuda", dtype=dtype)
        cos_cache = torch.randn(T, D // 2, device="cuda", dtype=dtype)
        sin_cache = torch.randn(T, D // 2, device="cuda", dtype=dtype)

        q_out = q.clone()
        torch.ops.bitsandbytes.rope_forward(q_out, cos_cache, sin_cache, H)

        ref = _ref_rope(q, cos_cache, sin_cache).to(dtype)

        torch.testing.assert_close(q_out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_is_inverse_rotation(self, dtype):
        """RoPE backward is forward with -sin. Verify: forward then backward = identity.

        Uses actual rotation angles (cos² + sin² = 1) rather than random values.
        """
        T, H, D = 16, 4, 32
        q = torch.randn(T, H, D, device="cuda", dtype=dtype)

        # Generate proper cos/sin from rotation angles (cos² + sin² = 1)
        angles = torch.randn(T, D // 2, device="cuda", dtype=torch.float32)
        cos_cache = torch.cos(angles).to(dtype)
        sin_cache = torch.sin(angles).to(dtype)

        # Forward
        q_rot = q.clone()
        torch.ops.bitsandbytes.rope_forward(q_rot, cos_cache, sin_cache, H)

        # Backward (negate sin)
        torch.ops.bitsandbytes.rope_forward(q_rot, cos_cache, -sin_cache, H)

        # Should recover original (up to fp16 precision)
        torch.testing.assert_close(q_rot, q, atol=5e-2, rtol=5e-2)

    def test_autograd_function(self):
        """Test end-to-end autograd through the RoPE wrapper."""
        T, H, D = 8, 4, 32
        q = torch.randn(T, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
        cos_cache = torch.randn(T, D // 2, device="cuda", dtype=torch.float16)
        sin_cache = torch.randn(T, D // 2, device="cuda", dtype=torch.float16)

        out = rope(q, cos_cache, sin_cache, H)
        assert out.shape == q.shape
        out.sum().backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()

    def test_gradient_correctness(self):
        """Compare autograd gradient against PyTorch reference gradient."""
        T, H, D = 8, 4, 32
        dtype = torch.float16

        q = torch.randn(T, H, D, device="cuda", dtype=dtype, requires_grad=True)
        cos_cache = torch.randn(T, D // 2, device="cuda", dtype=dtype)
        sin_cache = torch.randn(T, D // 2, device="cuda", dtype=dtype)

        out = rope(q, cos_cache, sin_cache, H)
        out.sum().backward()

        # Reference: PyTorch implementation
        q_ref = q.detach().clone().requires_grad_(True)
        ref_out = _ref_rope(q_ref, cos_cache, sin_cache).to(dtype)
        ref_out.sum().backward()

        torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=5e-2, rtol=5e-2)

    def test_different_head_dims(self):
        """Test with various head dimensions."""
        T, H = 16, 8
        for D in [32, 64, 128]:
            q = torch.randn(T, H, D, device="cuda", dtype=torch.float16)
            cos_cache = torch.randn(T, D // 2, device="cuda", dtype=torch.float16)
            sin_cache = torch.randn(T, D // 2, device="cuda", dtype=torch.float16)

            q_out = q.clone()
            torch.ops.bitsandbytes.rope_forward(q_out, cos_cache, sin_cache, H)
            ref = _ref_rope(q, cos_cache, sin_cache).to(torch.float16)

            torch.testing.assert_close(q_out, ref, atol=1e-2, rtol=1e-2)
