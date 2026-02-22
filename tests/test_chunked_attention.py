"""Tests for chunked Flash Attention (single-GPU ring attention).

Verifies:
- Chunked attention matches unchunked flash_attn output
- Causal masking is correct at chunk boundaries
- GQA (K/V fewer heads than Q) works correctly
- Different chunk sizes produce identical results
- Backward pass gradient correctness
- Full chunking (Q+K/V) matches Q-only chunking
"""

import pytest
import torch

from flash_attn import flash_attn_func

from bitsandbytes.attention import chunked_flash_attention, chunked_flash_attention_full

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class TestChunkedFlashAttention:
    """Tests for Q-only chunked attention."""

    def test_matches_unchunked(self):
        """Chunked attention should match unchunked flash_attn exactly."""
        B, S, H, D = 2, 512, 8, 64
        Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

        # Unchunked reference
        ref = flash_attn_func(Q, K, V, causal=True)

        # Chunked
        out = chunked_flash_attention(Q, K, V, chunk_size=128, causal=True)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("chunk_size", [64, 128, 256, 512])
    def test_chunk_size_invariance(self, chunk_size):
        """Different chunk sizes should produce identical results."""
        B, S, H, D = 2, 512, 8, 64
        Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

        ref = flash_attn_func(Q, K, V, causal=True)
        out = chunked_flash_attention(Q, K, V, chunk_size=chunk_size, causal=True)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_causal_masking_at_boundaries(self):
        """Verify causal masking is correct at chunk boundaries.

        Position at the start of chunk 2 should NOT attend to future positions.
        We verify by checking that changing future K/V values doesn't affect output.
        """
        B, S, H, D = 1, 256, 4, 32
        chunk_size = 128

        Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

        out1 = chunked_flash_attention(Q, K, V, chunk_size=chunk_size, causal=True)

        # Modify K/V at positions [200:256] — should NOT affect output at position 128
        K2 = K.clone()
        V2 = V.clone()
        K2[:, 200:, :, :] = torch.randn_like(K2[:, 200:, :, :])
        V2[:, 200:, :, :] = torch.randn_like(V2[:, 200:, :, :])

        out2 = chunked_flash_attention(Q, K2, V2, chunk_size=chunk_size, causal=True)

        # Positions 0-128 should be identical (they can't see positions 200+)
        torch.testing.assert_close(
            out1[:, :129], out2[:, :129],
            atol=0, rtol=0,
            msg="Causal masking violated at chunk boundary",
        )

    def test_gqa_support(self):
        """GQA: K/V have fewer heads than Q."""
        B, S, H_q, H_kv, D = 2, 256, 16, 2, 64

        Q = torch.randn(B, S, H_q, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H_kv, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H_kv, D, device="cuda", dtype=torch.float16)

        ref = flash_attn_func(Q, K, V, causal=True)
        out = chunked_flash_attention(Q, K, V, chunk_size=64, causal=True)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_backward_gradient(self):
        """Verify gradients flow correctly through chunked attention."""
        B, S, H, D = 2, 256, 8, 64
        Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
        K = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
        V = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)

        out = chunked_flash_attention(Q, K, V, chunk_size=64, causal=True)
        loss = out.sum()
        loss.backward()

        assert Q.grad is not None
        assert K.grad is not None
        assert V.grad is not None
        assert Q.grad.isfinite().all()
        assert K.grad.isfinite().all()
        assert V.grad.isfinite().all()

    def test_backward_matches_unchunked(self):
        """Chunked gradients should match unchunked flash_attn gradients."""
        B, S, H, D = 1, 256, 4, 32

        Q_base = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        K_base = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        V_base = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

        # Unchunked reference
        Q1 = Q_base.clone().requires_grad_(True)
        K1 = K_base.clone().requires_grad_(True)
        V1 = V_base.clone().requires_grad_(True)
        ref_out = flash_attn_func(Q1, K1, V1, causal=True)
        ref_out.sum().backward()

        # Chunked
        Q2 = Q_base.clone().requires_grad_(True)
        K2 = K_base.clone().requires_grad_(True)
        V2 = V_base.clone().requires_grad_(True)
        chunk_out = chunked_flash_attention(Q2, K2, V2, chunk_size=64, causal=True)
        chunk_out.sum().backward()

        torch.testing.assert_close(Q1.grad, Q2.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(K1.grad, K2.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(V1.grad, V2.grad, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_dtypes(self, dtype):
        """Both fp16 and bf16 should work."""
        B, S, H, D = 2, 256, 8, 64
        Q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        K = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        V = torch.randn(B, S, H, D, device="cuda", dtype=dtype)

        out = chunked_flash_attention(Q, K, V, chunk_size=64, causal=True)
        assert out.dtype == dtype
        assert out.shape == Q.shape

    def test_single_chunk_passthrough(self):
        """When S <= chunk_size, should be identical to direct flash_attn call."""
        B, S, H, D = 2, 128, 8, 64
        Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

        ref = flash_attn_func(Q, K, V, causal=True)
        out = chunked_flash_attention(Q, K, V, chunk_size=256, causal=True)

        torch.testing.assert_close(out, ref, atol=0, rtol=0)

    def test_uneven_last_chunk(self):
        """Handle sequence length not divisible by chunk_size."""
        B, S, H, D = 2, 300, 8, 64  # 300 not divisible by 128
        Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

        ref = flash_attn_func(Q, K, V, causal=True)
        out = chunked_flash_attention(Q, K, V, chunk_size=128, causal=True)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_longer_sequence(self):
        """Test with a longer sequence (2K tokens)."""
        B, S, H, D = 1, 2048, 8, 64
        Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

        ref = flash_attn_func(Q, K, V, causal=True)
        out = chunked_flash_attention(Q, K, V, chunk_size=512, causal=True)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_non_causal(self):
        """Non-causal (bidirectional) attention should also work."""
        B, S, H, D = 2, 256, 8, 64
        Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

        ref = flash_attn_func(Q, K, V, causal=False)
        out = chunked_flash_attention(Q, K, V, chunk_size=64, causal=False)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


class TestChunkedFlashAttentionFull:
    """Tests for fully chunked attention (Q+K/V) with logsumexp merging."""

    def test_matches_unchunked(self):
        """Fully chunked should match unchunked flash_attn."""
        B, S, H, D = 1, 256, 4, 32
        Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

        ref = flash_attn_func(Q, K, V, causal=True)
        out = chunked_flash_attention_full(
            Q, K, V, q_chunk_size=64, kv_chunk_size=64, causal=True,
        )

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_matches_q_only_chunking(self):
        """Full chunking should match Q-only chunking."""
        B, S, H, D = 1, 256, 4, 32
        Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

        ref = chunked_flash_attention(Q, K, V, chunk_size=64, causal=True)
        out = chunked_flash_attention_full(
            Q, K, V, q_chunk_size=64, kv_chunk_size=64, causal=True,
        )

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_gqa_full_chunking(self):
        """GQA with full Q+K/V chunking."""
        B, S, H_q, H_kv, D = 1, 256, 8, 2, 64
        Q = torch.randn(B, S, H_q, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, H_kv, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, H_kv, D, device="cuda", dtype=torch.float16)

        ref = flash_attn_func(Q, K, V, causal=True)
        out = chunked_flash_attention_full(
            Q, K, V, q_chunk_size=64, kv_chunk_size=64, causal=True,
        )

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
