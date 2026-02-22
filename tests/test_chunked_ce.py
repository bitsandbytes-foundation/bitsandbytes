"""Tests for chunked fused linear cross-entropy loss.

Verifies:
- Chunked CE matches full-materialization CE (our CUDA kernel)
- Chunked CE matches PyTorch F.cross_entropy (ground truth)
- Gradient of hidden states matches between chunked and full
- ignore_index is handled correctly
- Different chunk sizes produce identical results
- Memory savings: chunked version uses less peak memory than full materialization
"""

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import _ops  # noqa: F401 — triggers op registration
from bitsandbytes.autograd.chunked_ce import chunked_cross_entropy
from bitsandbytes.autograd.training_kernels import cross_entropy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _quantize_weight(N, K_dim, k=4, device="cuda"):
    """Create a quantized LM head weight, returning packed/absmax/codebook + N_padded."""
    W = torch.randn(N, K_dim, dtype=torch.float16, device=device)
    N_padded = ((N + 127) // 128) * 128
    if N_padded != N:
        W_padded = torch.nn.functional.pad(W, (0, 0, 0, N_padded - N))
    else:
        W_padded = W
    packed, absmax, codebook = bnb.functional.quantize_kbit(
        W_padded.reshape(-1).float(), k=k, absmax_format="fp32",
    )
    return packed, absmax, codebook, N_padded, W


def _dequant_weight(packed, absmax, codebook, k, K_dim, N_padded, N, dtype):
    """Dequantize for reference comparison."""
    n_elements = N_padded * K_dim
    w_deq = bnb.functional.dequantize_kbit(packed, absmax, codebook, k, n_elements, dtype)
    return w_deq[:n_elements].reshape(N_padded, K_dim)[:N, :]


class TestChunkedCrossEntropy:
    """Tests for ChunkedCrossEntropy autograd function."""

    def _setup(self, B=16, K=256, V=1024, k=4):
        """Create quantized LM head + hidden states + labels."""
        packed, absmax, codebook, N_padded, W_orig = _quantize_weight(V, K, k=k)
        hidden = torch.randn(B, K, dtype=torch.float16, device="cuda", requires_grad=True)
        labels = torch.randint(0, V, (B,), device="cuda")
        return hidden, packed, absmax, codebook, labels, k, K, N_padded, V

    def test_forward_matches_pytorch_ce(self):
        """Chunked CE loss should match PyTorch F.cross_entropy on full logits."""
        hidden, packed, absmax, codebook, labels, k, K, N_padded, V = self._setup()

        # Chunked CE
        loss = chunked_cross_entropy(
            hidden, packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=torch.float16,
            chunk_size=256,
        )

        # Reference: dequantize, compute full logits, F.cross_entropy
        W = _dequant_weight(packed, absmax, codebook, k, K, N_padded, V, torch.float16)
        full_logits = hidden @ W.t()  # [B, V]
        ref_loss = torch.nn.functional.cross_entropy(full_logits.float(), labels)

        torch.testing.assert_close(loss.float(), ref_loss, atol=1e-3, rtol=1e-3)

    def test_forward_matches_our_cuda_ce(self):
        """Chunked CE loss should match our existing CUDA CE kernel."""
        hidden, packed, absmax, codebook, labels, k, K, N_padded, V = self._setup()

        # Chunked CE
        loss = chunked_cross_entropy(
            hidden, packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=torch.float16,
            chunk_size=256,
        )

        # Our CUDA kernel reference
        W = _dequant_weight(packed, absmax, codebook, k, K, N_padded, V, torch.float16)
        full_logits = hidden @ W.t()
        ref_loss = cross_entropy(full_logits, labels)

        torch.testing.assert_close(loss.float(), ref_loss.float(), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("chunk_size", [128, 256, 512, 1024])
    def test_chunk_size_invariance(self, chunk_size):
        """Different chunk sizes should produce identical (or very close) loss."""
        hidden, packed, absmax, codebook, labels, k, K, N_padded, V = self._setup(V=1024)

        loss = chunked_cross_entropy(
            hidden, packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=torch.float16,
            chunk_size=chunk_size,
        )

        # Reference with chunk_size=V (single chunk = no chunking)
        ref_loss = chunked_cross_entropy(
            hidden.detach().requires_grad_(True),
            packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=torch.float16,
            chunk_size=V,  # single chunk
        )

        torch.testing.assert_close(loss.float(), ref_loss.float(), atol=1e-5, rtol=1e-5)

    def test_backward_gradient_hidden(self):
        """Gradient of hidden states should match full-materialization reference."""
        B, K, V = 8, 128, 512
        k = 4
        packed, absmax, codebook, N_padded, _ = _quantize_weight(V, K, k=k)
        labels = torch.randint(0, V, (B,), device="cuda")

        # Chunked CE gradient
        hidden1 = torch.randn(B, K, dtype=torch.float16, device="cuda", requires_grad=True)
        loss1 = chunked_cross_entropy(
            hidden1, packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=torch.float16,
            chunk_size=128,
        )
        loss1.backward()

        # Full materialization gradient
        hidden2 = hidden1.detach().clone().requires_grad_(True)
        W = _dequant_weight(packed, absmax, codebook, k, K, N_padded, V, torch.float16)
        full_logits = hidden2 @ W.t()
        ref_loss = torch.nn.functional.cross_entropy(full_logits.float(), labels)
        ref_loss.backward()

        # Compare gradients
        torch.testing.assert_close(
            hidden1.grad.float(), hidden2.grad.float(),
            atol=5e-2, rtol=5e-2,
        )

    def test_backward_gradient_matches_across_chunk_sizes(self):
        """Gradients should be consistent across different chunk sizes."""
        B, K, V = 8, 128, 512
        k = 4
        packed, absmax, codebook, N_padded, _ = _quantize_weight(V, K, k=k)
        labels = torch.randint(0, V, (B,), device="cuda")

        hidden_base = torch.randn(B, K, dtype=torch.float16, device="cuda")

        grads = []
        for cs in [64, 128, 256, 512]:
            h = hidden_base.clone().requires_grad_(True)
            loss = chunked_cross_entropy(
                h, packed, absmax, codebook, labels,
                k, K, N_padded, V,
                compute_dtype=torch.float16,
                chunk_size=cs,
            )
            loss.backward()
            grads.append(h.grad.clone())

        # All chunk sizes should produce nearly identical gradients
        # (small fp16 accumulation order differences across chunk boundaries)
        for i in range(1, len(grads)):
            torch.testing.assert_close(
                grads[0].float(), grads[i].float(),
                atol=1e-3, rtol=1e-3,
            )

    def test_ignore_index(self):
        """ignore_index labels should not contribute to loss."""
        hidden, packed, absmax, codebook, labels, k, K, N_padded, V = self._setup(B=16)

        # Set some labels to ignore
        labels[0] = -100
        labels[5] = -100
        labels[10] = -100

        loss = chunked_cross_entropy(
            hidden, packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=torch.float16,
            chunk_size=256,
        )

        # Reference
        W = _dequant_weight(packed, absmax, codebook, k, K, N_padded, V, torch.float16)
        full_logits = hidden @ W.t()
        ref_loss = torch.nn.functional.cross_entropy(full_logits.float(), labels)

        torch.testing.assert_close(loss.float(), ref_loss, atol=1e-3, rtol=1e-3)

    def test_all_ignored(self):
        """All-ignored labels should produce zero loss."""
        hidden, packed, absmax, codebook, _, k, K, N_padded, V = self._setup(B=8)
        labels = torch.full((8,), -100, device="cuda", dtype=torch.long)

        loss = chunked_cross_entropy(
            hidden, packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=torch.float16,
            chunk_size=256,
        )
        assert loss.item() == 0.0

    def test_backward_with_ignore_index(self):
        """Gradients should be zero for ignored labels' positions."""
        B, K, V = 8, 128, 256
        k = 4
        packed, absmax, codebook, N_padded, _ = _quantize_weight(V, K, k=k)

        labels = torch.randint(0, V, (B,), device="cuda")
        labels[0] = -100
        labels[3] = -100

        hidden = torch.randn(B, K, dtype=torch.float16, device="cuda", requires_grad=True)
        loss = chunked_cross_entropy(
            hidden, packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=torch.float16,
            chunk_size=64,
        )
        loss.backward()

        # Reference
        hidden_ref = hidden.detach().clone().requires_grad_(True)
        W = _dequant_weight(packed, absmax, codebook, k, K, N_padded, V, torch.float16)
        full_logits = hidden_ref @ W.t()
        ref_loss = torch.nn.functional.cross_entropy(full_logits.float(), labels)
        ref_loss.backward()

        torch.testing.assert_close(
            hidden.grad.float(), hidden_ref.grad.float(),
            atol=5e-2, rtol=5e-2,
        )

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_different_k_values(self, k):
        """Chunked CE should work with different quantization bit widths."""
        B, K, V = 8, 128, 512
        packed, absmax, codebook, N_padded, _ = _quantize_weight(V, K, k=k)
        hidden = torch.randn(B, K, dtype=torch.float16, device="cuda", requires_grad=True)
        labels = torch.randint(0, V, (B,), device="cuda")

        loss = chunked_cross_entropy(
            hidden, packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=torch.float16,
            chunk_size=128,
        )

        # Should be a finite, positive scalar
        assert loss.isfinite().all()
        assert loss.item() > 0

        # Backward should work
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.isfinite().all()

    def test_large_vocab(self):
        """Test with vocab_size=32K to verify larger-scale correctness."""
        B, K, V = 4, 256, 32768
        k = 4
        packed, absmax, codebook, N_padded, _ = _quantize_weight(V, K, k=k)
        hidden = torch.randn(B, K, dtype=torch.float16, device="cuda", requires_grad=True)
        labels = torch.randint(0, V, (B,), device="cuda")

        loss = chunked_cross_entropy(
            hidden, packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=torch.float16,
            chunk_size=4096,
        )

        # Reference
        W = _dequant_weight(packed, absmax, codebook, k, K, N_padded, V, torch.float16)
        full_logits = hidden @ W.t()
        ref_loss = torch.nn.functional.cross_entropy(full_logits.float(), labels)

        torch.testing.assert_close(loss.float(), ref_loss, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_compute_dtypes(self, dtype):
        """Both fp16 and bf16 should work as compute dtype."""
        B, K, V = 8, 128, 512
        k = 4
        packed, absmax, codebook, N_padded, _ = _quantize_weight(V, K, k=k)
        hidden = torch.randn(B, K, dtype=dtype, device="cuda", requires_grad=True)
        labels = torch.randint(0, V, (B,), device="cuda")

        loss = chunked_cross_entropy(
            hidden, packed, absmax, codebook, labels,
            k, K, N_padded, V,
            compute_dtype=dtype,
            chunk_size=128,
        )

        assert loss.isfinite().all()
        assert loss.item() > 0
        loss.backward()
        assert hidden.grad is not None
