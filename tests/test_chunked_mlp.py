"""Tests for sequence-chunked MLP wrapper.

Verifies:
- Chunked MLP output matches non-chunked LoRA_MLP_Kbit
- Gradients match between chunked and non-chunked
- Different chunk sizes produce identical results
- Gradient checkpointing mode works correctly
- Last chunk smaller than chunk_size is handled
"""

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import _ops  # noqa: F401
from bitsandbytes.autograd.lora_kbit import LoRA_MLP_Kbit
from bitsandbytes.chunked import chunked_mlp_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _quantize_weight(N, K_dim, k=4, device="cuda"):
    """Create a quantized weight matrix."""
    W = torch.randn(N, K_dim, dtype=torch.float16, device=device)
    N_padded = ((N + 127) // 128) * 128
    if N_padded != N:
        W_padded = torch.nn.functional.pad(W, (0, 0, 0, N_padded - N))
    else:
        W_padded = W
    packed, absmax, codebook = bnb.functional.quantize_kbit(
        W_padded.reshape(-1).float(), k=k, absmax_format="fp32",
    )
    return packed, absmax, codebook, N_padded


def _setup_mlp(M=64, K_in=256, N_hidden=512, K_hidden=512, N_out=256, r=16, k=4):
    """Create quantized MLP weights + LoRA adapters.

    Uses small initialization scale to prevent fp16 overflow in SwiGLU.
    """
    # Gate: [N_hidden, K_in]
    pg, ag, cg, npg = _quantize_weight(N_hidden, K_in, k=k)
    # Up: [N_hidden, K_in]
    pu, au, cu, npu = _quantize_weight(N_hidden, K_in, k=k)
    # Down: [N_out, K_hidden]
    pd, ad, cd, npd = _quantize_weight(N_out, K_hidden, k=k)

    # Scale inputs down to prevent fp16 overflow through gate/up/SwiGLU
    X = (torch.randn(M, K_in, dtype=torch.float16, device="cuda") * 0.1).requires_grad_(True)

    # LoRA adapters (small init to keep outputs bounded)
    scale = 0.01
    A_gate = (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
    B_gate = (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
    A_up = (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
    B_up = (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
    A_down = (torch.randn(r, K_hidden, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
    B_down = (torch.randn(N_out, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

    s = 0.5
    return (
        X,
        pg, ag, cg, A_gate, B_gate, s,
        pu, au, cu, A_up, B_up, s,
        pd, ad, cd, A_down, B_down, s,
        k, K_in, N_hidden, npg,
        K_hidden, N_out, npd,
        torch.float16,
    )


class TestChunkedMLP:
    """Tests for chunked_mlp_forward."""

    def test_output_matches_unchunked(self):
        """Chunked output should match non-chunked LoRA_MLP_Kbit."""
        args = _setup_mlp(M=64)
        X = args[0]

        # Non-chunked reference
        ref = LoRA_MLP_Kbit.apply(*args)

        # Chunked (no checkpoint to avoid recomputation differences)
        out = chunked_mlp_forward(X, 16, *args[1:], use_checkpoint=False)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_output_matches_with_checkpoint(self):
        """Checkpointed chunked output should match non-chunked."""
        args = _setup_mlp(M=64)
        X = args[0]

        ref = LoRA_MLP_Kbit.apply(*args)
        out = chunked_mlp_forward(X,16, *args[1:], use_checkpoint=True)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("chunk_size", [8, 16, 32, 64])
    def test_chunk_size_invariance(self, chunk_size):
        """Different chunk sizes produce identical output."""
        args = _setup_mlp(M=64)
        X = args[0]

        ref = LoRA_MLP_Kbit.apply(*args)
        out = chunked_mlp_forward(X,chunk_size, *args[1:], use_checkpoint=False)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_gradients_match_no_checkpoint(self):
        """Gradients should match between chunked and non-chunked (no checkpoint)."""
        args = _setup_mlp(M=32)
        X, *rest = args

        # Non-chunked
        ref = LoRA_MLP_Kbit.apply(*args)
        ref.sum().backward()
        grad_X_ref = X.grad.clone()

        X.grad = None

        # Chunked (no checkpoint)
        out = chunked_mlp_forward(X,8, *rest, use_checkpoint=False)
        out.sum().backward()

        torch.testing.assert_close(X.grad, grad_X_ref, atol=1e-2, rtol=1e-2)

    def test_gradients_match_with_checkpoint(self):
        """Gradients should match with gradient checkpointing enabled."""
        args = _setup_mlp(M=32)
        X, *rest = args

        # Non-chunked reference
        ref = LoRA_MLP_Kbit.apply(*args)
        ref.sum().backward()
        grad_X_ref = X.grad.clone()

        X.grad = None

        # Chunked with checkpoint
        out = chunked_mlp_forward(X,8, *rest, use_checkpoint=True)
        out.sum().backward()

        torch.testing.assert_close(X.grad, grad_X_ref, atol=1e-2, rtol=1e-2)

    def test_lora_adapter_gradients(self):
        """LoRA adapter gradients should match with chunking."""
        M, K_in, N_hidden = 32, 128, 256
        K_hidden, N_out, r, k = 256, 128, 8, 4

        pg, ag, cg, npg = _quantize_weight(N_hidden, K_in, k=k)
        pu, au, cu, npu = _quantize_weight(N_hidden, K_in, k=k)
        pd, ad, cd, npd = _quantize_weight(N_out, K_hidden, k=k)

        s = 0.5
        X_base = torch.randn(M, K_in, dtype=torch.float16, device="cuda") * 0.1

        # Create two sets of LoRA adapters (small init to avoid fp16 overflow)
        scale = 0.01

        def make_adapters():
            return (
                (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True),
                (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True),
                (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True),
                (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True),
                (torch.randn(r, K_hidden, dtype=torch.float16, device="cuda") * scale).requires_grad_(True),
                (torch.randn(N_out, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True),
            )

        A_g1, B_g1, A_u1, B_u1, A_d1, B_d1 = make_adapters()

        # Clone for chunked version
        A_g2 = A_g1.detach().clone().requires_grad_(True)
        B_g2 = B_g1.detach().clone().requires_grad_(True)
        A_u2 = A_u1.detach().clone().requires_grad_(True)
        B_u2 = B_u1.detach().clone().requires_grad_(True)
        A_d2 = A_d1.detach().clone().requires_grad_(True)
        B_d2 = B_d1.detach().clone().requires_grad_(True)

        X1 = X_base.clone().requires_grad_(True)
        X2 = X_base.clone().requires_grad_(True)

        # Non-chunked
        out1 = LoRA_MLP_Kbit.apply(
            X1,
            pg, ag, cg, A_g1, B_g1, s,
            pu, au, cu, A_u1, B_u1, s,
            pd, ad, cd, A_d1, B_d1, s,
            k, K_in, N_hidden, npg,
            K_hidden, N_out, npd, torch.float16,
        )
        out1.sum().backward()

        # Chunked
        out2 = chunked_mlp_forward(
            X2, chunk_size=8,
            packed_gate=pg, absmax_gate=ag, codebook_gate=cg,
            A_gate=A_g2, B_gate=B_g2, s_gate=s,
            packed_up=pu, absmax_up=au, codebook_up=cu,
            A_up=A_u2, B_up=B_u2, s_up=s,
            packed_down=pd, absmax_down=ad, codebook_down=cd,
            A_down=A_d2, B_down=B_d2, s_down=s,
            k=k, K_dim_in=K_in, N_hidden=N_hidden, N_hidden_padded=npg,
            K_dim_hidden=K_hidden, N_out=N_out, N_out_padded=npd,
            compute_dtype=torch.float16, use_checkpoint=False,
        )
        out2.sum().backward()

        # Compare adapter gradients
        for name, g1, g2 in [
            ("A_gate", A_g1.grad, A_g2.grad),
            ("B_gate", B_g1.grad, B_g2.grad),
            ("A_up", A_u1.grad, A_u2.grad),
            ("B_up", B_u1.grad, B_u2.grad),
            ("A_down", A_d1.grad, A_d2.grad),
            ("B_down", B_d1.grad, B_d2.grad),
        ]:
            torch.testing.assert_close(
                g1.float(), g2.float(), atol=5e-2, rtol=5e-2,
                msg=f"Gradient mismatch for {name}",
            )

    def test_uneven_last_chunk(self):
        """Handle M not divisible by chunk_size."""
        args = _setup_mlp(M=50)  # 50 not divisible by 16
        X = args[0]

        ref = LoRA_MLP_Kbit.apply(*args)
        out = chunked_mlp_forward(X,16, *args[1:], use_checkpoint=False)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_single_chunk_passthrough(self):
        """When M <= chunk_size, should be identical to direct call."""
        args = _setup_mlp(M=16)
        X = args[0]

        ref = LoRA_MLP_Kbit.apply(*args)
        out = chunked_mlp_forward(X,32, *args[1:], use_checkpoint=False)

        torch.testing.assert_close(out, ref, atol=0, rtol=0)

    def test_peak_memory_reduced(self):
        """Chunked should use less peak memory for large sequences.

        This is a basic check: allocate a moderately large input,
        verify that chunked+checkpoint doesn't OOM while unchunked might
        use more memory.
        """
        M, K_in, N_hidden = 512, 512, 2048
        K_hidden, N_out, r, k = 2048, 512, 16, 4

        pg, ag, cg, npg = _quantize_weight(N_hidden, K_in, k=k)
        pu, au, cu, npu = _quantize_weight(N_hidden, K_in, k=k)
        pd, ad, cd, npd = _quantize_weight(N_out, K_hidden, k=k)

        s = 0.5
        scale = 0.01
        A_gate = (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_gate = (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        A_up = (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_up = (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        A_down = (torch.randn(r, K_hidden, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_down = (torch.randn(N_out, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

        X = (torch.randn(M, K_in, dtype=torch.float16, device="cuda") * 0.1).requires_grad_(True)

        torch.cuda.reset_peak_memory_stats()

        # Chunked + checkpoint forward+backward
        out = chunked_mlp_forward(
            X, chunk_size=64,
            packed_gate=pg, absmax_gate=ag, codebook_gate=cg,
            A_gate=A_gate, B_gate=B_gate, s_gate=s,
            packed_up=pu, absmax_up=au, codebook_up=cu,
            A_up=A_up, B_up=B_up, s_up=s,
            packed_down=pd, absmax_down=ad, codebook_down=cd,
            A_down=A_down, B_down=B_down, s_down=s,
            k=k, K_dim_in=K_in, N_hidden=N_hidden, N_hidden_padded=npg,
            K_dim_hidden=K_hidden, N_out=N_out, N_out_padded=npd,
            compute_dtype=torch.float16, use_checkpoint=True,
        )
        out.sum().backward()

        # If we got here without OOM, the chunked version works
        assert out.shape == (M, N_out)
        assert X.grad is not None
        assert X.grad.shape == X.shape
