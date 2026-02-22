"""
Tests for fused LoRA autograd functions on kbit-quantized weights.

Verifies:
- LoRA_W_Kbit: single projection gradient correctness
- LoRA_QKV_Kbit: fused Q+K+V gradient correctness
- LoRA_MLP_Kbit: fused gate+up+down+SwiGLU gradient correctness
- All tests compare against naive (separate call) reference implementations
"""

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import _ops  # noqa: F401
from bitsandbytes.autograd.lora_kbit import LoRA_MLP_Kbit, LoRA_QKV_Kbit, LoRA_W_Kbit

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _quantize_weight(N, K_dim, k=4, device="cuda"):
    """Create a quantized weight matrix, returning packed/absmax/codebook + original."""
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


def _dequant_weight(packed, absmax, codebook, k, K_dim, N_padded, N, dtype):
    """Dequantize for reference comparison."""
    n_elements = N_padded * K_dim
    w_deq = bnb.functional.dequantize_kbit(packed, absmax, codebook, k, n_elements, dtype)
    return w_deq[:n_elements].reshape(N_padded, K_dim)[:N, :]


class TestLoRA_W_Kbit:
    """Tests for single-projection LoRA_W_Kbit."""

    def _setup(self, M=4, K=256, N=128, r=16, k=4):
        packed, absmax, codebook, N_padded = _quantize_weight(N, K, k=k)
        X = torch.randn(M, K, dtype=torch.float16, device="cuda", requires_grad=True)
        A = torch.randn(r, K, dtype=torch.float16, device="cuda", requires_grad=True)
        B = torch.randn(N, r, dtype=torch.float16, device="cuda", requires_grad=True)
        s = 0.5
        return X, packed, absmax, codebook, A, B, s, k, K, N_padded, N

    def test_forward_correctness(self):
        """Forward should match naive dequant + matmul + LoRA."""
        X, packed, absmax, codebook, A, B, s, k, K, N_padded, N = self._setup()

        out = LoRA_W_Kbit.apply(
            X, packed, absmax, codebook, A, B, s, k, K, N_padded, N, torch.float16,
        )

        # Reference
        W = _dequant_weight(packed, absmax, codebook, k, K, N_padded, N, torch.float16)
        ref = X @ W.t() + (X @ A.t()) @ B.t() * s

        diff = (out.float() - ref.float()).abs()
        assert diff.max().item() < 0.01, f"Forward max diff: {diff.max().item()}"

    def test_grad_A_correctness(self):
        """grad_A should match naive reference."""
        X, packed, absmax, codebook, A, B, s, k, K, N_padded, N = self._setup()

        out = LoRA_W_Kbit.apply(
            X, packed, absmax, codebook, A, B, s, k, K, N_padded, N, torch.float16,
        )
        out.sum().backward()

        # Reference: grad_A = s * (grad_output @ B)^T @ X where grad_output = ones
        grad_out = torch.ones_like(out)
        grad_A_ref = ((grad_out @ B).t() @ X) * s

        diff = (A.grad.float() - grad_A_ref.float()).abs()
        scale = grad_A_ref.float().abs().clamp(min=1e-3)
        rel_err = (diff / scale).max().item()
        assert rel_err < 0.02, f"grad_A relative error: {rel_err}"

    def test_grad_B_correctness(self):
        """grad_B should match naive reference."""
        X, packed, absmax, codebook, A, B, s, k, K, N_padded, N = self._setup()

        out = LoRA_W_Kbit.apply(
            X, packed, absmax, codebook, A, B, s, k, K, N_padded, N, torch.float16,
        )
        out.sum().backward()

        # Reference: grad_B = s * grad_output^T @ (X @ A^T)
        grad_out = torch.ones_like(out)
        Z = X @ A.t()
        grad_B_ref = (grad_out.t() @ Z) * s

        diff = (B.grad.float() - grad_B_ref.float()).abs()
        scale = grad_B_ref.float().abs().clamp(min=1e-3)
        rel_err = (diff / scale).max().item()
        assert rel_err < 0.02, f"grad_B relative error: {rel_err}"

    def test_grad_X_correctness(self):
        """grad_X should match naive reference."""
        X, packed, absmax, codebook, A, B, s, k, K, N_padded, N = self._setup()

        out = LoRA_W_Kbit.apply(
            X, packed, absmax, codebook, A, B, s, k, K, N_padded, N, torch.float16,
        )
        out.sum().backward()

        # Reference: grad_X = grad_output @ W + s * grad_output @ B @ A
        W = _dequant_weight(packed, absmax, codebook, k, K, N_padded, N, torch.float16)
        grad_out = torch.ones_like(out)
        grad_X_ref = grad_out @ W + (grad_out @ B @ A) * s

        diff = (X.grad.float() - grad_X_ref.float()).abs()
        scale = grad_X_ref.float().abs().clamp(min=1e-3)
        rel_err = (diff / scale).max().item()
        assert rel_err < 0.02, f"grad_X relative error: {rel_err}"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_all_k_values(self, k):
        """Gradients should work for all bit widths."""
        X, packed, absmax, codebook, A, B, s, _, K, N_padded, N = self._setup(k=k)
        out = LoRA_W_Kbit.apply(
            X, packed, absmax, codebook, A, B, s, k, K, N_padded, N, torch.float16,
        )
        out.sum().backward()
        assert A.grad is not None
        assert B.grad is not None
        assert X.grad is not None

    def test_shapes(self):
        """Output and gradient shapes should be correct."""
        M, K, N, r = 8, 512, 256, 32
        X, packed, absmax, codebook, A, B, s, k, _, N_padded, _ = self._setup(M=M, K=K, N=N, r=r)
        out = LoRA_W_Kbit.apply(
            X, packed, absmax, codebook, A, B, s, k, K, N_padded, N, torch.float16,
        )
        assert out.shape == (M, N)
        out.sum().backward()
        assert A.grad.shape == (r, K)
        assert B.grad.shape == (N, r)
        assert X.grad.shape == (M, K)


class TestLoRA_QKV_Kbit:
    """Tests for fused Q+K+V LoRA_QKV_Kbit."""

    def _setup(self, M=4, K=256, N=128, r=16, k=4):
        """Create Q/K/V weights and LoRA adapters."""
        projs = []
        for _ in range(3):
            packed, absmax, codebook, N_padded = _quantize_weight(N, K, k=k)
            A = torch.randn(r, K, dtype=torch.float16, device="cuda", requires_grad=True)
            B = torch.randn(N, r, dtype=torch.float16, device="cuda", requires_grad=True)
            projs.append((packed, absmax, codebook, A, B, 0.5))
        X = torch.randn(M, K, dtype=torch.float16, device="cuda", requires_grad=True)
        return X, projs, k, K, N_padded, N

    def test_forward_matches_separate(self):
        """Fused QKV should match three separate LoRA_W_Kbit calls."""
        X, projs, k, K, N_padded, N = self._setup()

        Q, Kp, V = LoRA_QKV_Kbit.apply(
            X,
            *projs[0][:3], projs[0][3], projs[0][4], projs[0][5],
            *projs[1][:3], projs[1][3], projs[1][4], projs[1][5],
            *projs[2][:3], projs[2][3], projs[2][4], projs[2][5],
            k, K, N_padded, N, torch.float16,
        )

        # Reference: three separate calls
        for i, (out, name) in enumerate([(Q, "Q"), (Kp, "K"), (V, "V")]):
            packed, absmax, codebook, A, B, s = projs[i]
            W = _dequant_weight(packed, absmax, codebook, k, K, N_padded, N, torch.float16)
            ref = X.detach() @ W.t() + (X.detach() @ A.detach().t()) @ B.detach().t() * s
            diff = (out.float() - ref.float()).abs().max().item()
            assert diff < 0.01, f"{name} forward max diff: {diff}"

    def test_gradients_match_separate(self):
        """Fused backward should match three separate LoRA_W_Kbit backwards."""
        M, K, N, r, k = 4, 256, 128, 16, 4

        # Fused path
        X_fused = torch.randn(M, K, dtype=torch.float16, device="cuda", requires_grad=True)
        projs_fused = []
        projs_sep = []
        for _ in range(3):
            packed, absmax, codebook, N_padded = _quantize_weight(N, K, k=k)
            A = torch.randn(r, K, dtype=torch.float16, device="cuda", requires_grad=True)
            B = torch.randn(N, r, dtype=torch.float16, device="cuda", requires_grad=True)
            projs_fused.append((packed, absmax, codebook, A, B, 0.5))
            # Separate path uses same data but independent grad computation
            A_sep = A.detach().clone().requires_grad_(True)
            B_sep = B.detach().clone().requires_grad_(True)
            projs_sep.append((packed, absmax, codebook, A_sep, B_sep, 0.5))

        Q, Kp, V = LoRA_QKV_Kbit.apply(
            X_fused,
            *projs_fused[0][:3], projs_fused[0][3], projs_fused[0][4], projs_fused[0][5],
            *projs_fused[1][:3], projs_fused[1][3], projs_fused[1][4], projs_fused[1][5],
            *projs_fused[2][:3], projs_fused[2][3], projs_fused[2][4], projs_fused[2][5],
            k, K, N_padded, N, torch.float16,
        )
        (Q.sum() + Kp.sum() + V.sum()).backward()

        # Separate path
        X_sep = X_fused.detach().clone().requires_grad_(True)
        total = torch.zeros(1, device="cuda")
        for packed, absmax, codebook, A, B, s in projs_sep:
            out = LoRA_W_Kbit.apply(
                X_sep, packed, absmax, codebook, A, B, s, k, K, N_padded, N, torch.float16,
            )
            total = total + out.sum()
        total.backward()

        # Compare grad_X
        diff = (X_fused.grad.float() - X_sep.grad.float()).abs()
        rel_err = (diff / X_sep.grad.float().abs().clamp(min=1e-3)).max().item()
        assert rel_err < 0.02, f"grad_X relative error: {rel_err}"

        # Compare grad_A and grad_B for each projection
        for i in range(3):
            for name, fused, sep in [
                ("A", projs_fused[i][3], projs_sep[i][3]),
                ("B", projs_fused[i][4], projs_sep[i][4]),
            ]:
                diff = (fused.grad.float() - sep.grad.float()).abs()
                rel_err = (diff / sep.grad.float().abs().clamp(min=1e-3)).max().item()
                assert rel_err < 0.02, f"Proj {i} grad_{name} relative error: {rel_err}"


class TestLoRA_MLP_Kbit:
    """Tests for fused gate+up+down MLP with SwiGLU."""

    def _setup(self, M=4, K_in=256, N_hidden=256, K_hidden=256, N_out=256, r=16, k=4):
        """Create gate/up/down weights and LoRA adapters.

        Uses smaller N_hidden to avoid fp16 overflow in SwiGLU activation chain.
        """
        # Use small scale to prevent fp16 overflow in multi-layer chain
        scale = 0.1
        X = (torch.randn(M, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

        packed_gate, absmax_gate, codebook_gate, N_hidden_padded = _quantize_weight(N_hidden, K_in, k=k)
        A_gate = (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_gate = (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

        packed_up, absmax_up, codebook_up, _ = _quantize_weight(N_hidden, K_in, k=k)
        A_up = (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_up = (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

        packed_down, absmax_down, codebook_down, N_out_padded = _quantize_weight(N_out, K_hidden, k=k)
        A_down = (torch.randn(r, K_hidden, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_down = (torch.randn(N_out, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

        s = 0.5
        return (
            X,
            packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s,
            packed_up, absmax_up, codebook_up, A_up, B_up, s,
            packed_down, absmax_down, codebook_down, A_down, B_down, s,
            k, K_in, N_hidden, N_hidden_padded, K_hidden, N_out, N_out_padded,
        )

    def test_forward_correctness(self):
        """Forward should match naive implementation."""
        args = self._setup()
        (
            X,
            packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s_gate,
            packed_up, absmax_up, codebook_up, A_up, B_up, s_up,
            packed_down, absmax_down, codebook_down, A_down, B_down, s_down,
            k, K_in, N_hidden, N_hidden_padded, K_hidden, N_out, N_out_padded,
        ) = args

        out = LoRA_MLP_Kbit.apply(*args, torch.float16)

        # Reference
        W_gate = _dequant_weight(packed_gate, absmax_gate, codebook_gate, k, K_in, N_hidden_padded, N_hidden, torch.float16)
        W_up = _dequant_weight(packed_up, absmax_up, codebook_up, k, K_in, N_hidden_padded, N_hidden, torch.float16)
        W_down = _dequant_weight(packed_down, absmax_down, codebook_down, k, K_hidden, N_out_padded, N_out, torch.float16)

        X_det = X.detach()
        e_ref = X_det @ W_gate.t() + (X_det @ A_gate.detach().t()) @ B_gate.detach().t() * s_gate
        g_ref = X_det @ W_up.t() + (X_det @ A_up.detach().t()) @ B_up.detach().t() * s_up
        h_ref = torch.nn.functional.silu(e_ref) * g_ref
        ref = h_ref @ W_down.t() + (h_ref @ A_down.detach().t()) @ B_down.detach().t() * s_down

        diff = (out.float() - ref.float()).abs()
        assert diff.max().item() < 0.5, f"Forward max diff: {diff.max().item()}"

    def test_gradient_flows(self):
        """All adapter gradients should be computed."""
        args = self._setup()
        (
            X,
            _, _, _, A_gate, B_gate, _,
            _, _, _, A_up, B_up, _,
            _, _, _, A_down, B_down, _,
            *_rest,
        ) = args

        out = LoRA_MLP_Kbit.apply(*args, torch.float16)
        out.sum().backward()

        for name, param in [
            ("A_gate", A_gate), ("B_gate", B_gate),
            ("A_up", A_up), ("B_up", B_up),
            ("A_down", A_down), ("B_down", B_down),
            ("X", X),
        ]:
            assert param.grad is not None, f"{name} gradient is None"
            assert not torch.all(param.grad == 0), f"{name} gradient is all zeros"

    def test_swiglu_backward(self):
        """SwiGLU backward should be correct."""
        # Test SwiGLU backward in isolation
        e = torch.randn(4, 128, dtype=torch.float32, device="cuda", requires_grad=True)
        g = torch.randn(4, 128, dtype=torch.float32, device="cuda", requires_grad=True)

        h = torch.nn.functional.silu(e) * g
        h.sum().backward()

        # Manual reference
        sig_e = torch.sigmoid(e.detach())
        silu_e = e.detach() * sig_e
        # dh/de = g * sigmoid(e) * (1 + e * (1 - sigmoid(e)))
        grad_e_ref = g.detach() * sig_e * (1.0 + e.detach() * (1.0 - sig_e))
        grad_g_ref = silu_e

        diff_e = (e.grad - grad_e_ref).abs().max().item()
        diff_g = (g.grad - grad_g_ref).abs().max().item()
        assert diff_e < 1e-5, f"SwiGLU grad_e diff: {diff_e}"
        assert diff_g < 1e-5, f"SwiGLU grad_g diff: {diff_g}"


class TestInPlaceOutput:
    """Tests for the out= parameter on all LoRA autograd functions."""

    def test_lora_w_kbit_out_forward(self):
        """LoRA_W_Kbit with out= should produce identical output."""
        M, K, N, r, k = 8, 256, 128, 16, 4
        packed, absmax, codebook, N_padded = _quantize_weight(N, K, k=k)
        X = torch.randn(M, K, dtype=torch.float16, device="cuda", requires_grad=True)
        A = torch.randn(r, K, dtype=torch.float16, device="cuda", requires_grad=True)
        B = torch.randn(N, r, dtype=torch.float16, device="cuda", requires_grad=True)
        s = 0.5

        # Without out=
        ref = LoRA_W_Kbit.apply(
            X.detach().requires_grad_(True), packed, absmax, codebook,
            A.detach().clone().requires_grad_(True),
            B.detach().clone().requires_grad_(True),
            s, k, K, N_padded, N, torch.float16,
        )

        # With out=
        out_buf = torch.empty(M, N, dtype=torch.float16, device="cuda")
        result = LoRA_W_Kbit.apply(
            X, packed, absmax, codebook, A, B, s, k, K, N_padded, N, torch.float16, out_buf,
        )

        assert result.data_ptr() == out_buf.data_ptr(), "Result should be the same tensor as out_buf"
        assert torch.allclose(result.float(), ref.float(), atol=0.1, rtol=0.02), \
            f"out= max abs diff: {(result.float() - ref.float()).abs().max().item()}"

    def test_lora_w_kbit_out_backward(self):
        """Gradients should be identical with and without out=."""
        M, K, N, r, k = 8, 256, 128, 16, 4
        packed, absmax, codebook, N_padded = _quantize_weight(N, K, k=k)
        s = 0.5

        # Without out=
        X1 = torch.randn(M, K, dtype=torch.float16, device="cuda", requires_grad=True)
        A1 = torch.randn(r, K, dtype=torch.float16, device="cuda", requires_grad=True)
        B1 = torch.randn(N, r, dtype=torch.float16, device="cuda", requires_grad=True)
        out1 = LoRA_W_Kbit.apply(X1, packed, absmax, codebook, A1, B1, s, k, K, N_padded, N, torch.float16)
        out1.sum().backward()

        # With out=
        X2 = X1.detach().clone().requires_grad_(True)
        A2 = A1.detach().clone().requires_grad_(True)
        B2 = B1.detach().clone().requires_grad_(True)
        out_buf = torch.empty(M, N, dtype=torch.float16, device="cuda")
        out2 = LoRA_W_Kbit.apply(X2, packed, absmax, codebook, A2, B2, s, k, K, N_padded, N, torch.float16, out_buf)
        out2.sum().backward()

        for name, g1, g2 in [("X", X1.grad, X2.grad), ("A", A1.grad, A2.grad), ("B", B1.grad, B2.grad)]:
            diff = (g1.float() - g2.float()).abs()
            rel_err = (diff / g1.float().abs().clamp(min=1e-3)).max().item()
            assert rel_err < 0.02, f"grad_{name} relative error with out=: {rel_err}"

    def test_lora_qkv_kbit_out_forward(self):
        """LoRA_QKV_Kbit with out_q/out_k/out_v should produce identical output."""
        M, K, N, r, k = 8, 256, 128, 16, 4

        projs = []
        for _ in range(3):
            packed, absmax, codebook, N_padded = _quantize_weight(N, K, k=k)
            A = torch.randn(r, K, dtype=torch.float16, device="cuda", requires_grad=True)
            B = torch.randn(N, r, dtype=torch.float16, device="cuda", requires_grad=True)
            projs.append((packed, absmax, codebook, A, B, 0.5))

        X = torch.randn(M, K, dtype=torch.float16, device="cuda", requires_grad=True)

        # Without out=
        Q_ref, K_ref, V_ref = LoRA_QKV_Kbit.apply(
            X.detach().requires_grad_(True),
            *projs[0][:3], projs[0][3].detach().clone().requires_grad_(True), projs[0][4].detach().clone().requires_grad_(True), projs[0][5],
            *projs[1][:3], projs[1][3].detach().clone().requires_grad_(True), projs[1][4].detach().clone().requires_grad_(True), projs[1][5],
            *projs[2][:3], projs[2][3].detach().clone().requires_grad_(True), projs[2][4].detach().clone().requires_grad_(True), projs[2][5],
            k, K, N_padded, N, torch.float16,
        )

        # With out=
        out_q = torch.empty(M, N, dtype=torch.float16, device="cuda")
        out_k = torch.empty(M, N, dtype=torch.float16, device="cuda")
        out_v = torch.empty(M, N, dtype=torch.float16, device="cuda")
        Q, Kp, V = LoRA_QKV_Kbit.apply(
            X,
            *projs[0][:3], projs[0][3], projs[0][4], projs[0][5],
            *projs[1][:3], projs[1][3], projs[1][4], projs[1][5],
            *projs[2][:3], projs[2][3], projs[2][4], projs[2][5],
            k, K, N_padded, N, torch.float16,
            out_q, out_k, out_v,
        )

        assert Q.data_ptr() == out_q.data_ptr(), "Q should be the same tensor as out_q"
        assert Kp.data_ptr() == out_k.data_ptr(), "K should be the same tensor as out_k"
        assert V.data_ptr() == out_v.data_ptr(), "V should be the same tensor as out_v"

        for name, result, ref in [("Q", Q, Q_ref), ("K", Kp, K_ref), ("V", V, V_ref)]:
            assert torch.allclose(result.float(), ref.float(), atol=0.1, rtol=0.02), \
                f"{name} out= max abs diff: {(result.float() - ref.float()).abs().max().item()}"

    def test_lora_qkv_kbit_out_backward(self):
        """QKV gradients should be identical with and without out=."""
        M, K, N, r, k = 4, 256, 128, 16, 4

        packed_list = []
        for _ in range(3):
            packed, absmax, codebook, N_padded = _quantize_weight(N, K, k=k)
            packed_list.append((packed, absmax, codebook, N_padded))

        s = 0.5

        # Without out=
        X1 = torch.randn(M, K, dtype=torch.float16, device="cuda", requires_grad=True)
        As1 = [torch.randn(r, K, dtype=torch.float16, device="cuda", requires_grad=True) for _ in range(3)]
        Bs1 = [torch.randn(N, r, dtype=torch.float16, device="cuda", requires_grad=True) for _ in range(3)]
        Q1, K1, V1 = LoRA_QKV_Kbit.apply(
            X1,
            packed_list[0][0], packed_list[0][1], packed_list[0][2], As1[0], Bs1[0], s,
            packed_list[1][0], packed_list[1][1], packed_list[1][2], As1[1], Bs1[1], s,
            packed_list[2][0], packed_list[2][1], packed_list[2][2], As1[2], Bs1[2], s,
            k, K, N_padded, N, torch.float16,
        )
        (Q1.sum() + K1.sum() + V1.sum()).backward()

        # With out=
        X2 = X1.detach().clone().requires_grad_(True)
        As2 = [a.detach().clone().requires_grad_(True) for a in As1]
        Bs2 = [b.detach().clone().requires_grad_(True) for b in Bs1]
        bufs = [torch.empty(M, N, dtype=torch.float16, device="cuda") for _ in range(3)]
        Q2, K2, V2 = LoRA_QKV_Kbit.apply(
            X2,
            packed_list[0][0], packed_list[0][1], packed_list[0][2], As2[0], Bs2[0], s,
            packed_list[1][0], packed_list[1][1], packed_list[1][2], As2[1], Bs2[1], s,
            packed_list[2][0], packed_list[2][1], packed_list[2][2], As2[2], Bs2[2], s,
            k, K, N_padded, N, torch.float16,
            bufs[0], bufs[1], bufs[2],
        )
        (Q2.sum() + K2.sum() + V2.sum()).backward()

        diff = (X1.grad.float() - X2.grad.float()).abs()
        rel_err = (diff / X1.grad.float().abs().clamp(min=1e-3)).max().item()
        assert rel_err < 0.02, f"grad_X relative error with out=: {rel_err}"
        for i in range(3):
            for name, g1, g2 in [("A", As1[i].grad, As2[i].grad), ("B", Bs1[i].grad, Bs2[i].grad)]:
                diff = (g1.float() - g2.float()).abs()
                rel_err = (diff / g1.float().abs().clamp(min=1e-3)).max().item()
                assert rel_err < 0.02, f"Proj {i} grad_{name} relative error with out=: {rel_err}"

    def test_lora_mlp_kbit_out_forward(self):
        """LoRA_MLP_Kbit with out= should produce identical output."""
        M, K_in, N_hidden, K_hidden, N_out, r, k = 4, 256, 256, 256, 256, 16, 4
        scale = 0.1
        X = (torch.randn(M, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

        packed_gate, absmax_gate, codebook_gate, N_hidden_padded = _quantize_weight(N_hidden, K_in, k=k)
        A_gate = (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_gate = (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

        packed_up, absmax_up, codebook_up, _ = _quantize_weight(N_hidden, K_in, k=k)
        A_up = (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_up = (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

        packed_down, absmax_down, codebook_down, N_out_padded = _quantize_weight(N_out, K_hidden, k=k)
        A_down = (torch.randn(r, K_hidden, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_down = (torch.randn(N_out, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

        s = 0.5
        common_args = (
            packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s,
            packed_up, absmax_up, codebook_up, A_up, B_up, s,
            packed_down, absmax_down, codebook_down, A_down, B_down, s,
            k, K_in, N_hidden, N_hidden_padded, K_hidden, N_out, N_out_padded,
            torch.float16,
        )

        # Without out=
        ref = LoRA_MLP_Kbit.apply(X.detach().requires_grad_(True), *common_args)

        # With out=
        out_buf = torch.empty(M, N_out, dtype=torch.float16, device="cuda")
        result = LoRA_MLP_Kbit.apply(X, *common_args, out_buf)

        assert result.data_ptr() == out_buf.data_ptr(), "Result should be the same tensor as out_buf"
        diff = (result.float() - ref.float()).abs()
        rel_err = (diff / ref.float().abs().clamp(min=1e-3)).max().item()
        assert rel_err < 0.02, f"MLP out= relative error: {rel_err}"

    def test_lora_mlp_kbit_out_backward(self):
        """MLP gradients should be identical with and without out=."""
        M, K_in, N_hidden, K_hidden, N_out, r, k = 4, 256, 256, 256, 256, 16, 4
        scale = 0.1

        packed_gate, absmax_gate, codebook_gate, N_hidden_padded = _quantize_weight(N_hidden, K_in, k=k)
        packed_up, absmax_up, codebook_up, _ = _quantize_weight(N_hidden, K_in, k=k)
        packed_down, absmax_down, codebook_down, N_out_padded = _quantize_weight(N_out, K_hidden, k=k)
        s = 0.5

        # Without out=
        X1 = (torch.randn(M, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        A_gate1 = (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_gate1 = (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        A_up1 = (torch.randn(r, K_in, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_up1 = (torch.randn(N_hidden, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        A_down1 = (torch.randn(r, K_hidden, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)
        B_down1 = (torch.randn(N_out, r, dtype=torch.float16, device="cuda") * scale).requires_grad_(True)

        out1 = LoRA_MLP_Kbit.apply(
            X1,
            packed_gate, absmax_gate, codebook_gate, A_gate1, B_gate1, s,
            packed_up, absmax_up, codebook_up, A_up1, B_up1, s,
            packed_down, absmax_down, codebook_down, A_down1, B_down1, s,
            k, K_in, N_hidden, N_hidden_padded, K_hidden, N_out, N_out_padded,
            torch.float16,
        )
        out1.sum().backward()

        # With out=
        X2 = X1.detach().clone().requires_grad_(True)
        A_gate2 = A_gate1.detach().clone().requires_grad_(True)
        B_gate2 = B_gate1.detach().clone().requires_grad_(True)
        A_up2 = A_up1.detach().clone().requires_grad_(True)
        B_up2 = B_up1.detach().clone().requires_grad_(True)
        A_down2 = A_down1.detach().clone().requires_grad_(True)
        B_down2 = B_down1.detach().clone().requires_grad_(True)

        out_buf = torch.empty(M, N_out, dtype=torch.float16, device="cuda")
        out2 = LoRA_MLP_Kbit.apply(
            X2,
            packed_gate, absmax_gate, codebook_gate, A_gate2, B_gate2, s,
            packed_up, absmax_up, codebook_up, A_up2, B_up2, s,
            packed_down, absmax_down, codebook_down, A_down2, B_down2, s,
            k, K_in, N_hidden, N_hidden_padded, K_hidden, N_out, N_out_padded,
            torch.float16,
            out_buf,
        )
        out2.sum().backward()

        diff = (X1.grad.float() - X2.grad.float()).abs()
        rel_err = (diff / X1.grad.float().abs().clamp(min=1e-3)).max().item()
        assert rel_err < 0.02, f"MLP grad_X relative error with out=: {rel_err}"
        for name, g1, g2 in [
            ("A_gate", A_gate1.grad, A_gate2.grad), ("B_gate", B_gate1.grad, B_gate2.grad),
            ("A_up", A_up1.grad, A_up2.grad), ("B_up", B_up1.grad, B_up2.grad),
            ("A_down", A_down1.grad, A_down2.grad), ("B_down", B_down1.grad, B_down2.grad),
        ]:
            diff = (g1.float() - g2.float()).abs()
            rel_err = (diff / g1.float().abs().clamp(min=1e-3)).max().item()
            assert rel_err < 0.02, f"MLP grad_{name} relative error with out=: {rel_err}"
