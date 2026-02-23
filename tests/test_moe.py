"""Tests for MoE router dispatch and chunked expert forward pass.

Verifies:
- Router dispatch: all tokens assigned to exactly top_k experts,
  expert weights sum to ~1.0, gather/scatter round-trip, offsets
- Expert forward: matches naive per-expert sequential computation,
  gradients flow through gather/scatter, chunk-size invariance
"""

import pytest
import torch
from scipy.stats import norm

import bitsandbytes  # noqa: F401 (loads CUDA ops)
from bitsandbytes.functional import quantize_kbit
from bitsandbytes.moe import moe_router_dispatch, moe_expert_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ─── Helpers ──────────────────────────────────────────────────────────────

def create_normal_float_codebook(k: int) -> torch.Tensor:
    n_levels = 1 << k
    quantiles = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    values = torch.tensor(norm.ppf(quantiles.numpy()), dtype=torch.float32)
    values = values / values.abs().max()
    return values


def quantize_expert_weights(num_experts, N, K, k):
    """Quantize expert weights in flat format (from quantize_kbit).

    Returns:
        packed_all: concatenated packed tensors for all experts
        absmax_all: concatenated absmax tensors for all experts
        codebook: shared codebook
        W_list: list of original weight matrices (for reference)
    """
    codebook = create_normal_float_codebook(k).cuda()

    # Pad N to multiple of 128 (required by kbit format)
    N_padded = ((N + 127) // 128) * 128

    packed_list = []
    absmax_list = []
    W_list = []

    for _ in range(num_experts):
        W = torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.1
        # Pad to N_padded
        if N != N_padded:
            W_padded = torch.nn.functional.pad(W, (0, 0, 0, N_padded - N))
        else:
            W_padded = W
        packed, absmax, _ = quantize_kbit(W_padded.flatten(), k=k, codebook=codebook)
        packed_list.append(packed)
        absmax_list.append(absmax)
        W_list.append(W)

    packed_all = torch.cat(packed_list, dim=0)
    absmax_all = torch.cat(absmax_list, dim=0)

    return packed_all, absmax_all, codebook, W_list


def setup_moe_expert_weights(num_experts, hidden_dim, intermediate_dim, k):
    """Set up gate/up/down expert weights for MoE testing.

    Returns a dict with all the quantized expert weights and reference weights.
    """
    gate_packed, gate_absmax, codebook, gate_W = quantize_expert_weights(
        num_experts, intermediate_dim, hidden_dim, k,
    )
    up_packed, up_absmax, _, up_W = quantize_expert_weights(
        num_experts, intermediate_dim, hidden_dim, k,
    )
    down_packed, down_absmax, _, down_W = quantize_expert_weights(
        num_experts, hidden_dim, intermediate_dim, k,
    )
    return {
        "gate_packed": gate_packed, "gate_absmax": gate_absmax,
        "up_packed": up_packed, "up_absmax": up_absmax,
        "down_packed": down_packed, "down_absmax": down_absmax,
        "codebook": codebook,
        "gate_W": gate_W, "up_W": up_W, "down_W": down_W,
    }


def naive_moe_forward(hidden, router_result, gate_W, up_W, down_W):
    """Naive per-expert sequential MoE forward for reference.

    Uses the original (dequantized-equivalent) weight matrices.
    """
    N = hidden.shape[0]
    output = torch.zeros_like(hidden)
    expert_indices = router_result["expert_indices"]
    expert_weights = router_result["expert_weights"]
    num_experts = len(gate_W)

    for e in range(num_experts):
        token_indices = router_result["token_indices_per_expert"][e]
        if len(token_indices) == 0:
            continue

        A = hidden[token_indices]  # [n_e, hidden_dim]

        # Gate + Up + SwiGLU + Down
        gate_out = A @ gate_W[e].t()
        up_out = A @ up_W[e].t()
        h = torch.nn.functional.silu(gate_out) * up_out
        down_out = h @ down_W[e].t()

        # Scatter with weights
        for i, tok_idx in enumerate(token_indices):
            slot_mask = expert_indices[tok_idx] == e
            weight = expert_weights[tok_idx][slot_mask].sum()
            output[tok_idx] += down_out[i] * weight

    return output


# ─── Router Dispatch Tests ────────────────────────────────────────────────

class TestMoERouterDispatch:

    def test_basic_routing(self):
        """Basic routing with 8 experts, top-2."""
        N, D = 32, 128
        num_experts, top_k = 8, 2
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        assert result["expert_indices"].shape == (N, top_k)
        assert result["expert_weights"].shape == (N, top_k)
        assert len(result["token_indices_per_expert"]) == num_experts
        assert result["expert_offsets"].shape == (num_experts + 1,)
        assert "sorted_weights" in result
        assert result["sorted_weights"].shape == (N * top_k,)

    def test_all_tokens_assigned_top_k(self):
        """Each token should be assigned to exactly top_k experts."""
        N, D = 64, 128
        num_experts, top_k = 16, 4
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        assert result["expert_indices"].shape == (N, top_k)
        assert (result["expert_indices"] >= 0).all()
        assert (result["expert_indices"] < num_experts).all()

        for i in range(N):
            experts = result["expert_indices"][i]
            assert len(experts.unique()) == top_k, f"Token {i} has duplicate experts"

    def test_expert_weights_sum_to_one(self):
        """Expert weights should sum to ~1.0 per token (softmax)."""
        N, D = 32, 64
        num_experts, top_k = 8, 2
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        weight_sums = result["expert_weights"].float().sum(dim=-1)
        torch.testing.assert_close(
            weight_sums,
            torch.ones(N, device="cuda"),
            atol=1e-3, rtol=1e-3,
        )

    def test_expert_weights_positive(self):
        """All expert weights should be positive (softmax output)."""
        N, D = 32, 64
        num_experts, top_k = 8, 2
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)
        assert (result["expert_weights"] > 0).all()

    def test_expert_offsets_consistency(self):
        """Expert offsets should be consistent with token counts."""
        N, D = 64, 128
        num_experts, top_k = 8, 2
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        offsets = result["expert_offsets"]
        assert offsets[0] == 0
        assert offsets[-1] == N * top_k

        for e in range(num_experts):
            expected_count = len(result["token_indices_per_expert"][e])
            actual_count = (offsets[e + 1] - offsets[e]).item()
            assert expected_count == actual_count, \
                f"Expert {e}: expected {expected_count}, got {actual_count}"

    def test_gather_scatter_round_trip(self):
        """Gathering and scattering should recover all token contributions."""
        N, D = 16, 64
        num_experts, top_k = 4, 2
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        # Simulate: for each expert, gather assigned tokens, process (identity),
        # scatter-add with weights
        output = torch.zeros_like(hidden)
        for e in range(num_experts):
            token_indices = result["token_indices_per_expert"][e]
            if len(token_indices) == 0:
                continue

            for idx in token_indices:
                token_experts = result["expert_indices"][idx]
                slot = (token_experts == e).nonzero(as_tuple=True)[0]
                weight = result["expert_weights"][idx, slot]
                output[idx] += hidden[idx] * weight

        torch.testing.assert_close(
            output.float(), hidden.float(),
            atol=1e-3, rtol=1e-3,
        )

    @pytest.mark.parametrize("top_k", [1, 2, 4, 8])
    def test_different_top_k(self, top_k):
        """Different top_k values should work correctly."""
        N, D = 32, 64
        num_experts = 16
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        assert result["expert_indices"].shape == (N, top_k)
        assert result["expert_weights"].shape == (N, top_k)
        assert result["expert_offsets"][-1] == N * top_k

    def test_sorted_indices(self):
        """Sorted indices should be sorted by expert ID."""
        N, D = 32, 64
        num_experts, top_k = 8, 2
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        sorted_experts = result["sorted_expert_indices"]
        assert (sorted_experts[1:] >= sorted_experts[:-1]).all()

    def test_sorted_weights_consistency(self):
        """sorted_weights should match the weights for each (token, expert) pair."""
        N, D = 16, 64
        num_experts, top_k = 4, 2
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        sorted_tok = result["sorted_token_indices"]
        sorted_exp = result["sorted_expert_indices"]
        sorted_w = result["sorted_weights"]

        for i in range(sorted_tok.shape[0]):
            tok = sorted_tok[i].item()
            exp = sorted_exp[i].item()
            # Find the slot in expert_indices for this (tok, exp) pair
            slot = (result["expert_indices"][tok] == exp).nonzero(as_tuple=True)[0]
            expected_w = result["expert_weights"][tok, slot].item()
            actual_w = sorted_w[i].item()
            assert abs(expected_w - actual_w) < 1e-3, \
                f"Weight mismatch at sorted pos {i}: expected {expected_w}, got {actual_w}"

    def test_single_token(self):
        """Edge case: single token."""
        N, D = 1, 64
        num_experts, top_k = 4, 2
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        assert result["expert_indices"].shape == (1, top_k)
        assert result["expert_offsets"][-1] == top_k

    def test_many_experts(self):
        """Test with 512 experts (Qwen3.5 scale)."""
        N, D = 64, 128
        num_experts, top_k = 512, 8
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        assert result["expert_indices"].shape == (N, top_k)
        assert result["expert_offsets"][-1] == N * top_k
        assert len(result["token_indices_per_expert"]) == num_experts


# ─── Expert Forward Tests ─────────────────────────────────────────────────

class TestMoEExpertForward:

    @pytest.fixture
    def moe_setup(self):
        """Set up MoE expert weights and router for testing."""
        num_experts = 4
        hidden_dim = 256
        intermediate_dim = 512  # Must be multiple of 128
        k = 4
        top_k = 2
        N_tokens = 16

        weights = setup_moe_expert_weights(num_experts, hidden_dim, intermediate_dim, k)

        hidden = torch.randn(N_tokens, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)
        router_result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        return {
            "hidden": hidden,
            "router_result": router_result,
            "weights": weights,
            "num_experts": num_experts,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "k": k,
            "top_k": top_k,
            "N_tokens": N_tokens,
        }

    def test_forward_output_shape(self, moe_setup):
        """Output should have same shape as input."""
        s = moe_setup
        output = moe_expert_forward(
            s["hidden"], s["router_result"],
            s["weights"]["gate_packed"], s["weights"]["gate_absmax"],
            s["weights"]["up_packed"], s["weights"]["up_absmax"],
            s["weights"]["down_packed"], s["weights"]["down_absmax"],
            s["weights"]["codebook"], s["k"],
            s["hidden_dim"], s["intermediate_dim"],
            s["num_experts"], expert_chunk_size=2,
        )
        assert output.shape == (s["N_tokens"], s["hidden_dim"])
        assert output.dtype == torch.float16
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_forward_matches_naive(self, moe_setup):
        """Chunked expert forward should match naive per-expert computation.

        We compare against a naive implementation that uses the dequantized
        weights from quantize_kbit, so the comparison validates the quantized
        weight handling, gather/scatter, and SwiGLU computation.
        """
        s = moe_setup
        w = s["weights"]

        # Get dequantized weights for naive reference
        from bitsandbytes.functional import dequantize_kbit
        inter_padded = ((s["intermediate_dim"] + 127) // 128) * 128
        hidden_padded = ((s["hidden_dim"] + 127) // 128) * 128
        n_gate = inter_padded * s["hidden_dim"]
        n_down = hidden_padded * s["intermediate_dim"]

        packed_per_gate = w["gate_packed"].numel() // s["num_experts"]
        absmax_per_gate = w["gate_absmax"].numel() // s["num_experts"]
        packed_per_down = w["down_packed"].numel() // s["num_experts"]
        absmax_per_down = w["down_absmax"].numel() // s["num_experts"]

        deq_gate = []
        deq_up = []
        deq_down = []
        for e in range(s["num_experts"]):
            # Gate
            p = w["gate_packed"][e * packed_per_gate: (e + 1) * packed_per_gate]
            a = w["gate_absmax"][e * absmax_per_gate: (e + 1) * absmax_per_gate]
            W = dequantize_kbit(p, a, w["codebook"], s["k"], n_gate, torch.float16)
            deq_gate.append(W[:n_gate].reshape(inter_padded, s["hidden_dim"])[:s["intermediate_dim"]])
            # Up
            p = w["up_packed"][e * packed_per_gate: (e + 1) * packed_per_gate]
            a = w["up_absmax"][e * absmax_per_gate: (e + 1) * absmax_per_gate]
            W = dequantize_kbit(p, a, w["codebook"], s["k"], n_gate, torch.float16)
            deq_up.append(W[:n_gate].reshape(inter_padded, s["hidden_dim"])[:s["intermediate_dim"]])
            # Down
            p = w["down_packed"][e * packed_per_down: (e + 1) * packed_per_down]
            a = w["down_absmax"][e * absmax_per_down: (e + 1) * absmax_per_down]
            W = dequantize_kbit(p, a, w["codebook"], s["k"], n_down, torch.float16)
            deq_down.append(W[:n_down].reshape(hidden_padded, s["intermediate_dim"])[:s["hidden_dim"]])

        # Naive forward with dequantized weights
        naive_out = naive_moe_forward(
            s["hidden"], s["router_result"], deq_gate, deq_up, deq_down,
        )

        # Chunked forward
        chunked_out = moe_expert_forward(
            s["hidden"], s["router_result"],
            w["gate_packed"], w["gate_absmax"],
            w["up_packed"], w["up_absmax"],
            w["down_packed"], w["down_absmax"],
            w["codebook"], s["k"],
            s["hidden_dim"], s["intermediate_dim"],
            s["num_experts"], expert_chunk_size=2,
        )

        torch.testing.assert_close(
            chunked_out.float(), naive_out.float(),
            atol=1e-2, rtol=1e-2,
        )

    def test_chunk_size_invariance(self, moe_setup):
        """Output should be the same regardless of expert_chunk_size."""
        s = moe_setup
        w = s["weights"]

        results = []
        for chunk_size in [1, 2, 4, s["num_experts"]]:
            out = moe_expert_forward(
                s["hidden"], s["router_result"],
                w["gate_packed"], w["gate_absmax"],
                w["up_packed"], w["up_absmax"],
                w["down_packed"], w["down_absmax"],
                w["codebook"], s["k"],
                s["hidden_dim"], s["intermediate_dim"],
                s["num_experts"], expert_chunk_size=chunk_size,
            )
            results.append(out)

        for i in range(1, len(results)):
            torch.testing.assert_close(
                results[0].float(), results[i].float(),
                atol=1e-4, rtol=1e-4,
                msg=f"chunk_size={[1, 2, 4, s['num_experts']][i]} differs from chunk_size=1",
            )

    def test_backward_produces_gradient(self, moe_setup):
        """Backward pass should produce non-zero gradient for hidden input."""
        s = moe_setup
        w = s["weights"]

        hidden = s["hidden"].clone().requires_grad_(True)
        router_result = moe_router_dispatch(
            hidden.detach(),
            torch.randn(s["num_experts"], s["hidden_dim"], device="cuda", dtype=torch.float16),
            s["num_experts"], s["top_k"],
        )

        output = moe_expert_forward(
            hidden, router_result,
            w["gate_packed"], w["gate_absmax"],
            w["up_packed"], w["up_absmax"],
            w["down_packed"], w["down_absmax"],
            w["codebook"], s["k"],
            s["hidden_dim"], s["intermediate_dim"],
            s["num_experts"], expert_chunk_size=2,
        )

        loss = output.sum()
        loss.backward()

        assert hidden.grad is not None, "No gradient computed for hidden"
        assert hidden.grad.shape == hidden.shape
        assert (hidden.grad != 0).any(), "All gradients are zero"
        assert torch.isfinite(hidden.grad).all(), "Gradient contains non-finite values"

    def test_gradient_numerical_check(self):
        """Numerical gradient check for the MoE expert forward."""
        num_experts = 2
        hidden_dim = 128
        intermediate_dim = 256
        k = 4
        top_k = 1
        N_tokens = 4

        weights = setup_moe_expert_weights(num_experts, hidden_dim, intermediate_dim, k)
        w = weights

        # Use float32 for numerical gradient check
        hidden = torch.randn(N_tokens, hidden_dim, device="cuda", dtype=torch.float32) * 0.05
        hidden.requires_grad_(True)

        # Fixed routing: deterministic
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float32)
        router_result = moe_router_dispatch(hidden.detach(), router_weight, num_experts, top_k)

        def func(h):
            return moe_expert_forward(
                h, router_result,
                w["gate_packed"], w["gate_absmax"],
                w["up_packed"], w["up_absmax"],
                w["down_packed"], w["down_absmax"],
                w["codebook"], k,
                hidden_dim, intermediate_dim,
                num_experts, expert_chunk_size=1,
            ).sum()

        # Compute analytical gradient
        output = func(hidden)
        output.backward()
        analytical_grad = hidden.grad.clone()

        # Compute numerical gradient
        eps = 1e-3
        numerical_grad = torch.zeros_like(hidden)
        for i in range(N_tokens):
            for j in range(min(4, hidden_dim)):  # Only check first 4 dims for speed
                h_plus = hidden.detach().clone()
                h_plus[i, j] += eps
                h_minus = hidden.detach().clone()
                h_minus[i, j] -= eps

                f_plus = func(h_plus).item()
                f_minus = func(h_minus).item()
                numerical_grad[i, j] = (f_plus - f_minus) / (2 * eps)

        # Compare (only the dims we computed numerically)
        for i in range(N_tokens):
            for j in range(min(4, hidden_dim)):
                a = analytical_grad[i, j].item()
                n = numerical_grad[i, j].item()
                if abs(n) > 1e-5:  # Only check where numerical grad is meaningful
                    rel_err = abs(a - n) / (abs(n) + 1e-8)
                    assert rel_err < 0.1, (
                        f"Gradient mismatch at [{i},{j}]: analytical={a:.6f}, "
                        f"numerical={n:.6f}, rel_err={rel_err:.4f}"
                    )

    def test_gradient_accumulation(self, moe_setup):
        """Gradients should accumulate correctly across multiple forward passes."""
        s = moe_setup
        w = s["weights"]

        hidden = s["hidden"].clone().requires_grad_(True)
        router_result = moe_router_dispatch(
            hidden.detach(),
            torch.randn(s["num_experts"], s["hidden_dim"], device="cuda", dtype=torch.float16),
            s["num_experts"], s["top_k"],
        )

        # Two forward passes with the same input
        out1 = moe_expert_forward(
            hidden, router_result,
            w["gate_packed"], w["gate_absmax"],
            w["up_packed"], w["up_absmax"],
            w["down_packed"], w["down_absmax"],
            w["codebook"], s["k"],
            s["hidden_dim"], s["intermediate_dim"],
            s["num_experts"], expert_chunk_size=2,
        )
        out2 = moe_expert_forward(
            hidden, router_result,
            w["gate_packed"], w["gate_absmax"],
            w["up_packed"], w["up_absmax"],
            w["down_packed"], w["down_absmax"],
            w["codebook"], s["k"],
            s["hidden_dim"], s["intermediate_dim"],
            s["num_experts"], expert_chunk_size=2,
        )

        loss = out1.sum() + out2.sum()
        loss.backward()

        assert hidden.grad is not None
        # Gradient should be 2x a single pass
        hidden2 = s["hidden"].clone().requires_grad_(True)
        out_single = moe_expert_forward(
            hidden2, router_result,
            w["gate_packed"], w["gate_absmax"],
            w["up_packed"], w["up_absmax"],
            w["down_packed"], w["down_absmax"],
            w["codebook"], s["k"],
            s["hidden_dim"], s["intermediate_dim"],
            s["num_experts"], expert_chunk_size=2,
        )
        out_single.sum().backward()

        torch.testing.assert_close(
            hidden.grad.float(), (2.0 * hidden2.grad).float(),
            atol=1e-3, rtol=1e-3,
        )

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_different_k_values(self, k):
        """Expert forward should work with different bit widths."""
        num_experts = 2
        hidden_dim = 128
        intermediate_dim = 256
        top_k = 1
        N_tokens = 8

        weights = setup_moe_expert_weights(num_experts, hidden_dim, intermediate_dim, k)

        hidden = torch.randn(N_tokens, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)
        router_result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        output = moe_expert_forward(
            hidden, router_result,
            weights["gate_packed"], weights["gate_absmax"],
            weights["up_packed"], weights["up_absmax"],
            weights["down_packed"], weights["down_absmax"],
            weights["codebook"], k,
            hidden_dim, intermediate_dim,
            num_experts, expert_chunk_size=1,
        )

        assert output.shape == (N_tokens, hidden_dim)
        assert torch.isfinite(output).all()

    def test_empty_expert(self):
        """Handle experts with no tokens routed to them."""
        num_experts = 8
        hidden_dim = 128
        intermediate_dim = 256
        k = 4
        top_k = 1
        N_tokens = 2  # With 8 experts and only 2 tokens, most experts get 0 tokens

        weights = setup_moe_expert_weights(num_experts, hidden_dim, intermediate_dim, k)

        hidden = torch.randn(N_tokens, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)
        router_result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        output = moe_expert_forward(
            hidden, router_result,
            weights["gate_packed"], weights["gate_absmax"],
            weights["up_packed"], weights["up_absmax"],
            weights["down_packed"], weights["down_absmax"],
            weights["codebook"], k,
            hidden_dim, intermediate_dim,
            num_experts, expert_chunk_size=4,
        )

        assert output.shape == (N_tokens, hidden_dim)
        assert torch.isfinite(output).all()

    def test_backward_chunk_size_invariance(self):
        """Gradients should be the same regardless of expert_chunk_size."""
        num_experts = 4
        hidden_dim = 128
        intermediate_dim = 256
        k = 4
        top_k = 2
        N_tokens = 8

        weights = setup_moe_expert_weights(num_experts, hidden_dim, intermediate_dim, k)
        w = weights

        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)
        base_hidden = torch.randn(N_tokens, hidden_dim, device="cuda", dtype=torch.float16) * 0.1

        # Use same routing for all chunk sizes
        router_result = moe_router_dispatch(base_hidden, router_weight, num_experts, top_k)

        grads = []
        for chunk_size in [1, 2, 4]:
            hidden = base_hidden.clone().requires_grad_(True)
            out = moe_expert_forward(
                hidden, router_result,
                w["gate_packed"], w["gate_absmax"],
                w["up_packed"], w["up_absmax"],
                w["down_packed"], w["down_absmax"],
                w["codebook"], k,
                hidden_dim, intermediate_dim,
                num_experts, expert_chunk_size=chunk_size,
            )
            out.sum().backward()
            grads.append(hidden.grad.clone())

        for i in range(1, len(grads)):
            torch.testing.assert_close(
                grads[0].float(), grads[i].float(),
                atol=1e-4, rtol=1e-4,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
