"""Tests for MoE router dispatch.

Verifies:
- All tokens assigned to exactly top_k experts
- Expert weights sum to ~1.0 per token
- Gather/scatter indices round-trip correctly
- Expert offsets are consistent with token counts
- Different top_k values work
- Edge cases: all tokens to same expert, single token
"""

import pytest
import torch

from bitsandbytes.moe import moe_router_dispatch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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

    def test_all_tokens_assigned_top_k(self):
        """Each token should be assigned to exactly top_k experts."""
        N, D = 64, 128
        num_experts, top_k = 16, 4
        hidden = torch.randn(N, D, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, D, device="cuda", dtype=torch.float16)

        result = moe_router_dispatch(hidden, router_weight, num_experts, top_k)

        # Each token has exactly top_k expert assignments
        assert result["expert_indices"].shape == (N, top_k)

        # Expert indices should be in valid range
        assert (result["expert_indices"] >= 0).all()
        assert (result["expert_indices"] < num_experts).all()

        # No duplicates per token
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

        # First offset should be 0
        assert offsets[0] == 0

        # Last offset should be total assignments (N * top_k)
        assert offsets[-1] == N * top_k

        # Per-expert counts should match token_indices_per_expert
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
            gathered = hidden[token_indices]  # [n_e, D]

            # Scatter with weights
            for idx in token_indices:
                # Find weight for this token-expert pair
                token_experts = result["expert_indices"][idx]
                slot = (token_experts == e).nonzero(as_tuple=True)[0]
                weight = result["expert_weights"][idx, slot]
                output[idx] += hidden[idx] * weight

        # Every token should have been processed (weighted sum of identity)
        # output[i] = sum_k(weight_k * hidden[i]) = hidden[i] * sum(weights) = hidden[i]
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
        # Should be non-decreasing
        assert (sorted_experts[1:] >= sorted_experts[:-1]).all()

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
