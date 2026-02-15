"""
Tests for kbit grouped expert GEMM kernel.

Verifies correctness by comparing grouped GEMM output against individual
kbit_gemm_prod calls for each expert.
"""

import pytest
import torch
from scipy.stats import norm

import bitsandbytes  # noqa: F401
from bitsandbytes import _ops  # noqa: F401

BLOCKSIZE = 32


def create_normal_float_codebook(k: int) -> torch.Tensor:
    n_levels = 1 << k
    quantiles = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    values = torch.tensor(norm.ppf(quantiles.numpy()), dtype=torch.float32)
    values = values / values.abs().max()
    return values


def prepare_expert_weights(K_dim, N, k, num_experts):
    """Quantize and repack weights for multiple experts.
    Returns (B_packed_all, B_absmax_all, codebook, W_list) where
    B_packed_all and B_absmax_all are concatenated across experts.
    """
    codebook = create_normal_float_codebook(k).cuda()

    packed_list = []
    absmax_list = []
    W_list = []

    for _ in range(num_experts):
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        packed_flat, absmax = torch.ops.bitsandbytes.quantize_kbit(
            W.reshape(-1), codebook, k
        )
        packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
            packed_flat, absmax.cuda(), K_dim, N, k
        )
        packed_list.append(packed_tiled)
        absmax_list.append(absmax_tiled)
        W_list.append(W)

    B_packed_all = torch.cat(packed_list, dim=0)
    B_absmax_all = torch.cat(absmax_list, dim=0)

    return B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list


class TestGroupedGemm:
    """Test grouped expert GEMM against individual kbit_gemm_prod calls."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_basic_correctness(self, k):
        """Basic test: all experts have same M, compare against individual calls."""
        K_dim, N = 2048, 512
        num_experts = 8
        M_per_expert = 4

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = (
            prepare_expert_weights(K_dim, N, k, num_experts)
        )

        # Build activations and expert_offsets
        A_list = []
        offsets = [0]
        for i in range(num_experts):
            A_i = torch.randn(M_per_expert, K_dim, dtype=torch.float16, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + M_per_expert)

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        # Grouped GEMM
        C_grouped = torch.ops.bitsandbytes.kbit_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, k, num_experts,
        )

        # Individual GEMM for each expert
        C_individual_list = []
        for i in range(num_experts):
            C_i = torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i], packed_list[i], absmax_list[i], codebook,
                K_dim, N, k, 1,
            )
            C_individual_list.append(C_i)
        C_individual = torch.cat(C_individual_list, dim=0)

        # Compare
        assert C_grouped.shape == C_individual.shape, (
            f"Shape mismatch: {C_grouped.shape} vs {C_individual.shape}"
        )
        assert torch.allclose(C_grouped, C_individual, rtol=1e-3, atol=1e-3), (
            f"Max diff: {(C_grouped - C_individual).abs().max().item():.6f}, "
            f"Mean diff: {(C_grouped - C_individual).abs().mean().item():.6f}"
        )

    @pytest.mark.parametrize("k", [4])
    def test_variable_M(self, k):
        """Experts with different M values."""
        K_dim, N = 2048, 512
        num_experts = 8
        M_values = [1, 3, 7, 2, 5, 1, 4, 8]

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = (
            prepare_expert_weights(K_dim, N, k, num_experts)
        )

        A_list = []
        offsets = [0]
        for i in range(num_experts):
            A_i = torch.randn(M_values[i], K_dim, dtype=torch.float16, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + M_values[i])

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.kbit_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, k, num_experts,
        )

        C_individual_list = []
        for i in range(num_experts):
            C_i = torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i], packed_list[i], absmax_list[i], codebook,
                K_dim, N, k, 1,
            )
            C_individual_list.append(C_i)
        C_individual = torch.cat(C_individual_list, dim=0)

        assert C_grouped.shape == C_individual.shape
        assert torch.allclose(C_grouped, C_individual, rtol=1e-3, atol=1e-3), (
            f"Max diff: {(C_grouped - C_individual).abs().max().item():.6f}"
        )

    @pytest.mark.parametrize("k", [4])
    def test_single_expert(self, k):
        """Single expert should match kbit_gemm_prod exactly."""
        K_dim, N = 2048, 512
        M = 8

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = (
            prepare_expert_weights(K_dim, N, k, 1)
        )

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
        expert_offsets = torch.tensor([0, M], dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.kbit_grouped_gemm(
            A, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, k, 1,
        )

        C_prod = torch.ops.bitsandbytes.kbit_gemm_prod(
            A, packed_list[0], absmax_list[0], codebook,
            K_dim, N, k, 1,
        )

        assert torch.allclose(C_grouped, C_prod, rtol=1e-3, atol=1e-3), (
            f"Max diff: {(C_grouped - C_prod).abs().max().item():.6f}"
        )

    @pytest.mark.parametrize("k", [4])
    def test_many_experts(self, k):
        """Many experts with M=1 (typical MoE inference)."""
        K_dim, N = 2048, 512
        num_experts = 64

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = (
            prepare_expert_weights(K_dim, N, k, num_experts)
        )

        A_list = []
        offsets = [0]
        for i in range(num_experts):
            A_i = torch.randn(1, K_dim, dtype=torch.float16, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + 1)

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.kbit_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, k, num_experts,
        )

        C_individual_list = []
        for i in range(num_experts):
            C_i = torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i], packed_list[i], absmax_list[i], codebook,
                K_dim, N, k, 1,
            )
            C_individual_list.append(C_i)
        C_individual = torch.cat(C_individual_list, dim=0)

        assert torch.allclose(C_grouped, C_individual, rtol=1e-3, atol=1e-3), (
            f"Max diff: {(C_grouped - C_individual).abs().max().item():.6f}"
        )

    @pytest.mark.parametrize("k", [4])
    def test_larger_N(self, k):
        """Test with N=2048 (MoE down projection shape)."""
        K_dim, N = 512, 2048
        num_experts = 8
        M_per_expert = 4

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = (
            prepare_expert_weights(K_dim, N, k, num_experts)
        )

        A_list = []
        offsets = [0]
        for i in range(num_experts):
            A_i = torch.randn(M_per_expert, K_dim, dtype=torch.float16, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + M_per_expert)

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.kbit_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, k, num_experts,
        )

        C_individual_list = []
        for i in range(num_experts):
            C_i = torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i], packed_list[i], absmax_list[i], codebook,
                K_dim, N, k, 1,
            )
            C_individual_list.append(C_i)
        C_individual = torch.cat(C_individual_list, dim=0)

        assert torch.allclose(C_grouped, C_individual, rtol=1e-3, atol=1e-3), (
            f"Max diff: {(C_grouped - C_individual).abs().max().item():.6f}"
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_bf16(self, dtype):
        """Test both fp16 and bf16."""
        k = 4
        K_dim, N = 2048, 512
        num_experts = 4
        M_per_expert = 4

        codebook = create_normal_float_codebook(k).cuda()

        packed_list = []
        absmax_list = []
        for _ in range(num_experts):
            W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
            packed_flat, absmax = torch.ops.bitsandbytes.quantize_kbit(
                W.reshape(-1), codebook, k
            )
            packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
                packed_flat, absmax.cuda(), K_dim, N, k
            )
            packed_list.append(packed_tiled)
            absmax_list.append(absmax_tiled)

        B_packed_all = torch.cat(packed_list, dim=0)
        B_absmax_all = torch.cat(absmax_list, dim=0)

        A_list = []
        offsets = [0]
        for i in range(num_experts):
            A_i = torch.randn(M_per_expert, K_dim, dtype=dtype, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + M_per_expert)

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.kbit_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, k, num_experts,
        )

        C_individual_list = []
        for i in range(num_experts):
            C_i = torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i], packed_list[i], absmax_list[i], codebook,
                K_dim, N, k, 1,
            )
            C_individual_list.append(C_i)
        C_individual = torch.cat(C_individual_list, dim=0)

        assert C_grouped.dtype == dtype
        assert torch.allclose(C_grouped, C_individual, rtol=1e-2, atol=1e-2), (
            f"Max diff: {(C_grouped - C_individual).abs().max().item():.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
