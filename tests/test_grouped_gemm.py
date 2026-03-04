"""
Tests for kbit grouped expert GEMM kernel.

Verifies correctness by comparing grouped GEMM output against individual
kbit_gemm_prod calls for each expert.
"""

import pytest
from scipy.stats import norm
import torch

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
        packed_flat, absmax = torch.ops.bitsandbytes.quantize_kbit(W.reshape(-1), codebook, k)
        packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(packed_flat, absmax.cuda(), K_dim, N, k)
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

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = prepare_expert_weights(
            K_dim, N, k, num_experts
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
            A_concat,
            B_packed_all,
            B_absmax_all,
            codebook,
            expert_offsets,
            K_dim,
            N,
            k,
            num_experts,
        )

        # Individual GEMM for each expert
        C_individual_list = []
        for i in range(num_experts):
            C_i = torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i],
                packed_list[i],
                absmax_list[i],
                codebook,
                K_dim,
                N,
                k,
                1,
            )
            C_individual_list.append(C_i)
        C_individual = torch.cat(C_individual_list, dim=0)

        # Compare
        assert C_grouped.shape == C_individual.shape, f"Shape mismatch: {C_grouped.shape} vs {C_individual.shape}"
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

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = prepare_expert_weights(
            K_dim, N, k, num_experts
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
            A_concat,
            B_packed_all,
            B_absmax_all,
            codebook,
            expert_offsets,
            K_dim,
            N,
            k,
            num_experts,
        )

        C_individual_list = []
        for i in range(num_experts):
            C_i = torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i],
                packed_list[i],
                absmax_list[i],
                codebook,
                K_dim,
                N,
                k,
                1,
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

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = prepare_expert_weights(K_dim, N, k, 1)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
        expert_offsets = torch.tensor([0, M], dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.kbit_grouped_gemm(
            A,
            B_packed_all,
            B_absmax_all,
            codebook,
            expert_offsets,
            K_dim,
            N,
            k,
            1,
        )

        C_prod = torch.ops.bitsandbytes.kbit_gemm_prod(
            A,
            packed_list[0],
            absmax_list[0],
            codebook,
            K_dim,
            N,
            k,
            1,
        )

        assert torch.allclose(C_grouped, C_prod, rtol=1e-3, atol=1e-3), (
            f"Max diff: {(C_grouped - C_prod).abs().max().item():.6f}"
        )

    @pytest.mark.parametrize("k", [4])
    def test_many_experts(self, k):
        """Many experts with M=1 (typical MoE inference)."""
        K_dim, N = 2048, 512
        num_experts = 64

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = prepare_expert_weights(
            K_dim, N, k, num_experts
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
            A_concat,
            B_packed_all,
            B_absmax_all,
            codebook,
            expert_offsets,
            K_dim,
            N,
            k,
            num_experts,
        )

        C_individual_list = []
        for i in range(num_experts):
            C_i = torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i],
                packed_list[i],
                absmax_list[i],
                codebook,
                K_dim,
                N,
                k,
                1,
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

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = prepare_expert_weights(
            K_dim, N, k, num_experts
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
            A_concat,
            B_packed_all,
            B_absmax_all,
            codebook,
            expert_offsets,
            K_dim,
            N,
            k,
            num_experts,
        )

        C_individual_list = []
        for i in range(num_experts):
            C_i = torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i],
                packed_list[i],
                absmax_list[i],
                codebook,
                K_dim,
                N,
                k,
                1,
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
            packed_flat, absmax = torch.ops.bitsandbytes.quantize_kbit(W.reshape(-1), codebook, k)
            packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(packed_flat, absmax.cuda(), K_dim, N, k)
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
            A_concat,
            B_packed_all,
            B_absmax_all,
            codebook,
            expert_offsets,
            K_dim,
            N,
            k,
            num_experts,
        )

        C_individual_list = []
        for i in range(num_experts):
            C_i = torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i],
                packed_list[i],
                absmax_list[i],
                codebook,
                K_dim,
                N,
                k,
                1,
            )
            C_individual_list.append(C_i)
        C_individual = torch.cat(C_individual_list, dim=0)

        assert C_grouped.dtype == dtype
        assert torch.allclose(C_grouped, C_individual, rtol=1e-2, atol=1e-2), (
            f"Max diff: {(C_grouped - C_individual).abs().max().item():.6f}"
        )


# ===================================================================
# VQ Grouped GEMM Tests
# ===================================================================

def prepare_vq_expert_weights(K_dim, N, p, num_experts):
    """Quantize and repack VQ weights for multiple experts.
    Returns (B_packed_all, B_absmax_all, codebook, packed_list, absmax_list, W_deq_list).
    W_deq_list contains dequantized weight matrices for reference computation.
    """
    from bitsandbytes.functional import create_vq_codebook, quantize_vq, repack_vq, dequantize_vq

    codebook = create_vq_codebook(p, device="cuda")

    packed_list = []
    absmax_list = []
    W_deq_list = []

    for _ in range(num_experts):
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        packed_flat, absmax_flat, _ = quantize_vq(W, p=p, codebook=codebook)
        W_deq = dequantize_vq(packed_flat, absmax_flat, codebook, p=p, n=N * K_dim).view(N, K_dim)
        packed_tiled, absmax_tiled = repack_vq(packed_flat, absmax_flat, K_dim, N, p=p)
        packed_list.append(packed_tiled)
        absmax_list.append(absmax_tiled)
        W_deq_list.append(W_deq)

    B_packed_all = torch.cat(packed_list, dim=0)
    B_absmax_all = torch.cat(absmax_list, dim=0)

    return B_packed_all, B_absmax_all, codebook, packed_list, absmax_list, W_deq_list


def vq_matmul_ref(A_list, W_deq_list):
    """Compute reference output via dequantized weights + matmul per expert."""
    C_ref_list = []
    for A_e, W_deq in zip(A_list, W_deq_list):
        C_ref_list.append((A_e.float() @ W_deq.float().T).half())
    return torch.cat(C_ref_list, dim=0)


class TestVQGroupedGemm:
    """Test VQ grouped expert GEMM against dequant+matmul reference."""

    def test_basic_correctness(self):
        """All experts same M=4, compare grouped output against dequant+matmul."""
        K_dim, N = 2048, 1536
        num_experts = 4
        M_per_expert = 4
        p = 2

        B_packed_all, B_absmax_all, codebook, packed_list, absmax_list, W_deq_list = prepare_vq_expert_weights(
            K_dim, N, p, num_experts
        )

        A_list = []
        offsets = [0]
        for i in range(num_experts):
            A_i = torch.randn(M_per_expert, K_dim, dtype=torch.float16, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + M_per_expert)

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.vq_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, p, num_experts, M_per_expert,
        )

        C_ref = vq_matmul_ref(A_list, W_deq_list)

        assert C_grouped.shape == C_ref.shape, f"Shape mismatch: {C_grouped.shape} vs {C_ref.shape}"
        diff = (C_grouped.float() - C_ref.float()).abs()
        rel_err = (diff / C_ref.float().abs().clamp(min=1.0)).max().item()
        assert rel_err < 0.01, (
            f"Max rel err: {rel_err:.6f}, Max abs diff: {diff.max().item():.6f}"
        )

    def test_varying_tokens(self):
        """Experts with 0, 1, 2, 4 tokens."""
        K_dim, N = 2048, 1536
        num_experts = 4
        M_values = [0, 1, 2, 4]
        max_M = max(M_values)
        p = 2

        B_packed_all, B_absmax_all, codebook, packed_list, absmax_list, W_deq_list = prepare_vq_expert_weights(
            K_dim, N, p, num_experts
        )

        A_list = []  # only non-zero experts
        A_all = []   # for reference, indexed by expert
        offsets = [0]
        for i in range(num_experts):
            if M_values[i] > 0:
                A_i = torch.randn(M_values[i], K_dim, dtype=torch.float16, device="cuda")
                A_all.append(A_i)
            else:
                A_all.append(None)
            offsets.append(offsets[-1] + M_values[i])

        A_concat_parts = [a for a in A_all if a is not None]
        A_concat = torch.cat(A_concat_parts, dim=0) if A_concat_parts else torch.empty(0, K_dim, dtype=torch.float16, device="cuda")
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.vq_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, p, num_experts, max_M,
        )

        # Reference via dequant+matmul for non-zero experts
        C_ref_list = []
        for i in range(num_experts):
            if M_values[i] == 0:
                continue
            C_ref_list.append((A_all[i].float() @ W_deq_list[i].float().T).half())
        C_ref = torch.cat(C_ref_list, dim=0)

        assert C_grouped.shape == C_ref.shape
        diff = (C_grouped.float() - C_ref.float()).abs()
        rel_err = (diff / C_ref.float().abs().clamp(min=1.0)).max().item()
        assert rel_err < 0.01, f"Max rel err: {rel_err:.6f}"

    def test_qwen3_shapes(self):
        """K=2048, N=512, 8 experts (Qwen3 MoE top-8 subset)."""
        K_dim, N = 2048, 512
        num_experts = 8
        M_per_expert = 2
        p = 2

        B_packed_all, B_absmax_all, codebook, packed_list, absmax_list, W_deq_list = prepare_vq_expert_weights(
            K_dim, N, p, num_experts
        )

        A_list = []
        offsets = [0]
        for i in range(num_experts):
            A_i = torch.randn(M_per_expert, K_dim, dtype=torch.float16, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + M_per_expert)

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.vq_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, p, num_experts, M_per_expert,
        )

        C_ref = vq_matmul_ref(A_list, W_deq_list)

        assert C_grouped.shape == C_ref.shape
        diff = (C_grouped.float() - C_ref.float()).abs()
        rel_err = (diff / C_ref.float().abs().clamp(min=1.0)).max().item()
        assert rel_err < 0.01, f"Max rel err: {rel_err:.6f}"

    def test_fixed_padding(self):
        """Fixed padding: pad all experts to pad_M=8, only fill 1-2 rows."""
        from bitsandbytes.functional import vq_moe_fixed_pad

        K_dim, N = 2048, 1536
        num_experts = 4
        pad_M = 8
        p = 2

        B_packed_all, B_absmax_all, codebook, packed_list, absmax_list, W_deq_list = prepare_vq_expert_weights(
            K_dim, N, p, num_experts
        )

        # Create tokens with 1-2 per expert
        total_tokens = 6
        tokens = torch.randn(total_tokens, K_dim, dtype=torch.float16, device="cuda")
        expert_indices = torch.tensor([0, 0, 1, 2, 2, 3], dtype=torch.int64, device="cuda")

        A_padded, offsets_fixed = vq_moe_fixed_pad(tokens, expert_indices, pad_M, K_dim, num_experts)

        C_padded = torch.ops.bitsandbytes.vq_grouped_gemm(
            A_padded, B_packed_all, B_absmax_all, codebook,
            offsets_fixed, K_dim, N, p, num_experts, pad_M,
        )

        # Verify real tokens match dequant+matmul reference
        # Expert 0: tokens 0,1; Expert 1: token 2; Expert 2: tokens 3,4; Expert 3: token 5
        M_per_expert = [2, 1, 2, 1]
        token_idx = 0
        for e in range(num_experts):
            me = M_per_expert[e]
            if me == 0:
                continue
            A_e = tokens[token_idx : token_idx + me]
            C_ref = (A_e.float() @ W_deq_list[e].float().T).half()
            C_actual = C_padded[e * pad_M : e * pad_M + me]
            diff = (C_actual.float() - C_ref.float()).abs()
            rel_err = (diff / C_ref.float().abs().clamp(min=1.0)).max().item()
            assert rel_err < 0.01, f"Expert {e}: Max rel err: {rel_err:.6f}"
            token_idx += me

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_dtype(self, dtype):
        """Test both fp16 and bf16 activations."""
        K_dim, N = 2048, 512
        num_experts = 4
        M_per_expert = 4
        p = 2

        B_packed_all, B_absmax_all, codebook, packed_list, absmax_list, W_deq_list = prepare_vq_expert_weights(
            K_dim, N, p, num_experts
        )

        A_list = []
        offsets = [0]
        for i in range(num_experts):
            A_i = torch.randn(M_per_expert, K_dim, dtype=dtype, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + M_per_expert)

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.vq_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, p, num_experts, M_per_expert,
        )

        # Reference: dequant+matmul in float32 for dtype-agnostic comparison
        C_ref_list = []
        for i in range(num_experts):
            C_ref_list.append((A_list[i].float() @ W_deq_list[i].float().T))
        C_ref = torch.cat(C_ref_list, dim=0)

        assert C_grouped.dtype == dtype
        diff = (C_grouped.float() - C_ref).abs()
        scale = C_ref.abs().clamp(min=1.0)
        rel_err = (diff / scale).max().item()
        # bf16 MMA accumulation has ~0.5% relative error vs fp32 reference;
        # with K=2048 the absolute error can be larger due to accumulation
        max_rel = 0.5 if dtype == torch.bfloat16 else 0.01
        assert rel_err < max_rel, f"Max rel err: {rel_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
