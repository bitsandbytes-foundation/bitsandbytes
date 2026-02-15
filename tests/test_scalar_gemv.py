"""
Tests for kbit scalar GEMV kernel (M=1..4).

Verifies correctness by comparing scalar GEMV output against a
dequantize + matmul reference using the same flat-layout data.
The grouped GEMV tests still compare against individual kbit_gemm_prod calls.
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


def prepare_weights(K_dim, N, k):
    """Quantize a single weight matrix. Returns flat data for scalar GEMV
    and repacked data for MMA/grouped reference kernels."""
    codebook = create_normal_float_codebook(k).cuda()
    W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
    packed_flat, absmax_flat = torch.ops.bitsandbytes.quantize_kbit(
        W.reshape(-1), codebook, k
    )
    # Repacked data for MMA reference kernel
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
        packed_flat, absmax_flat.cuda(), K_dim, N, k
    )
    return packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, W


def dequant_reference(packed_flat, absmax_flat, codebook, k, N, K_dim):
    """Dequantize using float32 absmax directly (no E4M4 encoding).
    Matches the GEMV kernel's precision exactly."""
    num_blocks = N * (K_dim // 32)
    packed = packed_flat[:num_blocks * k].view(num_blocks, k)  # [B, k] int32
    j = torch.arange(32, device=packed.device)  # [32]

    # Extract k-bit index for each of the 32 elements per block
    indices = torch.zeros(num_blocks, 32, dtype=torch.int32, device=packed.device)
    for b in range(k):
        bits = (packed[:, b:b+1] >> j.unsqueeze(0)) & 1  # [B, 32]
        indices += bits << b

    # Codebook lookup + absmax scale
    W_flat = codebook[indices.long()] * absmax_flat[:num_blocks].unsqueeze(1)
    return W_flat.reshape(N, K_dim)


def prepare_expert_weights(K_dim, N, k, num_experts):
    """Quantize and repack weights for multiple experts."""
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


def assert_close(actual, expected, max_rel_err=0.05, label=""):
    """Assert that actual and expected are close using relative error.

    The GEMV kernel and torch matmul accumulate in different FMA orders,
    producing small numerical differences (~1-3% for fp16, ~5-15% for bf16).
    We use relative error with a floor of 1.0 to avoid division-by-near-zero.
    """
    diff = (actual.float() - expected.float()).abs()
    scale = expected.float().abs().clamp(min=1.0)
    rel_err = (diff / scale).max().item()
    assert rel_err < max_rel_err, (
        f"{label}Max rel err: {rel_err:.6f}, "
        f"Max abs diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}"
    )


class TestScalarGemv:
    """Test scalar GEMV against dequantize + matmul reference (same float32 absmax)."""

    @pytest.mark.parametrize("M", [1, 2, 3, 4])
    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_basic_correctness(self, M, k):
        """Compare scalar GEMV against dequant + matmul reference."""
        K_dim, N = 2048, 512
        packed_flat, absmax_flat, _, _, codebook, W = prepare_weights(K_dim, N, k)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_scalar = torch.ops.bitsandbytes.kbit_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, k,
        )
        W_deq = dequant_reference(packed_flat, absmax_flat, codebook, k, N, K_dim)
        C_ref = (A.float() @ W_deq.T).to(A.dtype)

        assert C_scalar.shape == C_ref.shape
        assert_close(C_scalar, C_ref, max_rel_err=0.10, label=f"k={k}, M={M}: ")

    @pytest.mark.parametrize("K_dim,N", [
        (2048, 5120),
        (5120, 2048),
        (2048, 4096),
        (512, 2048),
    ])
    def test_various_shapes(self, K_dim, N):
        """Test with shapes matching real model projections."""
        k = 4
        M = 1
        packed_flat, absmax_flat, _, _, codebook, W = prepare_weights(K_dim, N, k)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_scalar = torch.ops.bitsandbytes.kbit_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, k,
        )
        W_deq = dequant_reference(packed_flat, absmax_flat, codebook, k, N, K_dim)
        C_ref = (A.float() @ W_deq.T).to(A.dtype)

        assert_close(C_scalar, C_ref, max_rel_err=0.10, label=f"Shape ({K_dim},{N}): ")

    @pytest.mark.parametrize("M", [1, 2, 3, 4])
    def test_large_shape(self, M):
        """Test large shape with all M values."""
        k = 4
        K_dim, N = 2048, 5120
        packed_flat, absmax_flat, _, _, codebook, W = prepare_weights(K_dim, N, k)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_scalar = torch.ops.bitsandbytes.kbit_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, k,
        )
        W_deq = dequant_reference(packed_flat, absmax_flat, codebook, k, N, K_dim)
        C_ref = (A.float() @ W_deq.T).to(A.dtype)

        assert_close(C_scalar, C_ref, max_rel_err=0.10, label=f"M={M}, large: ")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_dtype(self, dtype):
        """Test both fp16 and bf16."""
        k = 4
        K_dim, N = 2048, 512
        M = 2
        packed_flat, absmax_flat, _, _, codebook, W = prepare_weights(K_dim, N, k)

        A = torch.randn(M, K_dim, dtype=dtype, device="cuda")

        C_scalar = torch.ops.bitsandbytes.kbit_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, k,
        )
        W_deq = dequant_reference(packed_flat, absmax_flat, codebook, k, N, K_dim)
        C_ref = (A.float() @ W_deq.T).to(dtype)

        assert C_scalar.dtype == dtype
        tol = 0.25 if dtype == torch.bfloat16 else 0.10
        assert_close(C_scalar, C_ref, max_rel_err=tol, label=f"dtype={dtype}: ")


class TestGroupedScalarGemv:
    """Test grouped scalar GEMV against individual kbit_gemm_prod calls."""

    @pytest.mark.parametrize("k", [4])
    def test_basic_grouped(self, k):
        """Basic grouped test: M=1 per expert."""
        K_dim, N = 2048, 512
        num_experts = 8

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

        C_grouped = torch.ops.bitsandbytes.kbit_grouped_scalar_gemv(
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
        assert_close(C_grouped, C_individual, label="grouped basic: ")

    @pytest.mark.parametrize("k", [4])
    def test_variable_M(self, k):
        """Experts with different M values (all <=4)."""
        K_dim, N = 2048, 512
        num_experts = 8
        M_values = [1, 2, 3, 4, 3, 1, 2, 1]

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

        C_grouped = torch.ops.bitsandbytes.kbit_grouped_scalar_gemv(
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
        assert_close(C_grouped, C_individual, label="grouped variable-M: ")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_grouped_dtype(self, dtype):
        """Test grouped scalar GEMV with both dtypes."""
        k = 4
        K_dim, N = 2048, 512
        num_experts = 4

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
            A_i = torch.randn(2, K_dim, dtype=dtype, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + 2)

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        C_grouped = torch.ops.bitsandbytes.kbit_grouped_scalar_gemv(
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
        tol = 0.25 if dtype == torch.bfloat16 else 0.05
        assert_close(C_grouped, C_individual, max_rel_err=tol, label=f"grouped dtype={dtype}: ")

    @pytest.mark.parametrize("k", [4])
    def test_larger_N(self, k):
        """Test with N=2048."""
        K_dim, N = 512, 2048
        num_experts = 8

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

        C_grouped = torch.ops.bitsandbytes.kbit_grouped_scalar_gemv(
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

        assert_close(C_grouped, C_individual, label="grouped larger-N: ")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
