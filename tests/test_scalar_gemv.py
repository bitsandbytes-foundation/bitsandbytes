"""
Tests for kbit scalar GEMV kernel (M=1..4).

Verifies correctness by comparing scalar GEMV output against a
dequantize + matmul reference using the same flat-layout data.
"""

import pytest
from scipy.stats import norm
import torch

import bitsandbytes  # noqa: F401
from bitsandbytes import _ops  # noqa: F401

BLOCKSIZE = 32
E4M4_BIAS = 11


def decode_e4m4_absmax(raw: torch.Tensor) -> torch.Tensor:
    """Decode uint8 E4M4 absmax values to float32."""
    raw_int = raw.int()
    e = raw_int >> 4
    m = raw_int & 0xF
    # Normal: 2^(e - BIAS) * (1 + m/16)
    result = (2.0 ** (e.float() - E4M4_BIAS)) * (1.0 + m.float() / 16.0)
    result[raw == 0] = 0.0
    return result


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
    packed_flat, absmax_flat = torch.ops.bitsandbytes.quantize_kbit(W.reshape(-1), codebook, k)
    # Repacked data for MMA reference kernel
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(packed_flat, absmax_flat.cuda(), K_dim, N, k)
    return packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, W


def dequant_reference(packed_flat, absmax_flat, codebook, k, N, K_dim):
    """Dequantize using E4M4-decoded absmax.
    Matches the GEMV kernel's precision exactly."""
    num_blocks = N * (K_dim // 32)
    packed = packed_flat[: num_blocks * k].view(num_blocks, k)  # [B, k] int32
    j = torch.arange(32, device=packed.device)  # [32]

    # Extract k-bit index for each of the 32 elements per block
    indices = torch.zeros(num_blocks, 32, dtype=torch.int32, device=packed.device)
    for b in range(k):
        bits = (packed[:, b : b + 1] >> j.unsqueeze(0)) & 1  # [B, 32]
        indices += bits << b

    # Decode E4M4 absmax to float for reference computation
    absmax_decoded = decode_e4m4_absmax(absmax_flat[:num_blocks])

    # Codebook lookup + absmax scale
    W_flat = codebook[indices.long()] * absmax_decoded.unsqueeze(1)
    return W_flat.reshape(N, K_dim)


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
        packed_flat, absmax_flat, _, _, codebook, _W = prepare_weights(K_dim, N, k)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_scalar = torch.ops.bitsandbytes.kbit_scalar_gemv(
            A,
            packed_flat,
            absmax_flat,
            codebook,
            K_dim,
            N,
            k,
        )
        W_deq = dequant_reference(packed_flat, absmax_flat, codebook, k, N, K_dim)
        C_ref = (A.float() @ W_deq.T).to(A.dtype)

        assert C_scalar.shape == C_ref.shape
        assert_close(C_scalar, C_ref, max_rel_err=0.10, label=f"k={k}, M={M}: ")

    @pytest.mark.parametrize(
        "K_dim,N",
        [
            (2048, 5120),
            (5120, 2048),
            (2048, 4096),
            (512, 2048),
        ],
    )
    def test_various_shapes(self, K_dim, N):
        """Test with shapes matching real model projections."""
        k = 4
        M = 1
        packed_flat, absmax_flat, _, _, codebook, _W = prepare_weights(K_dim, N, k)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_scalar = torch.ops.bitsandbytes.kbit_scalar_gemv(
            A,
            packed_flat,
            absmax_flat,
            codebook,
            K_dim,
            N,
            k,
        )
        W_deq = dequant_reference(packed_flat, absmax_flat, codebook, k, N, K_dim)
        C_ref = (A.float() @ W_deq.T).to(A.dtype)

        assert_close(C_scalar, C_ref, max_rel_err=0.10, label=f"Shape ({K_dim},{N}): ")

    @pytest.mark.parametrize("M", [1, 2, 3, 4])
    def test_large_shape(self, M):
        """Test large shape with all M values."""
        k = 4
        K_dim, N = 2048, 5120
        packed_flat, absmax_flat, _, _, codebook, _W = prepare_weights(K_dim, N, k)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_scalar = torch.ops.bitsandbytes.kbit_scalar_gemv(
            A,
            packed_flat,
            absmax_flat,
            codebook,
            K_dim,
            N,
            k,
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
        packed_flat, absmax_flat, _, _, codebook, _W = prepare_weights(K_dim, N, k)

        A = torch.randn(M, K_dim, dtype=dtype, device="cuda")

        C_scalar = torch.ops.bitsandbytes.kbit_scalar_gemv(
            A,
            packed_flat,
            absmax_flat,
            codebook,
            K_dim,
            N,
            k,
        )
        W_deq = dequant_reference(packed_flat, absmax_flat, codebook, k, N, K_dim)
        C_ref = (A.float() @ W_deq.T).to(dtype)

        assert C_scalar.dtype == dtype
        tol = 0.25 if dtype == torch.bfloat16 else 0.10
        assert_close(C_scalar, C_ref, max_rel_err=tol, label=f"dtype={dtype}: ")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
