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


# ===========================================================================
# VQ Codebook Scalar GEMV Tests
# ===========================================================================


def prepare_vq_weights(K_dim, N, p, dtype=torch.float16):
    """Quantize a weight matrix with VQ codebook. Returns flat and tiled data."""
    from bitsandbytes.functional import create_vq_codebook, quantize_vq, repack_vq

    codebook = create_vq_codebook(p, device="cuda")
    W = torch.randn(N, K_dim, dtype=dtype, device="cuda")

    packed_flat, absmax_flat, codebook = quantize_vq(W, p=p, codebook=codebook)
    packed_tiled, absmax_tiled = repack_vq(packed_flat, absmax_flat, K_dim, N, p=p)

    return packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, W


def vq_dequant_reference(packed_flat, absmax_flat, codebook, p, N, K_dim):
    """Dequantize VQ packed data using the Python-level dequantize_vq kernel.
    Returns [N, K_dim] weight matrix matching GEMV kernel precision."""
    from bitsandbytes.functional import dequantize_vq

    n_total = N * K_dim
    W_flat = dequantize_vq(packed_flat, absmax_flat, codebook, p=p, n=n_total)
    return W_flat.reshape(N, K_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVQQuantDequantRoundtrip:
    """Test VQ quantize-dequantize roundtrip quality."""

    @pytest.mark.parametrize("p", [2, 4])
    def test_roundtrip_mse(self, p):
        """VQ quantize -> dequantize roundtrip has reasonable MSE."""
        from bitsandbytes.functional import create_vq_codebook, quantize_vq, dequantize_vq

        N, K_dim = 512, 2048
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        codebook = create_vq_codebook(p, device="cuda")

        packed, absmax, _ = quantize_vq(W, p=p, codebook=codebook)
        n_total = N * K_dim
        W_deq = dequantize_vq(packed, absmax, codebook, p=p, n=n_total, dtype=torch.float16)
        W_deq = W_deq.reshape(N, K_dim)

        mse = ((W.float() - W_deq.float()) ** 2).mean().item()
        orig_var = (W.float() ** 2).mean().item()
        nmse = mse / orig_var

        # p=2 (4 bits/wt) should have NMSE < 0.1, p=4 (2 bits/wt) < 0.2
        threshold = 0.10 if p == 2 else 0.20
        assert nmse < threshold, f"p={p}: NMSE {nmse:.4f} exceeds threshold {threshold}"

    @pytest.mark.parametrize("p", [2, 4])
    def test_roundtrip_shapes(self, p):
        """Roundtrip preserves tensor shape."""
        from bitsandbytes.functional import create_vq_codebook, quantize_vq, dequantize_vq

        N, K_dim = 256, 512
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        codebook = create_vq_codebook(p, device="cuda")

        packed, absmax, _ = quantize_vq(W, p=p, codebook=codebook)
        n_total = N * K_dim
        W_deq = dequantize_vq(packed, absmax, codebook, p=p, n=n_total, dtype=torch.float16)

        assert W_deq.numel() == n_total, f"Expected {n_total} elements, got {W_deq.numel()}"

    @pytest.mark.parametrize("p", [2, 4])
    def test_flat_vs_tiled_dequant(self, p):
        """Flat dequant and tiled dequant produce same results."""
        from bitsandbytes.functional import create_vq_codebook, quantize_vq, dequantize_vq, repack_vq

        N, K_dim = 256, 512
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        codebook = create_vq_codebook(p, device="cuda")

        packed_flat, absmax_flat, _ = quantize_vq(W, p=p, codebook=codebook)
        packed_tiled, absmax_tiled = repack_vq(packed_flat, absmax_flat, K_dim, N, p=p)

        # Flat dequant
        n_total = N * K_dim
        W_flat = dequantize_vq(packed_flat, absmax_flat, codebook, p=p, n=n_total)

        # Tiled dequant
        W_tiled = torch.ops.bitsandbytes.dequantize_vq_tiled(
            packed_tiled, codebook, absmax_tiled, p, K_dim, N, torch.float16
        )

        assert torch.equal(W_flat, W_tiled), (
            f"p={p}: flat vs tiled dequant mismatch. "
            f"Max diff: {(W_flat.float() - W_tiled.float()).abs().max().item()}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVQScalarGemv:
    """Test VQ scalar GEMV against dequantize + matmul reference."""

    @pytest.mark.parametrize("M", [1, 2, 3, 4])
    @pytest.mark.parametrize("p", [2, 4])
    def test_basic_correctness(self, M, p):
        """VQ scalar GEMV matches dequant + matmul reference for all M and p."""
        K_dim, N = 2048, 512
        packed_flat, absmax_flat, _, _, codebook, _W = prepare_vq_weights(K_dim, N, p)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_scalar = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p,
        )
        W_deq = vq_dequant_reference(packed_flat, absmax_flat, codebook, p, N, K_dim)
        C_ref = (A.float() @ W_deq.float().T).to(A.dtype)

        assert C_scalar.shape == C_ref.shape
        assert_close(C_scalar, C_ref, max_rel_err=0.10, label=f"p={p}, M={M}: ")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("p", [2, 4])
    def test_dtype(self, dtype, p):
        """VQ scalar GEMV works with both fp16 and bf16."""
        K_dim, N = 2048, 512
        M = 2
        packed_flat, absmax_flat, _, _, codebook, _W = prepare_vq_weights(K_dim, N, p, dtype=dtype)

        A = torch.randn(M, K_dim, dtype=dtype, device="cuda")

        C_scalar = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p,
        )
        W_deq = vq_dequant_reference(packed_flat, absmax_flat, codebook, p, N, K_dim)
        C_ref = (A.float() @ W_deq.float().T).to(dtype)

        assert C_scalar.dtype == dtype
        tol = 0.25 if dtype == torch.bfloat16 else 0.10
        assert_close(C_scalar, C_ref, max_rel_err=tol, label=f"dtype={dtype}, p={p}: ")

    @pytest.mark.parametrize(
        "K_dim,N",
        [
            (2048, 5120),
            (5120, 2048),
            (2048, 4096),
            (512, 2048),
        ],
    )
    @pytest.mark.parametrize("p", [2, 4])
    def test_various_shapes(self, K_dim, N, p):
        """VQ scalar GEMV works for shapes matching real model projections."""
        M = 1
        packed_flat, absmax_flat, _, _, codebook, _W = prepare_vq_weights(K_dim, N, p)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_scalar = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p,
        )
        W_deq = vq_dequant_reference(packed_flat, absmax_flat, codebook, p, N, K_dim)
        C_ref = (A.float() @ W_deq.float().T).to(A.dtype)

        assert_close(C_scalar, C_ref, max_rel_err=0.10, label=f"p={p}, ({K_dim},{N}): ")

    @pytest.mark.parametrize("K_dim", [32, 64, 2048, 5120])
    @pytest.mark.parametrize("p", [2, 4])
    def test_edge_k_dimensions(self, K_dim, p):
        """VQ scalar GEMV works for edge K dimensions including minimum."""
        N = 128  # Minimum tile size
        M = 1
        packed_flat, absmax_flat, _, _, codebook, _W = prepare_vq_weights(K_dim, N, p)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_scalar = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p,
        )
        W_deq = vq_dequant_reference(packed_flat, absmax_flat, codebook, p, N, K_dim)
        C_ref = (A.float() @ W_deq.float().T).to(A.dtype)

        assert_close(C_scalar, C_ref, max_rel_err=0.10, label=f"p={p}, K={K_dim}: ")

    @pytest.mark.parametrize("M", [1, 2, 3, 4])
    @pytest.mark.parametrize("p", [2, 4])
    def test_flat_vs_tiled_gemv(self, M, p):
        """Flat-layout and tiled-layout GEMV produce identical results."""
        K_dim, N = 2048, 512
        packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, _W = prepare_vq_weights(K_dim, N, p)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_flat = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p,
        )
        C_tiled = torch.ops.bitsandbytes.vq_scalar_gemv_tiled(
            A, packed_tiled, absmax_tiled, codebook, K_dim, N, p,
        )

        assert torch.equal(C_flat, C_tiled), (
            f"p={p}, M={M}: flat vs tiled GEMV mismatch. "
            f"Max diff: {(C_flat.float() - C_tiled.float()).abs().max().item()}"
        )

    @pytest.mark.parametrize("M", [1, 2, 3, 4])
    @pytest.mark.parametrize("p", [2, 4])
    def test_large_shape(self, M, p):
        """VQ scalar GEMV correctness for a large shape (2048x5120)."""
        K_dim, N = 2048, 5120
        packed_flat, absmax_flat, _, _, codebook, _W = prepare_vq_weights(K_dim, N, p)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C_scalar = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p,
        )
        W_deq = vq_dequant_reference(packed_flat, absmax_flat, codebook, p, N, K_dim)
        C_ref = (A.float() @ W_deq.float().T).to(A.dtype)

        assert_close(C_scalar, C_ref, max_rel_err=0.10, label=f"p={p}, M={M}, large: ")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
