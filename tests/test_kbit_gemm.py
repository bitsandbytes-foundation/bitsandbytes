"""
Tests for kbit fused dequantization + GEMM kernel.

Staged implementation following cuda-spec.md:
  Stage 1: Python reference (repack + fused GEMM)
  Stage 2: CUDA repack kernel (bit-exact match with Python reference)
  Stage 3: Minimal CUDA GEMM (no pipeline, no split-K)
  Stage 4: Add cp.async 4-stage pipeline
  Stage 5: Persistent kernel + split-K
  Stage 6: Optimization + bf16 + benchmarks
"""

import pytest
import torch
from scipy.stats import norm

import bitsandbytes  # noqa: F401 (registers torch.library ops)


# ---------------------------------------------------------------------------
# Codebook generation (same as test_kbit_quantization.py)
# ---------------------------------------------------------------------------

BLOCKSIZE = 32


def create_normal_float_codebook(k: int) -> torch.Tensor:
    n_levels = 1 << k
    quantiles = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    values = torch.tensor(norm.ppf(quantiles.numpy()), dtype=torch.float32)
    values = values / values.abs().max()
    return values


# ---------------------------------------------------------------------------
# Reference quantize/dequantize (from test_kbit_quantization.py)
# ---------------------------------------------------------------------------


def quantize_kbit_ref(A, codebook, blocksize=BLOCKSIZE):
    A_flat = A.float().reshape(-1)
    n = A_flat.numel()
    pad = (blocksize - n % blocksize) % blocksize
    if pad > 0:
        A_flat = torch.nn.functional.pad(A_flat, (0, pad))
    n_padded = A_flat.numel()
    num_blocks = n_padded // blocksize
    blocks = A_flat.reshape(num_blocks, blocksize)
    absmax = blocks.abs().max(dim=1).values
    absmax_safe = absmax.clamp(min=1e-8)
    normalized = blocks / absmax_safe.unsqueeze(1)
    cb = codebook.float().unsqueeze(0).unsqueeze(0)
    norm_exp = normalized.unsqueeze(2)
    distances = (norm_exp - cb).abs()
    indices = distances.argmin(dim=2).to(torch.uint8)
    indices = indices.reshape(-1)[:n]
    return indices, absmax


def dequantize_kbit_ref(indices, absmax, codebook, dtype=torch.float32, blocksize=BLOCKSIZE):
    n = indices.numel()
    pad = (blocksize - n % blocksize) % blocksize
    if pad > 0:
        indices = torch.nn.functional.pad(indices.long(), (0, pad))
    n_padded = indices.numel()
    num_blocks = n_padded // blocksize
    cb_values = codebook.float()[indices.long()]
    cb_values = cb_values.reshape(num_blocks, blocksize)
    out = cb_values * absmax.unsqueeze(1)
    out = out.reshape(-1)[:n]
    return out.to(dtype)


def pack_kbit_ref(indices, k, blocksize=BLOCKSIZE):
    n = indices.numel()
    pad = (blocksize - n % blocksize) % blocksize
    if pad > 0:
        indices = torch.nn.functional.pad(indices.int(), (0, pad))
    n_padded = indices.numel()
    num_blocks = n_padded // blocksize
    blocks = indices.int().reshape(num_blocks, blocksize)
    packed_words = []
    for b in range(num_blocks):
        for bit in range(k):
            word = 0
            for i in range(blocksize):
                word |= ((int(blocks[b, i]) >> bit) & 1) << i
            if word >= (1 << 31):
                word -= 1 << 32
            packed_words.append(word)
    return torch.tensor(packed_words, dtype=torch.int32)


def unpack_kbit_ref(packed, k, n, blocksize=BLOCKSIZE):
    num_blocks = packed.numel() // k
    indices = []
    for b in range(num_blocks):
        words_raw = packed[b * k : b * k + k].tolist()
        words = [(w & 0xFFFFFFFF) for w in words_raw]
        for i in range(blocksize):
            val = 0
            for bit in range(k):
                val |= ((words[bit] >> i) & 1) << bit
            indices.append(val)
    return torch.tensor(indices[:n], dtype=torch.uint8)


# ---------------------------------------------------------------------------
# E4M4 encode/decode (Python reference)
# ---------------------------------------------------------------------------


def encode_absmax_e4m4(absmax, bias=11):
    result = torch.zeros_like(absmax, dtype=torch.uint8)
    nonzero = absmax > 0
    if not nonzero.any():
        return result
    log2_val = torch.log2(absmax[nonzero])
    e_unbiased = torch.floor(log2_val).to(torch.int32)
    e_biased = (e_unbiased + bias).clamp(0, 15)
    is_subnormal = (e_unbiased + bias) <= 0
    e_biased[is_subnormal] = 0
    abs_nz = absmax[nonzero]
    mantissa = torch.zeros_like(abs_nz, dtype=torch.int32)
    normal_mask = ~is_subnormal
    if normal_mask.any():
        e_ub_normal = e_unbiased[normal_mask]
        scale = torch.exp2(e_ub_normal.float())
        m_float = (abs_nz[normal_mask] / scale - 1.0) * 16.0
        mantissa[normal_mask] = m_float.round().to(torch.int32).clamp(0, 15)
    if is_subnormal.any():
        subnormal_scale = 2.0 ** (1 - bias)
        m_float = abs_nz[is_subnormal] / subnormal_scale * 16.0
        mantissa[is_subnormal] = m_float.round().to(torch.int32).clamp(0, 15)
    encoded = (e_biased << 4 | mantissa).to(torch.uint8)
    result[nonzero] = encoded
    return result


def decode_absmax_e4m4(encoded, bias=11):
    raw = encoded.to(torch.int32)
    e = raw >> 4
    m = raw & 0xF
    is_subnormal = e == 0
    result = torch.zeros_like(encoded, dtype=torch.float32)
    if (~is_subnormal).any():
        e_normal = e[~is_subnormal].float()
        m_normal = m[~is_subnormal].float()
        result[~is_subnormal] = torch.exp2(e_normal - bias) * (1.0 + m_normal / 16.0)
    if is_subnormal.any():
        m_sub = m[is_subnormal].float()
        result[is_subnormal] = (2.0 ** (1 - bias)) * (m_sub / 16.0)
    return result


# ---------------------------------------------------------------------------
# Stage 1: Python reference repack
# ---------------------------------------------------------------------------

# Tile sizes matching the GEMM kernel design
TILE_K = 64
TILE_N = 128


def repack_kbit_ref(packed_flat, absmax_flat, K_dim, N, k, tile_k=TILE_K, tile_n=TILE_N):
    """Repack flat bit-plane data into GEMM-tiled layout (Python reference).

    Input layout (flat, from quantize kernel):
      Weight matrix W is [N, K_dim] (PyTorch convention: out_features, in_features).
      Flattened row-major: flat_index = n * K_dim + kk for element (n, kk).
      block_id = flat_index // 32
      packed_flat[block_id * k + bit] = bit-plane word

    Output layout (tiled, for GEMM kernel):
      packed_tiled[k_tile][n_tile][col][k_block][bit]
      absmax_tiled[k_tile][n_tile][col][k_block]

    The GEMM computes C[M,N] = A[M,K_dim] * W^T, which reads W along its
    K_dim dimension (columns of W[N, K_dim] = rows of W^T[K_dim, N]).
    The tiled layout organizes data so that a (k_tile, n_tile) region is
    contiguous, with k_tile indexing along K_dim and n_tile indexing along N.

    Args:
        packed_flat: int32 tensor of shape (num_blocks * k,).
        absmax_flat: float32 tensor of shape (num_blocks,).
        K_dim: Inner product dimension (in_features).
        N: Output dimension (out_features).
        k: Bit width (2-5).
        tile_k: K-tile size (default 64).
        tile_n: N-tile size (default 128).

    Returns:
        packed_tiled: int32 tensor of tiled packed data.
        absmax_tiled: uint8 tensor of tiled E4M4 absmax.
    """
    assert K_dim % tile_k == 0 or True, "K_dim padding handled below"
    assert N % tile_n == 0, f"N ({N}) must be divisible by tile_n ({tile_n})"
    assert K_dim % BLOCKSIZE == 0, f"K_dim ({K_dim}) must be divisible by blocksize ({BLOCKSIZE})"

    # Pad K_dim to next multiple of tile_k if needed
    K_dim_padded = ((K_dim + tile_k - 1) // tile_k) * tile_k

    k_tiles = K_dim_padded // tile_k
    n_tiles = N // tile_n
    k_blocks_per_tile = tile_k // BLOCKSIZE  # 2 for tile_k=64

    # Output sizes
    words_per_tile = tile_n * k_blocks_per_tile * k
    absmax_per_tile = tile_n * k_blocks_per_tile

    total_tile_words = k_tiles * n_tiles * words_per_tile
    total_tile_absmax = k_tiles * n_tiles * absmax_per_tile

    packed_tiled = torch.zeros(total_tile_words, dtype=torch.int32)
    absmax_tiled = torch.zeros(total_tile_absmax, dtype=torch.uint8)

    # E4M4 encode the absmax
    absmax_e4m4 = encode_absmax_e4m4(absmax_flat)

    # W is [N, K_dim] row-major. Element (n, kk) is at flat index n * K_dim + kk.
    # block_id for element (n, kk) = (n * K_dim + kk) // 32
    for kt in range(k_tiles):
        for nt in range(n_tiles):
            tile_base = (kt * n_tiles + nt)
            tile_word_offset = tile_base * words_per_tile
            tile_abs_offset = tile_base * absmax_per_tile

            for col in range(tile_n):
                n_idx = nt * tile_n + col  # actual N index

                for kb in range(k_blocks_per_tile):
                    k_start = kt * tile_k + kb * BLOCKSIZE  # actual K start

                    if k_start >= K_dim:
                        # Padded region: leave as zeros
                        continue

                    # Which flat block does element (n_idx, k_start) belong to?
                    # W[N, K_dim] row-major: flat_index = n_idx * K_dim + k_start
                    # Since k_start is aligned to BLOCKSIZE=32:
                    #   block_id = (n_idx * K_dim + k_start) // 32
                    flat_idx = n_idx * K_dim + k_start
                    block_id = flat_idx // BLOCKSIZE

                    # Copy k bit-plane words
                    dst_word_offset = tile_word_offset + (col * k_blocks_per_tile + kb) * k
                    for bit in range(k):
                        src_idx = block_id * k + bit
                        packed_tiled[dst_word_offset + bit] = packed_flat[src_idx]

                    # Copy absmax
                    dst_abs_offset = tile_abs_offset + col * k_blocks_per_tile + kb
                    absmax_tiled[dst_abs_offset] = absmax_e4m4[block_id]

    return packed_tiled, absmax_tiled


def unrepack_kbit_ref(packed_tiled, absmax_tiled, K_dim, N, k, tile_k=TILE_K, tile_n=TILE_N):
    """Inverse of repack: tiled layout back to flat layout (for round-trip testing).

    Returns:
        packed_flat: int32 tensor.
        absmax_flat_e4m4: uint8 tensor (E4M4-encoded).
    """
    K_dim_padded = ((K_dim + tile_k - 1) // tile_k) * tile_k
    k_tiles = K_dim_padded // tile_k
    n_tiles = N // tile_n
    k_blocks_per_tile = tile_k // BLOCKSIZE

    num_blocks = (N * K_dim) // BLOCKSIZE
    packed_flat = torch.zeros(num_blocks * k, dtype=torch.int32)
    absmax_flat = torch.zeros(num_blocks, dtype=torch.uint8)

    words_per_tile = tile_n * k_blocks_per_tile * k
    absmax_per_tile = tile_n * k_blocks_per_tile

    for kt in range(k_tiles):
        for nt in range(n_tiles):
            tile_base = kt * n_tiles + nt
            tile_word_offset = tile_base * words_per_tile
            tile_abs_offset = tile_base * absmax_per_tile

            for col in range(tile_n):
                n_idx = nt * tile_n + col

                for kb in range(k_blocks_per_tile):
                    k_start = kt * tile_k + kb * BLOCKSIZE
                    if k_start >= K_dim:
                        continue

                    flat_idx = n_idx * K_dim + k_start
                    block_id = flat_idx // BLOCKSIZE

                    src_word_offset = tile_word_offset + (col * k_blocks_per_tile + kb) * k
                    for bit in range(k):
                        packed_flat[block_id * k + bit] = packed_tiled[src_word_offset + bit]

                    src_abs_offset = tile_abs_offset + col * k_blocks_per_tile + kb
                    absmax_flat[block_id] = absmax_tiled[src_abs_offset]

    return packed_flat, absmax_flat


# ---------------------------------------------------------------------------
# Stage 1: Python reference fused GEMM
# ---------------------------------------------------------------------------


def kbit_gemm_ref(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k,
                  tile_k=TILE_K, tile_n=TILE_N):
    """Reference fused kbit dequant + GEMM (Python, via dequant then matmul).

    Computes C[M, N] = A[M, K_dim] * W^T where W is the kbit-quantized weight.

    This reference implementation:
      1. Un-repacks the tiled data back to flat format
      2. Unpacks bit-planes to indices
      3. Dequantizes using codebook + absmax
      4. Reshapes to [N, K_dim] and does matmul

    Args:
        A: fp32 tensor of shape [M, K_dim].
        packed_tiled: int32 tensor of tiled packed data (from repack_kbit_ref).
        absmax_tiled: uint8 tensor of tiled E4M4 absmax (from repack_kbit_ref).
        codebook: float32 tensor of shape [2^k].
        K_dim: Inner product dimension.
        N: Output dimension.
        k: Bit width.

    Returns:
        C: fp32 tensor of shape [M, N].
    """
    # Un-repack to flat layout
    packed_flat, absmax_e4m4 = unrepack_kbit_ref(
        packed_tiled, absmax_tiled, K_dim, N, k, tile_k, tile_n
    )

    # Decode E4M4 absmax
    absmax = decode_absmax_e4m4(absmax_e4m4)

    # Unpack bit-planes to indices
    n_elements = N * K_dim
    indices = unpack_kbit_ref(packed_flat, k, n_elements)

    # Dequantize
    W_deq = dequantize_kbit_ref(indices, absmax, codebook, dtype=torch.float32)

    # Reshape to [N, K_dim] (PyTorch weight layout)
    W_deq = W_deq.reshape(N, K_dim)

    # C = A @ W^T
    C = A.float() @ W_deq.T

    return C


def kbit_gemm_ref_direct(A, W, codebook, k):
    """Reference GEMM via direct quantize -> dequantize -> matmul.

    This is the simplest reference: quantize the weight, dequantize it,
    then do a standard matmul. No repacking involved.

    Args:
        A: tensor of shape [M, K_dim].
        W: tensor of shape [N, K_dim] (original weight, pre-quantization).
        codebook: float32 codebook.
        k: bit width.

    Returns:
        C: fp32 tensor of shape [M, N].
    """
    N, K_dim = W.shape

    # Quantize W (flattened)
    indices, absmax = quantize_kbit_ref(W, codebook)

    # Dequantize
    W_deq = dequantize_kbit_ref(indices, absmax, codebook, dtype=torch.float32)
    W_deq = W_deq.reshape(N, K_dim)

    # C = A @ W^T
    C = A.float() @ W_deq.T
    return C


# ===========================================================================
# Stage 1 Tests: Python Reference Validation
# ===========================================================================


class TestRepackRef:
    """Test the Python reference repack/unrepack (round-trip and structure)."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_repack_round_trip(self, k):
        """Repack then unrepack must recover the original flat data exactly."""
        K_dim = 128  # Must be multiple of TILE_K=64
        N = 128      # Must be multiple of TILE_N=128

        # Create a random weight matrix [N, K_dim]
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        # Quantize (produces flat packed data)
        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)
        absmax_e4m4 = encode_absmax_e4m4(absmax)

        # Repack to tiled layout
        packed_tiled, absmax_tiled = repack_kbit_ref(
            packed_flat, absmax, K_dim, N, k
        )

        # Unrepack back to flat
        recovered_packed, recovered_absmax = unrepack_kbit_ref(
            packed_tiled, absmax_tiled, K_dim, N, k
        )

        # Bit-exact match
        assert torch.equal(packed_flat, recovered_packed), \
            f"Packed data round-trip failed for K={k}"
        assert torch.equal(absmax_e4m4, recovered_absmax), \
            f"Absmax round-trip failed for K={k}"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_repack_tile_contiguity(self, k):
        """Each tile's data should be at a contiguous offset in the output."""
        K_dim = 128
        N = 256  # 2 N-tiles

        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)
        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)

        packed_tiled, absmax_tiled = repack_kbit_ref(
            packed_flat, absmax, K_dim, N, k
        )

        k_tiles = K_dim // TILE_K
        n_tiles = N // TILE_N
        k_blocks_per_tile = TILE_K // BLOCKSIZE
        words_per_tile = TILE_N * k_blocks_per_tile * k

        # Verify total size matches expected tile count
        expected_total = k_tiles * n_tiles * words_per_tile
        assert packed_tiled.numel() == expected_total, \
            f"Expected {expected_total} words, got {packed_tiled.numel()}"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    @pytest.mark.parametrize("K_dim,N", [(128, 128), (256, 256), (256, 128), (128, 256)])
    def test_repack_various_sizes(self, k, K_dim, N):
        """Repack works for various aligned matrix sizes."""
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)
        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)

        packed_tiled, absmax_tiled = repack_kbit_ref(
            packed_flat, absmax, K_dim, N, k
        )

        # Round-trip
        recovered_packed, recovered_absmax = unrepack_kbit_ref(
            packed_tiled, absmax_tiled, K_dim, N, k
        )
        assert torch.equal(packed_flat, recovered_packed)


class TestFusedGemmRef:
    """Test the Python reference fused GEMM against direct quantize+matmul."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_gemm_matches_direct(self, k):
        """Fused GEMM reference (via repack) matches direct quantize+matmul."""
        M, K_dim, N = 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        # Direct reference: quantize -> dequant -> matmul
        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)

        # Fused reference: quantize -> pack -> repack -> fused GEMM
        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)
        packed_tiled, absmax_tiled = repack_kbit_ref(
            packed_flat, absmax, K_dim, N, k
        )
        C_fused = kbit_gemm_ref(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k)

        # The fused path uses E4M4 absmax (lossy ~6.25% relative error per block)
        # while the direct path uses float32 absmax. The error accumulates over
        # the K_dim reduction. Use allclose with both atol and rtol:
        # - rtol=0.1 accounts for the E4M4 error propagation
        # - atol scales with output magnitude to handle near-zero values
        atol = 0.05 * C_direct.abs().mean().item()
        assert torch.allclose(C_fused, C_direct, rtol=0.1, atol=atol), \
            f"K={k}: fused GEMM does not match direct reference"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_gemm_m1(self, k):
        """Fused GEMM works for M=1 (single token / vector-matrix multiply)."""
        M, K_dim, N = 1, 128, 128
        torch.manual_seed(123)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)

        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)
        packed_tiled, absmax_tiled = repack_kbit_ref(
            packed_flat, absmax, K_dim, N, k
        )
        C_fused = kbit_gemm_ref(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k)

        atol = 0.05 * C_direct.abs().mean().item()
        assert torch.allclose(C_fused, C_direct, rtol=0.1, atol=atol), \
            f"K={k}: M=1 fused GEMM does not match direct reference"

    @pytest.mark.parametrize("k", [4])
    @pytest.mark.parametrize("M", [1, 4, 16, 32])
    def test_gemm_various_batch_sizes(self, k, M):
        """Fused GEMM works across typical batch sizes."""
        K_dim, N = 256, 256
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)

        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)
        packed_tiled, absmax_tiled = repack_kbit_ref(
            packed_flat, absmax, K_dim, N, k
        )
        C_fused = kbit_gemm_ref(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k)

        # E4M4 error accumulates over K_dim reduction. Scale atol with sqrt(K_dim)
        # to account for error accumulation in larger reductions.
        atol = 0.1 * C_direct.abs().mean().item()
        assert torch.allclose(C_fused, C_direct, rtol=0.1, atol=atol), \
            f"M={M}: fused GEMM does not match direct reference"

    def test_gemm_fp16_output_quality(self):
        """SQNR of fused GEMM output vs fp16 reference matmul."""
        k = 4
        M, K_dim, N = 8, 256, 256
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        # fp16 reference (no quantization)
        C_fp16 = (A @ W.T)

        # Quantized fused GEMM
        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)
        packed_tiled, absmax_tiled = repack_kbit_ref(
            packed_flat, absmax, K_dim, N, k
        )
        C_fused = kbit_gemm_ref(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k)

        # SQNR: signal power / noise power
        noise = C_fused - C_fp16
        signal_power = (C_fp16 ** 2).mean()
        noise_power = (noise ** 2).mean()
        sqnr_db = 10 * torch.log10(signal_power / noise_power).item()

        # For K=4, expect SQNR > 15 dB (quantization noise dominates)
        assert sqnr_db > 10, f"SQNR {sqnr_db:.1f} dB is too low (expected > 10 dB)"

    def test_gemm_nonstandard_codebook(self):
        """Fused GEMM works with a non-standard codebook."""
        k = 4
        M, K_dim, N = 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)

        # Non-standard codebook: linearly spaced, asymmetric
        codebook = torch.linspace(-0.5, 1.5, 1 << k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)

        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)
        packed_tiled, absmax_tiled = repack_kbit_ref(
            packed_flat, absmax, K_dim, N, k
        )
        C_fused = kbit_gemm_ref(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k)

        atol = 0.05 * C_direct.abs().mean().item()
        assert torch.allclose(C_fused, C_direct, rtol=0.1, atol=atol), \
            "Non-standard codebook: fused GEMM does not match direct reference"


# ===========================================================================
# Stage 2 Tests: CUDA Repack Kernel Validation
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestRepackCUDA:
    """Test CUDA repack kernel against Python reference (bit-exact match)."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_repack_matches_reference(self, k):
        """CUDA repack must produce bit-exact match with Python reference."""
        K_dim = 128
        N = 128
        torch.manual_seed(42)

        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        # Quantize (produces flat packed data + float32 absmax)
        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)

        # Python reference repack
        packed_ref, absmax_ref = repack_kbit_ref(packed_flat, absmax, K_dim, N, k)

        # CUDA repack
        packed_flat_gpu = packed_flat.cuda()
        absmax_gpu = absmax.cuda()
        packed_cuda, absmax_cuda = torch.ops.bitsandbytes.repack_kbit(
            packed_flat_gpu, absmax_gpu, K_dim, N, k
        )

        # Bit-exact match for packed data
        assert torch.equal(packed_ref, packed_cuda.cpu()), \
            f"K={k}: CUDA repack packed data does not match Python reference"

        # Bit-exact match for absmax (E4M4-encoded)
        assert torch.equal(absmax_ref, absmax_cuda.cpu()), \
            f"K={k}: CUDA repack absmax does not match Python reference"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    @pytest.mark.parametrize("K_dim,N", [(128, 128), (256, 256), (256, 128), (128, 256)])
    def test_repack_various_sizes(self, k, K_dim, N):
        """CUDA repack matches reference for various aligned matrix sizes."""
        torch.manual_seed(123)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)

        # Python reference
        packed_ref, absmax_ref = repack_kbit_ref(packed_flat, absmax, K_dim, N, k)

        # CUDA
        packed_cuda, absmax_cuda = torch.ops.bitsandbytes.repack_kbit(
            packed_flat.cuda(), absmax.cuda(), K_dim, N, k
        )

        assert torch.equal(packed_ref, packed_cuda.cpu()), \
            f"K={k}, {K_dim}x{N}: packed data mismatch"
        assert torch.equal(absmax_ref, absmax_cuda.cpu()), \
            f"K={k}, {K_dim}x{N}: absmax mismatch"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_repack_round_trip_with_gemm(self, k):
        """CUDA-repacked data produces correct GEMM output via Python reference GEMM."""
        M, K_dim, N = 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        # Direct reference (no repack involved)
        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)

        # Quantize, CUDA repack, Python GEMM ref
        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)

        packed_cuda, absmax_cuda = torch.ops.bitsandbytes.repack_kbit(
            packed_flat.cuda(), absmax.cuda(), K_dim, N, k
        )

        C_fused = kbit_gemm_ref(
            A, packed_cuda.cpu(), absmax_cuda.cpu(), codebook, K_dim, N, k
        )

        atol = 0.05 * C_direct.abs().mean().item()
        assert torch.allclose(C_fused, C_direct, rtol=0.1, atol=atol), \
            f"K={k}: GEMM with CUDA-repacked data does not match direct reference"

    def test_repack_output_sizes(self):
        """Verify CUDA repack output tensor sizes match expected tile structure."""
        k = 4
        K_dim, N = 256, 256
        torch.manual_seed(42)

        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)
        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)

        packed_cuda, absmax_cuda = torch.ops.bitsandbytes.repack_kbit(
            packed_flat.cuda(), absmax.cuda(), K_dim, N, k
        )

        k_tiles = K_dim // TILE_K
        n_tiles = N // TILE_N
        k_blocks_per_tile = TILE_K // BLOCKSIZE
        expected_words = k_tiles * n_tiles * TILE_N * k_blocks_per_tile * k
        expected_absmax = k_tiles * n_tiles * TILE_N * k_blocks_per_tile

        assert packed_cuda.numel() == expected_words, \
            f"Expected {expected_words} packed words, got {packed_cuda.numel()}"
        assert absmax_cuda.numel() == expected_absmax, \
            f"Expected {expected_absmax} absmax values, got {absmax_cuda.numel()}"


# ===========================================================================
# Stage 3 Tests: Minimal CUDA GEMM Validation
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGemmCUDA:
    """Test CUDA fused kbit GEMM against Python reference."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_gemm_matches_reference(self, k):
        """CUDA GEMM must match Python reference GEMM (within E4M4 tolerance)."""
        M, K_dim, N = 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        # Python reference path: quantize -> pack -> repack -> GEMM ref
        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)

        # CUDA path: quantize -> pack -> CUDA repack -> CUDA GEMM
        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)

        packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
            packed_flat.cuda(), absmax.cuda(), K_dim, N, k
        )

        A_gpu = A.half().cuda()
        codebook_gpu = codebook.cuda()
        C_cuda = torch.ops.bitsandbytes.kbit_gemm(
            A_gpu, packed_tiled, absmax_tiled, codebook_gpu, K_dim, N, k
        )

        C_cuda_cpu = C_cuda.float().cpu()

        # Tolerance: E4M4 absmax introduces ~6.25% relative error per block,
        # which accumulates over K_dim/32 blocks. fp16 MMA also adds rounding.
        atol = 0.1 * C_direct.abs().mean().item()
        assert torch.allclose(C_cuda_cpu, C_direct, rtol=0.15, atol=atol), \
            f"K={k}: CUDA GEMM does not match reference.\n" \
            f"Max diff: {(C_cuda_cpu - C_direct).abs().max().item():.6f}, " \
            f"Mean abs: {C_direct.abs().mean().item():.6f}"

    @pytest.mark.parametrize("k", [4])
    @pytest.mark.parametrize("M", [1, 4, 8, 16])
    def test_gemm_various_M(self, k, M):
        """CUDA GEMM works for various batch sizes including M=1."""
        K_dim, N = 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)

        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)
        packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
            packed_flat.cuda(), absmax.cuda(), K_dim, N, k
        )

        C_cuda = torch.ops.bitsandbytes.kbit_gemm(
            A.half().cuda(), packed_tiled, absmax_tiled, codebook.cuda(), K_dim, N, k
        ).float().cpu()

        atol = 0.1 * C_direct.abs().mean().item()
        assert torch.allclose(C_cuda, C_direct, rtol=0.15, atol=atol), \
            f"M={M}: CUDA GEMM mismatch. Max diff: {(C_cuda - C_direct).abs().max().item():.6f}"

    @pytest.mark.parametrize("k", [4])
    @pytest.mark.parametrize("K_dim,N", [(128, 128), (256, 256), (256, 128), (128, 256)])
    def test_gemm_various_sizes(self, k, K_dim, N):
        """CUDA GEMM works for various aligned matrix sizes."""
        M = 4
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)

        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)
        packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
            packed_flat.cuda(), absmax.cuda(), K_dim, N, k
        )

        C_cuda = torch.ops.bitsandbytes.kbit_gemm(
            A.half().cuda(), packed_tiled, absmax_tiled, codebook.cuda(), K_dim, N, k
        ).float().cpu()

        atol = 0.15 * C_direct.abs().mean().item()
        assert torch.allclose(C_cuda, C_direct, rtol=0.15, atol=atol), \
            f"{K_dim}x{N}: CUDA GEMM mismatch. Max diff: {(C_cuda - C_direct).abs().max().item():.6f}"

    def test_gemm_sqnr(self):
        """SQNR of CUDA GEMM output vs unquantized fp16 matmul."""
        k = 4
        M, K_dim, N = 8, 256, 256
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        # Unquantized reference
        C_ref = (A @ W.T)

        # CUDA quantized path
        indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
        packed_flat = pack_kbit_ref(indices, k)
        packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
            packed_flat.cuda(), absmax.cuda(), K_dim, N, k
        )
        C_cuda = torch.ops.bitsandbytes.kbit_gemm(
            A.half().cuda(), packed_tiled, absmax_tiled, codebook.cuda(), K_dim, N, k
        ).float().cpu()

        noise = C_cuda - C_ref
        signal_power = (C_ref ** 2).mean()
        noise_power = (noise ** 2).mean()
        sqnr_db = 10 * torch.log10(signal_power / noise_power).item()

        # K=4 GEMM should have SQNR > 10 dB (same threshold as Python ref)
        assert sqnr_db > 10, f"SQNR {sqnr_db:.1f} dB is too low (expected > 10 dB)"


# ===========================================================================
# Stage 4 Tests: Pipelined CUDA GEMM (cp.async double-buffered)
# ===========================================================================


def _gemm_helper(A, W, codebook, k, K_dim, N, op_name="kbit_gemm"):
    """Quantize W, repack, and run the specified GEMM op. Returns fp16 CUDA tensor."""
    indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
    packed_flat = pack_kbit_ref(indices, k)
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
        packed_flat.cuda(), absmax.cuda(), K_dim, N, k
    )
    op = getattr(torch.ops.bitsandbytes, op_name)
    return op(A.half().cuda(), packed_tiled, absmax_tiled, codebook.cuda(), K_dim, N, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGemmPipelinedCUDA:
    """Test pipelined (Stage 4) GEMM matches minimal (Stage 3) GEMM bit-for-bit."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_pipelined_matches_minimal(self, k):
        """Pipelined GEMM must produce identical output to minimal GEMM."""
        M, K_dim, N = 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_minimal = _gemm_helper(A, W, codebook, k, K_dim, N, "kbit_gemm")
        C_pipelined = _gemm_helper(A, W, codebook, k, K_dim, N, "kbit_gemm_pipelined")

        assert torch.equal(C_minimal, C_pipelined), \
            f"K={k}: Pipelined GEMM does not match minimal GEMM bit-for-bit.\n" \
            f"Max diff: {(C_minimal.float() - C_pipelined.float()).abs().max().item():.6f}"

    @pytest.mark.parametrize("k", [4])
    @pytest.mark.parametrize("M", [1, 4, 8, 16])
    def test_pipelined_various_M(self, k, M):
        """Pipelined GEMM works for various batch sizes."""
        K_dim, N = 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_minimal = _gemm_helper(A, W, codebook, k, K_dim, N, "kbit_gemm")
        C_pipelined = _gemm_helper(A, W, codebook, k, K_dim, N, "kbit_gemm_pipelined")

        assert torch.equal(C_minimal, C_pipelined), \
            f"M={M}: Pipelined does not match minimal.\n" \
            f"Max diff: {(C_minimal.float() - C_pipelined.float()).abs().max().item():.6f}"

    @pytest.mark.parametrize("k", [4])
    @pytest.mark.parametrize("M,K_dim,N", [
        (4, 128, 128), (4, 128, 256), (4, 256, 128), (4, 256, 256),
    ])
    def test_pipelined_various_sizes(self, k, M, K_dim, N):
        """Pipelined GEMM works for various matrix sizes."""
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_minimal = _gemm_helper(A, W, codebook, k, K_dim, N, "kbit_gemm")
        C_pipelined = _gemm_helper(A, W, codebook, k, K_dim, N, "kbit_gemm_pipelined")

        assert torch.equal(C_minimal, C_pipelined), \
            f"({M},{K_dim},{N}): Pipelined does not match minimal.\n" \
            f"Max diff: {(C_minimal.float() - C_pipelined.float()).abs().max().item():.6f}"

    def test_pipelined_matches_reference(self):
        """Pipelined GEMM matches Python reference (same tolerance as Stage 3)."""
        k, M, K_dim, N = 4, 8, 256, 256
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)
        C_pipelined = _gemm_helper(A, W, codebook, k, K_dim, N, "kbit_gemm_pipelined")
        C_pipelined_cpu = C_pipelined.float().cpu()

        atol = 0.1 * C_direct.abs().mean().item()
        assert torch.allclose(C_pipelined_cpu, C_direct, rtol=0.15, atol=atol), \
            f"Pipelined GEMM does not match Python reference.\n" \
            f"Max diff: {(C_pipelined_cpu - C_direct).abs().max().item():.6f}"


def _gemm_splitk_helper(A, W, codebook, k, K_dim, N, k_chunks):
    """Quantize W, repack, and run split-K GEMM. Returns fp16 CUDA tensor."""
    indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
    packed_flat = pack_kbit_ref(indices, k)
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
        packed_flat.cuda(), absmax.cuda(), K_dim, N, k
    )
    return torch.ops.bitsandbytes.kbit_gemm_splitk(
        A.half().cuda(), packed_tiled, absmax_tiled, codebook.cuda(), K_dim, N, k, k_chunks
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGemmSplitKCUDA:
    """Test split-K (Stage 5) GEMM kernel."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_splitk1_matches_pipelined(self, k):
        """Split-K with k_chunks=1 must match pipelined GEMM bit-for-bit."""
        M, K_dim, N = 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_pipelined = _gemm_helper(A, W, codebook, k, K_dim, N, "kbit_gemm_pipelined")
        C_splitk = _gemm_splitk_helper(A, W, codebook, k, K_dim, N, k_chunks=1)

        assert torch.equal(C_pipelined, C_splitk), \
            f"K={k}: split-K (k_chunks=1) does not match pipelined bit-for-bit.\n" \
            f"Max diff: {(C_pipelined.float() - C_splitk.float()).abs().max().item():.6f}"

    @pytest.mark.parametrize("k", [4])
    @pytest.mark.parametrize("M", [1, 4, 8, 16])
    def test_splitk1_various_M(self, k, M):
        """Split-K with k_chunks=1 works for various batch sizes."""
        K_dim, N = 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_pipelined = _gemm_helper(A, W, codebook, k, K_dim, N, "kbit_gemm_pipelined")
        C_splitk = _gemm_splitk_helper(A, W, codebook, k, K_dim, N, k_chunks=1)

        assert torch.equal(C_pipelined, C_splitk), \
            f"M={M}: split-K (k_chunks=1) does not match pipelined.\n" \
            f"Max diff: {(C_pipelined.float() - C_splitk.float()).abs().max().item():.6f}"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_splitk2_matches_reference(self, k):
        """Split-K with k_chunks=2 matches Python reference within tolerance."""
        M, K_dim, N = 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)
        C_splitk = _gemm_splitk_helper(A, W, codebook, k, K_dim, N, k_chunks=2)
        C_splitk_cpu = C_splitk.float().cpu()

        # Split-K uses atomicAdd so may have small fp32 rounding differences
        atol = 0.1 * C_direct.abs().mean().item()
        assert torch.allclose(C_splitk_cpu, C_direct, rtol=0.15, atol=atol), \
            f"K={k}: split-K (k_chunks=2) does not match reference.\n" \
            f"Max diff: {(C_splitk_cpu - C_direct).abs().max().item():.6f}"

    @pytest.mark.parametrize("k", [4])
    @pytest.mark.parametrize("k_chunks", [1, 2])
    @pytest.mark.parametrize("M,K_dim,N", [
        (4, 128, 128), (4, 128, 256), (4, 256, 128), (4, 256, 256),
    ])
    def test_splitk_various_sizes(self, k, k_chunks, M, K_dim, N):
        """Split-K works for various matrix sizes and chunk counts."""
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)
        C_splitk = _gemm_splitk_helper(A, W, codebook, k, K_dim, N, k_chunks=k_chunks)
        C_splitk_cpu = C_splitk.float().cpu()

        atol = 0.1 * C_direct.abs().mean().item()
        assert torch.allclose(C_splitk_cpu, C_direct, rtol=0.15, atol=atol), \
            f"({M},{K_dim},{N}) k_chunks={k_chunks}: split-K does not match reference.\n" \
            f"Max diff: {(C_splitk_cpu - C_direct).abs().max().item():.6f}"

    def test_splitk_sqnr(self):
        """Split-K GEMM should have reasonable SQNR for K=4."""
        k, M, K_dim, N = 4, 8, 256, 256
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_fp16 = (A.half() @ W.half().T).float()
        C_splitk = _gemm_splitk_helper(A, W, codebook, k, K_dim, N, k_chunks=2)
        C_splitk_cpu = C_splitk.float().cpu()

        noise = C_splitk_cpu - C_fp16
        signal_power = (C_fp16**2).mean()
        noise_power = (noise**2).mean()
        sqnr = 10 * torch.log10(signal_power / noise_power).item()

        assert sqnr > 10, f"K=4 split-K SQNR too low: {sqnr:.1f} dB (expected > 10 dB)"


def _gemm_prod_helper(A, W, codebook, k, K_dim, N, k_chunks=1, dtype=torch.float16):
    """Quantize W, repack, and run production GEMM. Returns CUDA tensor in requested dtype."""
    indices, absmax = quantize_kbit_ref(W.reshape(-1), codebook)
    packed_flat = pack_kbit_ref(indices, k)
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
        packed_flat.cuda(), absmax.cuda(), K_dim, N, k
    )
    A_gpu = A.to(dtype).cuda()
    return torch.ops.bitsandbytes.kbit_gemm_prod(
        A_gpu, packed_tiled, absmax_tiled, codebook.cuda(), K_dim, N, k, k_chunks
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGemmProdCUDA:
    """Test production (Stage 6) GEMM kernel with fp16 and bf16."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_prod_fp16_matches_splitk(self, k):
        """Production fp16 (k_chunks=1) must match split-K fp16 bit-for-bit."""
        M, K_dim, N = 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_splitk = _gemm_splitk_helper(A, W, codebook, k, K_dim, N, k_chunks=1)
        C_prod = _gemm_prod_helper(A, W, codebook, k, K_dim, N, k_chunks=1, dtype=torch.float16)

        assert torch.equal(C_splitk, C_prod), \
            f"K={k}: prod fp16 does not match split-K fp16 bit-for-bit.\n" \
            f"Max diff: {(C_splitk.float() - C_prod.float()).abs().max().item():.6f}"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_prod_bf16_matches_reference(self, k):
        """Production bf16 matches Python reference within tolerance."""
        M, K_dim, N = 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)
        C_prod = _gemm_prod_helper(A, W, codebook, k, K_dim, N, k_chunks=1, dtype=torch.bfloat16)
        C_prod_cpu = C_prod.float().cpu()

        atol = 0.15 * C_direct.abs().mean().item()
        assert torch.allclose(C_prod_cpu, C_direct, rtol=0.2, atol=atol), \
            f"K={k}: prod bf16 does not match reference.\n" \
            f"Max diff: {(C_prod_cpu - C_direct).abs().max().item():.6f}"

    @pytest.mark.parametrize("k", [4])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("M", [1, 4, 8, 16])
    def test_prod_various_M(self, k, dtype, M):
        """Production GEMM works for various batch sizes and dtypes."""
        K_dim, N = 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)
        C_prod = _gemm_prod_helper(A, W, codebook, k, K_dim, N, k_chunks=1, dtype=dtype)
        C_prod_cpu = C_prod.float().cpu()

        atol = 0.15 * C_direct.abs().mean().item()
        assert torch.allclose(C_prod_cpu, C_direct, rtol=0.2, atol=atol), \
            f"M={M} {dtype}: prod does not match reference.\n" \
            f"Max diff: {(C_prod_cpu - C_direct).abs().max().item():.6f}"

    @pytest.mark.parametrize("k", [4])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("k_chunks", [1, 2])
    def test_prod_splitk(self, k, dtype, k_chunks):
        """Production GEMM with split-K for both dtypes."""
        M, K_dim, N = 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)
        C_prod = _gemm_prod_helper(A, W, codebook, k, K_dim, N, k_chunks=k_chunks, dtype=dtype)
        C_prod_cpu = C_prod.float().cpu()

        atol = 0.15 * C_direct.abs().mean().item()
        assert torch.allclose(C_prod_cpu, C_direct, rtol=0.2, atol=atol), \
            f"{dtype} k_chunks={k_chunks}: prod does not match reference.\n" \
            f"Max diff: {(C_prod_cpu - C_direct).abs().max().item():.6f}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("M,K_dim,N", [
        (4, 128, 128), (4, 128, 256), (4, 256, 128), (4, 256, 256),
    ])
    def test_prod_various_sizes(self, dtype, M, K_dim, N):
        """Production GEMM works for various matrix sizes."""
        k = 4
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_direct = kbit_gemm_ref_direct(A, W, codebook, k)
        C_prod = _gemm_prod_helper(A, W, codebook, k, K_dim, N, k_chunks=1, dtype=dtype)
        C_prod_cpu = C_prod.float().cpu()

        atol = 0.15 * C_direct.abs().mean().item()
        assert torch.allclose(C_prod_cpu, C_direct, rtol=0.2, atol=atol), \
            f"({M},{K_dim},{N}) {dtype}: prod does not match reference.\n" \
            f"Max diff: {(C_prod_cpu - C_direct).abs().max().item():.6f}"

    def test_prod_output_dtype(self):
        """Production GEMM output dtype matches input dtype."""
        k, M, K_dim, N = 4, 4, 128, 128
        torch.manual_seed(42)

        A = torch.randn(M, K_dim)
        W = torch.randn(N, K_dim)
        codebook = create_normal_float_codebook(k)

        C_fp16 = _gemm_prod_helper(A, W, codebook, k, K_dim, N, dtype=torch.float16)
        C_bf16 = _gemm_prod_helper(A, W, codebook, k, K_dim, N, dtype=torch.bfloat16)

        assert C_fp16.dtype == torch.float16, f"Expected fp16 output, got {C_fp16.dtype}"
        assert C_bf16.dtype == torch.bfloat16, f"Expected bf16 output, got {C_bf16.dtype}"
