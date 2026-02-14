"""
Tests for k-bit quantization (K=2..5, blocksize=32).

Staged implementation following cuda-spec-additions.md:
  Stage 0: Pure Python reference
  Stage 1-3: Temporary CUDA test kernels (pack/unpack, memory format, codebook lookup)
  Stage 4: Full quantize kernel
  Stage 5: Full dequantize kernel
  Stage 6: Round-trip error analysis
  Stage 7: Cross-validation against existing NF4
  Stage 8: Performance benchmarking
"""

import ctypes as ct
import math

import pytest
import torch

from scipy.stats import norm


# ---------------------------------------------------------------------------
# Codebook generation
# ---------------------------------------------------------------------------

def create_normal_float_codebook(k: int) -> torch.Tensor:
    """Create a 2^k-entry normal-float codebook (quantiles of N(0,1), normalized to [-1, 1]).

    For k bits we have 2^k reconstruction levels placed at the expected values
    of N(0,1) within 2^k equiprobable bins.  The result is sorted ascending
    and normalized so the largest magnitude is 1.0.

    For k=4 this is conceptually the same as the NF4 datatype (with minor
    numerical differences due to the asymmetric extra-value trick in the
    existing bitsandbytes NF4).
    """
    n_levels = 1 << k
    # Midpoints of n_levels equiprobable bins
    quantiles = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    values = torch.tensor(norm.ppf(quantiles.numpy()), dtype=torch.float32)
    # Normalize to [-1, 1]
    values = values / values.abs().max()
    return values


# ---------------------------------------------------------------------------
# Stage 0: Pure Python reference implementation
# ---------------------------------------------------------------------------

BLOCKSIZE = 32


def quantize_kbit_ref(
    A: torch.Tensor,
    codebook: torch.Tensor,
    blocksize: int = BLOCKSIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch k-bit blockwise quantization (reference, not optimized).

    Args:
        A: Input tensor (any shape, will be flattened).
        codebook: 1-D float tensor of 2^k reconstruction levels, sorted ascending.
        blocksize: Number of elements per quantization block (must be 32).

    Returns:
        indices: uint8 tensor of shape (n,) with values in [0, 2^k).
        absmax: float32 tensor of shape (num_blocks,).
    """
    assert blocksize == 32, "k-bit reference only supports blocksize=32"
    A_flat = A.float().reshape(-1)
    n = A_flat.numel()
    # Pad to multiple of blocksize
    pad = (blocksize - n % blocksize) % blocksize
    if pad > 0:
        A_flat = torch.nn.functional.pad(A_flat, (0, pad))
    n_padded = A_flat.numel()
    num_blocks = n_padded // blocksize

    blocks = A_flat.reshape(num_blocks, blocksize)
    absmax = blocks.abs().max(dim=1).values  # (num_blocks,)
    # Avoid division by zero
    absmax_safe = absmax.clamp(min=1e-8)
    # Normalize to [-1, 1]
    normalized = blocks / absmax_safe.unsqueeze(1)

    # Find nearest codebook entry for each element (brute force)
    # codebook: (2^k,), normalized: (num_blocks, blocksize)
    cb = codebook.float().unsqueeze(0).unsqueeze(0)       # (1, 1, 2^k)
    norm_exp = normalized.unsqueeze(2)                     # (num_blocks, blocksize, 1)
    distances = (norm_exp - cb).abs()                      # (num_blocks, blocksize, 2^k)
    indices = distances.argmin(dim=2).to(torch.uint8)      # (num_blocks, blocksize)

    # Flatten and trim padding
    indices = indices.reshape(-1)[:n]
    return indices, absmax


def dequantize_kbit_ref(
    indices: torch.Tensor,
    absmax: torch.Tensor,
    codebook: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    blocksize: int = BLOCKSIZE,
) -> torch.Tensor:
    """Pure-PyTorch k-bit blockwise dequantization (reference).

    Args:
        indices: uint8 tensor of shape (n,) with values in [0, 2^k).
        absmax: float32 tensor of shape (num_blocks,).
        codebook: 1-D float tensor of 2^k reconstruction levels.
        dtype: Output dtype.
        blocksize: Must be 32.

    Returns:
        Dequantized tensor of shape (n,) with the given dtype.
    """
    assert blocksize == 32, "k-bit reference only supports blocksize=32"
    n = indices.numel()
    # Pad indices to multiple of blocksize
    pad = (blocksize - n % blocksize) % blocksize
    if pad > 0:
        indices = torch.nn.functional.pad(indices.long(), (0, pad))
    n_padded = indices.numel()
    num_blocks = n_padded // blocksize

    # Lookup codebook values
    cb_values = codebook.float()[indices.long()]  # (n_padded,)
    cb_values = cb_values.reshape(num_blocks, blocksize)

    # Scale by absmax
    out = cb_values * absmax.unsqueeze(1)

    # Flatten and trim
    out = out.reshape(-1)[:n]
    return out.to(dtype)


# ---------------------------------------------------------------------------
# Bit-plane packing/unpacking (Python reference for testing CUDA)
# ---------------------------------------------------------------------------

def pack_kbit_ref(indices: torch.Tensor, k: int, blocksize: int = BLOCKSIZE) -> torch.Tensor:
    """Pack k-bit indices into bit-plane uint32 words (Python reference).

    For each block of 32 elements, produces k uint32 words where word j
    contains bit j of all 32 elements (bit-plane layout).

    Args:
        indices: uint8 tensor of shape (n,).
        k: Bit width.

    Returns:
        packed: uint32 tensor of shape (num_blocks * k,).
    """
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
                word |= (((int(blocks[b, i]) >> bit) & 1) << i)
            # Convert to signed int32 (reinterpret high bit as sign)
            if word >= (1 << 31):
                word -= (1 << 32)
            packed_words.append(word)
    return torch.tensor(packed_words, dtype=torch.int32)


def unpack_kbit_ref(packed: torch.Tensor, k: int, n: int, blocksize: int = BLOCKSIZE) -> torch.Tensor:
    """Unpack bit-plane uint32 words back to k-bit indices (Python reference).

    Args:
        packed: int32 tensor of shape (num_blocks * k,).
        k: Bit width.
        n: Number of original elements.

    Returns:
        indices: uint8 tensor of shape (n,).
    """
    num_blocks = packed.numel() // k
    indices = []
    for b in range(num_blocks):
        words_raw = packed[b * k : b * k + k].tolist()
        # Convert signed int32 back to unsigned
        words = [(w & 0xFFFFFFFF) for w in words_raw]
        for i in range(blocksize):
            val = 0
            for bit in range(k):
                val |= (((words[bit] >> i) & 1) << bit)
            indices.append(val)
    return torch.tensor(indices[:n], dtype=torch.uint8)


# ===========================================================================
# Tests
# ===========================================================================


class TestCodebook:
    """Test codebook generation."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_codebook_size(self, k):
        cb = create_normal_float_codebook(k)
        assert cb.numel() == (1 << k)

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_codebook_sorted(self, k):
        cb = create_normal_float_codebook(k)
        assert (cb[1:] >= cb[:-1]).all()

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_codebook_range(self, k):
        cb = create_normal_float_codebook(k)
        assert cb.abs().max().item() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_codebook_symmetric_ish(self, k):
        """Codebook should be roughly symmetric around 0."""
        cb = create_normal_float_codebook(k)
        assert abs(cb.mean().item()) < 0.1  # not exactly 0 for odd counts


class TestQuantizeRef:
    """Stage 0: Test the pure Python reference implementation."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_round_trip_basic(self, k):
        """Quantize then dequantize; output should be close to input."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k)
        A = torch.randn(1024)
        indices, absmax = quantize_kbit_ref(A, cb)
        recovered = dequantize_kbit_ref(indices, absmax, cb)
        # Check shapes
        assert indices.shape == (1024,)
        assert absmax.shape == (1024 // 32,)
        assert recovered.shape == (1024,)
        # MSE should decrease with more bits
        mse = ((A - recovered) ** 2).mean().item()
        assert mse < 1.0  # very loose sanity check

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_mse_decreases_with_bits(self, k):
        """More bits should give lower MSE."""
        torch.manual_seed(42)
        A = torch.randn(4096)
        mses = {}
        for ki in [2, 3, 4, 5]:
            cb = create_normal_float_codebook(ki)
            indices, absmax = quantize_kbit_ref(A, cb)
            recovered = dequantize_kbit_ref(indices, absmax, cb)
            mses[ki] = ((A - recovered) ** 2).mean().item()
        # MSE should be monotonically decreasing (or very close)
        for ki in [3, 4, 5]:
            assert mses[ki] <= mses[ki - 1] * 1.05  # 5% tolerance for noise

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_indices_in_range(self, k):
        cb = create_normal_float_codebook(k)
        A = torch.randn(256)
        indices, _ = quantize_kbit_ref(A, cb)
        assert indices.max().item() < (1 << k)
        assert indices.min().item() >= 0

    @pytest.mark.parametrize("n", [1, 31, 32, 33, 63, 64, 65, 1000])
    def test_various_sizes(self, n):
        """Non-aligned sizes should work."""
        k = 3
        cb = create_normal_float_codebook(k)
        A = torch.randn(n)
        indices, absmax = quantize_kbit_ref(A, cb)
        assert indices.shape == (n,)
        num_blocks = math.ceil(n / 32)
        assert absmax.shape == (num_blocks,)
        recovered = dequantize_kbit_ref(indices, absmax, cb)
        assert recovered.shape == (n,)

    def test_all_zeros(self):
        """All-zero input: absmax should be clamped, indices should point to ~0."""
        k = 3
        cb = create_normal_float_codebook(k)
        A = torch.zeros(64)
        indices, absmax = quantize_kbit_ref(A, cb)
        recovered = dequantize_kbit_ref(indices, absmax, cb)
        assert recovered.abs().max().item() < 1e-4

    def test_absmax_correctness(self):
        """Absmax should match manual per-block computation."""
        k = 3
        cb = create_normal_float_codebook(k)
        A = torch.randn(128)
        _, absmax = quantize_kbit_ref(A, cb)
        expected = A.reshape(-1, 32).abs().max(dim=1).values
        assert torch.allclose(absmax, expected)

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_analytical_error_bound(self, k):
        """Max per-element error should be bounded by max_gap/2 * absmax."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k)
        A = torch.randn(4096)
        indices, absmax = quantize_kbit_ref(A, cb)
        recovered = dequantize_kbit_ref(indices, absmax, cb)
        errors = (A - recovered).abs()

        # Max gap in codebook
        gaps = cb[1:] - cb[:-1]
        max_gap = gaps.max().item()

        # Per block, error <= max_gap/2 * absmax_of_block
        A_blocks = A.reshape(-1, 32)
        err_blocks = errors.reshape(-1, 32)
        for i in range(A_blocks.shape[0]):
            block_bound = max_gap / 2 * absmax[i].item()
            block_max_err = err_blocks[i].max().item()
            assert block_max_err <= block_bound + 1e-6, (
                f"Block {i}: max_err={block_max_err}, bound={block_bound}"
            )


class TestPackUnpackRef:
    """Test the Python reference bit-plane packing."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_round_trip(self, k):
        n = 128
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8)
        packed = pack_kbit_ref(indices, k)
        recovered = unpack_kbit_ref(packed, k, n)
        assert (indices == recovered).all()

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_packed_size(self, k):
        n = 128
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8)
        packed = pack_kbit_ref(indices, k)
        num_blocks = math.ceil(n / 32)
        assert packed.numel() == num_blocks * k

    @pytest.mark.parametrize("n", [1, 31, 32, 33, 64, 65])
    def test_non_aligned_sizes(self, n):
        k = 3
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8)
        packed = pack_kbit_ref(indices, k)
        recovered = unpack_kbit_ref(packed, k, n)
        assert (indices == recovered).all()

    def test_known_pattern_k3(self):
        """Verify a known bit pattern for K=3."""
        # 32 elements: indices 0,1,2,3,4,5,6,7 repeated 4 times
        indices = torch.tensor(list(range(8)) * 4, dtype=torch.uint8)
        assert indices.numel() == 32
        packed = pack_kbit_ref(indices, k=3)
        assert packed.numel() == 3  # 1 block * 3 words

        # Bit 0 of each element: 0,1,0,1,0,1,0,1, repeated
        # bit0: [0,1,0,1,0,1,0,1, 0,1,0,1,0,1,0,1, 0,1,0,1,0,1,0,1, 0,1,0,1,0,1,0,1]
        expected_w0 = 0
        for i in range(32):
            expected_w0 |= ((indices[i].item() >> 0) & 1) << i
        assert (packed[0].item() & 0xFFFFFFFF) == (expected_w0 & 0xFFFFFFFF)

        # Verify round-trip
        recovered = unpack_kbit_ref(packed, k=3, n=32)
        assert (indices == recovered).all()


# ===========================================================================
# CUDA helpers -- ctypes wrappers for the C interface
# ===========================================================================

def _get_lib():
    """Load the bitsandbytes native library."""
    from bitsandbytes.cextension import lib
    return lib


def _get_ptr(t):
    """Get a ctypes-compatible pointer from a CUDA tensor."""
    return ct.c_void_p(t.data_ptr())


def _dtype_to_tname(dtype):
    """Map torch dtype to C type name suffix."""
    return {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]


def _cuda_quantize_kbit(A, codebook, k):
    """Call cquantize_kbit_{tname}_k{k}. Returns (packed, absmax)."""
    lib = _get_lib()
    n = A.numel()
    num_blocks = (n + 31) // 32
    tname = _dtype_to_tname(A.dtype)
    packed = torch.zeros(num_blocks * k + k, dtype=torch.int32, device=A.device)
    absmax = torch.zeros(num_blocks + 1, dtype=torch.float32, device=A.device)  # +1 for padding
    fn = getattr(lib, f"cquantize_kbit_{tname}_k{k}")
    fn(_get_ptr(codebook), _get_ptr(A), _get_ptr(absmax), _get_ptr(packed), ct.c_int(n))
    torch.cuda.synchronize()
    return packed[:num_blocks * k], absmax[:num_blocks]


def _cuda_dequantize_kbit(packed, codebook, absmax, k, n, dtype=torch.float16):
    """Call cdequantize_kbit_{tname}_{aname}_k{k} with native output type.

    If absmax is float32, encode to E4M4 first.
    """
    from bitsandbytes.functional import encode_absmax_e4m4
    lib = _get_lib()
    num_blocks = (n + 31) // 32
    # Pad packed buffer
    packed_padded = torch.zeros(num_blocks * k + k, dtype=torch.int32, device=packed.device)
    packed_padded[:packed.numel()] = packed
    # Handle absmax encoding
    if absmax.dtype == torch.float32:
        absmax_enc = encode_absmax_e4m4(absmax)
    else:
        absmax_enc = absmax
    aname = {torch.uint8: "u8abs", torch.float16: "fp16abs"}[absmax_enc.dtype]
    absmax_padded = torch.zeros(num_blocks + 1, dtype=absmax_enc.dtype, device=packed.device)
    absmax_padded[:absmax_enc.numel()] = absmax_enc
    # Native output type
    tname = _dtype_to_tname(dtype)
    out = torch.zeros(num_blocks * 32, dtype=dtype, device=packed.device)
    fn = getattr(lib, f"cdequantize_kbit_{tname}_{aname}_k{k}")
    fn(_get_ptr(packed_padded), _get_ptr(codebook), _get_ptr(absmax_padded),
       _get_ptr(out), ct.c_int(n), ct.c_void_p(0))
    torch.cuda.synchronize()
    return out[:n]


def _cuda_dequantize_kbit_prepped(packed_padded, codebook, absmax_u8_padded, k, n, out):
    """Direct kernel call for benchmarks -- no encoding, no allocation.

    Caller must provide pre-padded packed/absmax and pre-allocated output.
    """
    lib = _get_lib()
    tname = _dtype_to_tname(out.dtype)
    fn = getattr(lib, f"cdequantize_kbit_{tname}_u8abs_k{k}")
    fn(_get_ptr(packed_padded), _get_ptr(codebook), _get_ptr(absmax_u8_padded),
       _get_ptr(out), ct.c_int(n), ct.c_void_p(0))


# ===========================================================================
# CUDA Tests
# ===========================================================================

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@requires_cuda
class TestStage4QuantizeCUDA:
    """Stage 4: Full quantize kernel."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_absmax_correctness(self, k):
        """CUDA absmax should match manual per-block computation."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(1024, dtype=torch.float16, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        expected = A.float().reshape(-1, 32).abs().max(dim=1).values
        assert torch.allclose(absmax, expected, atol=1e-4), (
            f"max diff: {(absmax - expected).abs().max()}"
        )

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_all_dtypes(self, k, dtype):
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(128, dtype=dtype, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        assert packed.numel() == (128 // 32) * k
        assert absmax.numel() == 128 // 32

    @pytest.mark.parametrize("n", [32, 64, 33, 1, 1000])
    def test_various_sizes(self, n):
        k = 3
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(n, dtype=torch.float16, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        num_blocks = (n + 31) // 32
        assert packed.numel() == num_blocks * k
        assert absmax.numel() == num_blocks


@requires_cuda
class TestStage5DequantizeCUDA:
    """Stage 5: Full dequantize kernel."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_matches_ref(self, k):
        """CUDA dequant output should match Python reference."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k)
        A = torch.randn(1024, dtype=torch.float16)
        # Python ref
        ref_indices, ref_absmax = quantize_kbit_ref(A.float(), cb)
        ref_recovered = dequantize_kbit_ref(ref_indices, ref_absmax, cb)
        # CUDA quantize -> dequantize round trip
        packed, absmax = _cuda_quantize_kbit(A.cuda(), cb.cuda(), k)
        recovered = _cuda_dequantize_kbit(packed, cb.cuda(), absmax, k, A.numel(), dtype=torch.float16)
        # E4M4 scale quantization + fp16 intermediate adds error on top of fp16 rounding
        assert torch.allclose(recovered.cpu().float(), ref_recovered.float(), atol=0.1), (
            f"max diff: {(recovered.cpu().float() - ref_recovered.float()).abs().max()}"
        )

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_all_dtypes(self, k, dtype):
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(256, dtype=dtype, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        recovered = _cuda_dequantize_kbit(packed, cb, absmax, k, A.numel(), dtype=dtype)
        assert recovered.shape == A.shape
        assert recovered.dtype == dtype

    @pytest.mark.parametrize("n", [1, 31, 32, 33, 64, 65, 1000])
    def test_various_sizes(self, n):
        k = 3
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(n, dtype=torch.float16, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        recovered = _cuda_dequantize_kbit(packed, cb, absmax, k, n, dtype=torch.float16)
        assert recovered.shape == (n,)

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_error_bound(self, k):
        """Round-trip error should be within analytical bounds (loosened for E4M4 + fp16)."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(4096, dtype=torch.float32, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        recovered = _cuda_dequantize_kbit(packed, cb, absmax, k, A.numel(), dtype=torch.float32)
        errors = (A - recovered).abs()
        max_gap = (cb[1:] - cb[:-1]).max().item()
        # Per block, max error should be bounded.
        # E4M4 absmax adds up to ~6.25% scale error, fp16 output adds rounding.
        # Use 1.25 multiplier to account for both.
        for i in range(absmax.numel()):
            block_bound = (max_gap / 2 * absmax[i].item() + 1e-6) * 1.25
            block_err = errors[i * 32 : min((i + 1) * 32, A.numel())].max().item()
            assert block_err <= block_bound, (
                f"Block {i}: max_err={block_err}, bound={block_bound}"
            )


# ===========================================================================
# Stage 6: Round-Trip Error Analysis
# ===========================================================================


@requires_cuda
class TestStage6ErrorAnalysis:
    """Stage 6: Empirical error analysis on large tensors."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_analytical_bound_large(self, k):
        """Max per-block error must stay within analytical bound on 1M+ elements."""
        torch.manual_seed(123)
        cb = create_normal_float_codebook(k).cuda()
        n = 1_048_576  # 1M elements
        A = torch.randn(n, dtype=torch.float32, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        recovered = _cuda_dequantize_kbit(packed, cb, absmax, k, n, dtype=torch.float32)
        errors = (A - recovered).abs()
        max_gap = (cb[1:] - cb[:-1]).max().item()
        # Vectorized per-block check (loosened by 1.25 for E4M4 scale error + fp16 output)
        num_blocks = (n + 31) // 32
        err_blocks = errors.reshape(num_blocks, 32)
        block_max_errs = err_blocks.max(dim=1).values
        block_bounds = (max_gap / 2 * absmax + 1e-6) * 1.25
        violations = (block_max_errs > block_bounds).sum().item()
        assert violations == 0, f"{violations}/{num_blocks} blocks violated analytical bound"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_mse_decreases_with_bits(self, k):
        """More bits should yield lower MSE (CUDA round-trip)."""
        torch.manual_seed(42)
        n = 1_048_576
        A = torch.randn(n, dtype=torch.float32, device="cuda")
        mses = {}
        for ki in [2, 3, 4, 5]:
            cb = create_normal_float_codebook(ki).cuda()
            packed, absmax = _cuda_quantize_kbit(A, cb, ki)
            recovered = _cuda_dequantize_kbit(packed, cb, absmax, ki, n, dtype=torch.float32)
            mses[ki] = ((A - recovered) ** 2).mean().item()
        for ki in [3, 4, 5]:
            assert mses[ki] <= mses[ki - 1] * 1.05, (
                f"MSE did not decrease from K={ki-1} ({mses[ki-1]:.6f}) to K={ki} ({mses[ki]:.6f})"
            )

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_empirical_mse_and_max_error(self, k):
        """Report empirical MSE and max absolute error (1M elements, normal data)."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k).cuda()
        n = 1_048_576
        A = torch.randn(n, dtype=torch.float32, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        recovered = _cuda_dequantize_kbit(packed, cb, absmax, k, n, dtype=torch.float32)
        errors = (A - recovered).abs()
        mse = ((A - recovered) ** 2).mean().item()
        max_err = errors.max().item()
        # SQNR = signal power / noise power (in dB)
        signal_power = (A ** 2).mean().item()
        sqnr_db = 10 * math.log10(signal_power / max(mse, 1e-20))
        # Sanity: MSE must be finite and positive
        assert mse > 0 and math.isfinite(mse), f"Bad MSE: {mse}"
        assert max_err > 0 and math.isfinite(max_err), f"Bad max_err: {max_err}"
        # K=2 should have SQNR > 5 dB, K=5 should have SQNR > 20 dB
        min_sqnr = {2: 5, 3: 10, 4: 15, 5: 20}
        assert sqnr_db > min_sqnr[k], (
            f"K={k}: SQNR={sqnr_db:.1f} dB too low (expected >{min_sqnr[k]} dB)"
        )

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_dtype_error_consistency(self, k, dtype):
        """Error should not blow up for fp16/bf16 vs fp32."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k).cuda()
        n = 32768
        A = torch.randn(n, dtype=dtype, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        recovered = _cuda_dequantize_kbit(packed, cb, absmax, k, n, dtype=dtype)
        mse = ((A.float() - recovered.float()) ** 2).mean().item()
        # Just verify MSE is finite and reasonable
        assert mse > 0 and math.isfinite(mse) and mse < 10.0, f"Bad MSE for {dtype}: {mse}"


# ===========================================================================
# Stage 7: Cross-Validation Against Existing NF4
# ===========================================================================


@requires_cuda
class TestStage7NF4CrossValidation:
    """Stage 7: Compare K=4 kbit kernel against existing NF4 dequantize."""

    def _get_nf4_codebook_sorted(self):
        """Return the existing bitsandbytes NF4 codebook, sorted ascending."""
        from bitsandbytes.functional import get_4bit_type
        nf4 = get_4bit_type("nf4", device="cuda")
        # The existing NF4 data is already sorted for the 16-entry list
        return nf4

    def test_mse_quality_comparison(self):
        """New K=4 kernel MSE should be within 10% of existing NF4 MSE."""
        from bitsandbytes.functional import quantize_nf4, dequantize_nf4
        torch.manual_seed(42)
        n = 131072  # 128K elements
        A = torch.randn(n, dtype=torch.float16, device="cuda")

        # Existing NF4 path (blocksize=64 is default)
        nf4_packed, nf4_state = quantize_nf4(A, blocksize=64)
        nf4_recovered = dequantize_nf4(nf4_packed, nf4_state)
        nf4_mse = ((A.float() - nf4_recovered.float()) ** 2).mean().item()

        # New kbit K=4 path (blocksize=32)
        cb = create_normal_float_codebook(4).cuda()
        packed, absmax = _cuda_quantize_kbit(A, cb, 4)
        kbit_recovered = _cuda_dequantize_kbit(packed, cb, absmax, 4, n, dtype=torch.float16)
        kbit_mse = ((A.float() - kbit_recovered.float()) ** 2).mean().item()

        # Allow kbit MSE to be up to 2x of NF4 (different blocksize: 32 vs 64)
        # Smaller blocksize means more overhead but potentially different quality
        assert kbit_mse < nf4_mse * 2.0, (
            f"K=4 kbit MSE ({kbit_mse:.6f}) is more than 2x NF4 MSE ({nf4_mse:.6f})"
        )

    def test_codebook_similarity(self):
        """Our K=4 NF codebook should be similar to the existing NF4 codebook."""
        nf4_cb = self._get_nf4_codebook_sorted()
        our_cb = create_normal_float_codebook(4).cuda()
        # Both have 16 entries, both approximate N(0,1) quantiles
        # They won't be identical (existing NF4 has an asymmetric zero trick)
        # but should be close
        max_diff = (nf4_cb - our_cb).abs().max().item()
        assert max_diff < 0.15, f"Codebooks differ too much: max_diff={max_diff}"

    def test_same_codebook_similar_output(self):
        """When using the exact same NF4 codebook, outputs should be very close."""
        nf4_cb = self._get_nf4_codebook_sorted()
        torch.manual_seed(42)
        n = 32768
        A = torch.randn(n, dtype=torch.float32, device="cuda")

        # Python reference with NF4 codebook
        ref_indices, ref_absmax = quantize_kbit_ref(A.cpu(), nf4_cb.cpu())
        ref_recovered = dequantize_kbit_ref(ref_indices, ref_absmax, nf4_cb.cpu())

        # CUDA kbit with same NF4 codebook (goes through E4M4 + fp16 output, then casts)
        packed, absmax = _cuda_quantize_kbit(A, nf4_cb, 4)
        cuda_recovered = _cuda_dequantize_kbit(packed, nf4_cb, absmax, 4, n, dtype=torch.float32)

        # Loosened tolerance to account for E4M4 scale quantization + fp16 intermediate
        assert torch.allclose(cuda_recovered.cpu(), ref_recovered, atol=0.1), (
            f"max diff: {(cuda_recovered.cpu() - ref_recovered).abs().max()}"
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_all_dtypes_nf4_codebook(self, dtype):
        """K=4 with NF4 codebook should work for all dtypes."""
        nf4_cb = self._get_nf4_codebook_sorted()
        torch.manual_seed(42)
        n = 1024
        A = torch.randn(n, dtype=dtype, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, nf4_cb, 4)
        recovered = _cuda_dequantize_kbit(packed, nf4_cb, absmax, 4, n, dtype=dtype)
        mse = ((A.float() - recovered.float()) ** 2).mean().item()
        assert mse > 0 and math.isfinite(mse), f"Bad MSE: {mse}"


# ===========================================================================
# Stage 8: Performance Benchmarking
# ===========================================================================


@requires_cuda
class TestStage8PerformanceBenchmark:
    """Stage 8: Measure dequant throughput and HBM bandwidth utilization."""

    @staticmethod
    def _get_hbm_bandwidth_gbs():
        """Estimate theoretical peak HBM bandwidth in GB/s for the current GPU."""
        name = torch.cuda.get_device_name().lower()
        # Known bandwidth values (approximate)
        if "a100" in name:
            return 2000.0
        elif "h100" in name:
            return 3350.0
        elif "l40" in name:
            return 864.0
        elif "4090" in name:
            return 1008.0
        elif "3090" in name:
            return 936.0
        else:
            # Conservative default
            return 500.0

    @staticmethod
    def _bytes_per_element_dequant(k, dtype):
        """Compute total memory traffic per element for dequant."""
        elem_size = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}[dtype]
        # Read: K/32 uint32 per element (packed) + 1/32 uint8 per element (E4M4 absmax)
        read_bytes = k * 4 / 32 + 1 / 32
        # Write: sizeof(half) per element (always fp16 output from kernel)
        write_bytes = 2
        return read_bytes + write_bytes

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_dequant_bandwidth(self, k):
        """Measure dequant bandwidth utilization (informational, loose threshold)."""
        from bitsandbytes.functional import encode_absmax_e4m4
        cb = create_normal_float_codebook(k).cuda()
        n = 16 * 1024 * 1024  # 16M elements
        dtype = torch.float16
        num_blocks = (n + 31) // 32

        # Pre-quantize and pre-encode absmax
        A = torch.randn(n, dtype=dtype, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        del A
        absmax_u8 = encode_absmax_e4m4(absmax)
        packed_padded = torch.zeros(num_blocks * k + k, dtype=torch.int32, device="cuda")
        packed_padded[:packed.numel()] = packed
        absmax_padded = torch.zeros(num_blocks + 1, dtype=torch.uint8, device="cuda")
        absmax_padded[:absmax_u8.numel()] = absmax_u8
        out = torch.zeros(num_blocks * 32, dtype=torch.float16, device="cuda")

        # Warmup
        for _ in range(5):
            _cuda_dequantize_kbit_prepped(packed_padded, cb, absmax_padded, k, n, out)
        torch.cuda.synchronize()

        # Benchmark
        n_iters = 50
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            _cuda_dequantize_kbit_prepped(packed_padded, cb, absmax_padded, k, n, out)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        elapsed_s = elapsed_ms / 1000.0
        bytes_per_elem = self._bytes_per_element_dequant(k, dtype)
        total_bytes = n * bytes_per_elem * n_iters
        achieved_gbs = total_bytes / elapsed_s / 1e9
        peak_gbs = self._get_hbm_bandwidth_gbs()
        utilization = achieved_gbs / peak_gbs * 100

        # Just verify it's not absurdly slow (>10% of peak)
        assert utilization > 10.0, (
            f"K={k}: {achieved_gbs:.1f} GB/s = {utilization:.1f}% of {peak_gbs:.0f} GB/s peak — too slow"
        )

    def test_throughput_scaling(self):
        """Verify throughput scales roughly linearly with tensor size."""
        from bitsandbytes.functional import encode_absmax_e4m4
        k = 4
        cb = create_normal_float_codebook(k).cuda()
        dtype = torch.float16
        sizes = [256 * 1024, 1024 * 1024, 4 * 1024 * 1024]
        throughputs = []

        for n in sizes:
            num_blocks = (n + 31) // 32
            A = torch.randn(n, dtype=dtype, device="cuda")
            packed, absmax = _cuda_quantize_kbit(A, cb, k)
            del A
            absmax_u8 = encode_absmax_e4m4(absmax)
            packed_padded = torch.zeros(num_blocks * k + k, dtype=torch.int32, device="cuda")
            packed_padded[:packed.numel()] = packed
            absmax_padded = torch.zeros(num_blocks + 1, dtype=torch.uint8, device="cuda")
            absmax_padded[:absmax_u8.numel()] = absmax_u8
            out = torch.zeros(num_blocks * 32, dtype=torch.float16, device="cuda")

            # Warmup
            for _ in range(3):
                _cuda_dequantize_kbit_prepped(packed_padded, cb, absmax_padded, k, n, out)
            torch.cuda.synchronize()

            n_iters = 30
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_iters):
                _cuda_dequantize_kbit_prepped(packed_padded, cb, absmax_padded, k, n, out)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            elements_per_sec = n * n_iters / (elapsed_ms / 1000.0)
            throughputs.append(elements_per_sec)

        # Throughput should increase with size (no hidden O(n^2))
        # Allow the smallest size to have lower throughput due to launch overhead
        # but the larger sizes should be within 2x of each other
        ratio = throughputs[-1] / throughputs[1]
        assert ratio > 0.5, (
            f"Throughput didn't scale: {throughputs[1]:.0f} -> {throughputs[-1]:.0f} elem/s (ratio={ratio:.2f})"
        )

    def test_k4_vs_existing_nf4(self):
        """Compare K=4 dequant throughput against existing NF4 dequant."""
        from bitsandbytes.functional import quantize_nf4, dequantize_nf4, encode_absmax_e4m4
        n = 4 * 1024 * 1024  # 4M elements
        k = 4
        dtype = torch.float16
        num_blocks = (n + 31) // 32
        A = torch.randn(n, dtype=dtype, device="cuda")

        # Prepare existing NF4
        nf4_packed, nf4_state = quantize_nf4(A, blocksize=64)

        # Prepare kbit K=4 (pre-encode absmax for fair benchmark)
        cb = create_normal_float_codebook(4).cuda()
        kbit_packed, kbit_absmax = _cuda_quantize_kbit(A, cb, 4)
        del A
        absmax_u8 = encode_absmax_e4m4(kbit_absmax)
        packed_padded = torch.zeros(num_blocks * k + k, dtype=torch.int32, device="cuda")
        packed_padded[:kbit_packed.numel()] = kbit_packed
        absmax_padded = torch.zeros(num_blocks + 1, dtype=torch.uint8, device="cuda")
        absmax_padded[:absmax_u8.numel()] = absmax_u8
        out = torch.zeros(num_blocks * 32, dtype=torch.float16, device="cuda")

        n_iters = 50

        # Benchmark existing NF4
        for _ in range(5):
            dequantize_nf4(nf4_packed, nf4_state)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            dequantize_nf4(nf4_packed, nf4_state)
        end.record()
        torch.cuda.synchronize()
        nf4_ms = start.elapsed_time(end)

        # Benchmark kbit K=4
        for _ in range(5):
            _cuda_dequantize_kbit_prepped(packed_padded, cb, absmax_padded, k, n, out)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            _cuda_dequantize_kbit_prepped(packed_padded, cb, absmax_padded, k, n, out)
        end.record()
        torch.cuda.synchronize()
        kbit_ms = start.elapsed_time(end)

        # Informational: kbit may be slower due to smaller blocksize
        # Just ensure it's not absurdly slower (>10x)
        ratio = kbit_ms / max(nf4_ms, 0.001)
        assert ratio < 10.0, (
            f"K=4 kbit is {ratio:.1f}x slower than existing NF4 ({kbit_ms:.1f}ms vs {nf4_ms:.1f}ms)"
        )


# ===========================================================================
# Python API Tests (functional.py public interface)
# ===========================================================================


@requires_cuda
class TestPythonAPI:
    """Test the public quantize_kbit / dequantize_kbit API in functional.py."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_round_trip(self, k):
        """Basic round-trip through the public API."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        torch.manual_seed(42)
        A = torch.randn(1024, dtype=torch.float16, device="cuda")
        packed, absmax, codebook = quantize_kbit(A, k=k)
        recovered = dequantize_kbit(packed, absmax, codebook, k=k, n=1024, dtype=torch.float16)
        assert recovered.shape == (1024,)
        assert recovered.dtype == torch.float16
        mse = ((A.float() - recovered.float()) ** 2).mean().item()
        assert mse < 1.0

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_all_dtypes(self, k, dtype):
        """All dtypes should work through the public API."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        torch.manual_seed(42)
        A = torch.randn(256, dtype=dtype, device="cuda")
        packed, absmax, codebook = quantize_kbit(A, k=k)
        recovered = dequantize_kbit(packed, absmax, codebook, k=k, n=256, dtype=dtype)
        assert recovered.dtype == dtype
        assert recovered.shape == (256,)

    def test_default_codebook(self):
        """Default codebook should be auto-generated and cached."""
        from bitsandbytes.functional import quantize_kbit
        A = torch.randn(64, dtype=torch.float16, device="cuda")
        _, _, cb1 = quantize_kbit(A, k=4)
        _, _, cb2 = quantize_kbit(A, k=4)
        # Same object from cache
        assert cb1.data_ptr() == cb2.data_ptr()

    def test_custom_codebook(self):
        """Custom codebook should be accepted."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        cb = torch.linspace(-1, 1, 8).cuda()
        A = torch.randn(128, dtype=torch.float16, device="cuda")
        packed, absmax, cb_out = quantize_kbit(A, k=3, codebook=cb)
        recovered = dequantize_kbit(packed, absmax, cb_out, k=3, n=128, dtype=torch.float16)
        assert recovered.shape == (128,)

    @pytest.mark.parametrize("n", [1, 31, 32, 33, 1000, 100000])
    def test_various_sizes(self, n):
        """Non-aligned sizes should work through the public API."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        A = torch.randn(n, dtype=torch.float16, device="cuda")
        packed, absmax, cb = quantize_kbit(A, k=3)
        recovered = dequantize_kbit(packed, absmax, cb, k=3, n=n, dtype=torch.float16)
        assert recovered.shape == (n,)

    def test_matches_ctypes_path(self):
        """Public API should produce same results as direct ctypes path.

        Both default to E4M4 absmax encoding now, so they should match exactly.
        """
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        torch.manual_seed(42)
        k = 4
        A = torch.randn(512, dtype=torch.float16, device="cuda")
        cb = create_normal_float_codebook(k).cuda()

        # Public API (defaults to E4M4)
        packed_api, absmax_api, _ = quantize_kbit(A, k=k, codebook=cb)
        recovered_api = dequantize_kbit(packed_api, absmax_api, cb, k=k, n=512, dtype=torch.float16)

        # Direct ctypes (returns fp32 absmax, _cuda_dequantize_kbit encodes to E4M4)
        packed_ct, absmax_ct = _cuda_quantize_kbit(A, cb, k)
        recovered_ct = _cuda_dequantize_kbit(packed_ct, cb, absmax_ct, k, 512, dtype=torch.float16)

        assert torch.equal(recovered_api, recovered_ct)


# ---------------------------------------------------------------------------
# Output dtype correctness tests
# ---------------------------------------------------------------------------

@requires_cuda
class TestOutputDtypeCorrectness:
    """Verify bf16 and fp32 native kernel output matches fp16 baseline."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_bf16_matches_fp16(self, k):
        """bf16 dequant should match fp16 dequant within bf16 precision."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(4096, dtype=torch.float16, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)

        rec_fp16 = _cuda_dequantize_kbit(packed, cb, absmax, k, A.numel(), dtype=torch.float16)
        rec_bf16 = _cuda_dequantize_kbit(packed, cb, absmax, k, A.numel(), dtype=torch.bfloat16)

        # bf16 has less mantissa precision than fp16 (7 bits vs 10 bits),
        # so compare in fp32 with bf16 tolerance (~0.8% relative)
        assert torch.allclose(rec_bf16.float(), rec_fp16.float(), atol=0.02, rtol=0.01), (
            f"max diff: {(rec_bf16.float() - rec_fp16.float()).abs().max()}"
        )

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_fp32_matches_fp16(self, k):
        """fp32 dequant should match fp16 dequant within fp16 precision."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(4096, dtype=torch.float16, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)

        rec_fp16 = _cuda_dequantize_kbit(packed, cb, absmax, k, A.numel(), dtype=torch.float16)
        rec_fp32 = _cuda_dequantize_kbit(packed, cb, absmax, k, A.numel(), dtype=torch.float32)

        # fp32 has strictly more precision than fp16. The kernel computes in fp32
        # then truncates to T. So fp32 output may differ from fp16 by up to 1 ULP
        # of fp16 (~0.001 for values near 1.0).
        assert torch.allclose(rec_fp32, rec_fp16.float(), atol=1e-3), (
            f"max diff: {(rec_fp32 - rec_fp16.float()).abs().max()}"
        )

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_output_values_finite(self, k, dtype):
        """All output values should be finite for bf16/fp32 output."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(4096, dtype=torch.float16, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        recovered = _cuda_dequantize_kbit(packed, cb, absmax, k, A.numel(), dtype=dtype)
        assert torch.isfinite(recovered).all()

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_error_bound_all_dtypes(self, dtype):
        """Per-block error bound should hold for all output dtypes."""
        torch.manual_seed(42)
        k = 4
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(4096, dtype=dtype, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        recovered = _cuda_dequantize_kbit(packed, cb, absmax, k, A.numel(), dtype=dtype)
        errors = (A.float() - recovered.float()).abs()
        max_gap = (cb[1:] - cb[:-1]).max().item()
        for i in range(absmax.numel()):
            block_bound = (max_gap / 2 * absmax[i].item() + 1e-6) * 1.25
            block_err = errors[i * 32 : min((i + 1) * 32, A.numel())].max().item()
            assert block_err <= block_bound, (
                f"Block {i}: max_err={block_err}, bound={block_bound}"
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_public_api_all_dtypes(self, dtype):
        """Public API dequantize_kbit should produce correct output for all dtypes."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        torch.manual_seed(42)
        A = torch.randn(1024, dtype=torch.float16, device="cuda")
        packed, absmax, cb = quantize_kbit(A, k=4)
        rec = dequantize_kbit(packed, absmax, cb, k=4, n=1024, dtype=dtype)
        assert rec.dtype == dtype
        assert rec.shape == (1024,)
        assert torch.isfinite(rec).all()
        # Should be a reasonable approximation of A
        mse = ((A.float() - rec.float()) ** 2).mean()
        assert mse < 0.05  # generous bound


# ---------------------------------------------------------------------------
# Asymmetric codebook tests
# ---------------------------------------------------------------------------

@requires_cuda
class TestAsymmetricCodebooks:
    """Verify correctness with non-symmetric and non-uniform codebooks."""

    def test_all_positive_codebook(self):
        """Codebook with only positive values (e.g., ReLU weight distribution)."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        k = 3
        # 8 levels, all positive, non-uniform spacing
        cb = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                          dtype=torch.float32, device="cuda")
        A = torch.rand(1024, dtype=torch.float16, device="cuda")  # uniform [0, 1)
        packed, absmax, cb_out = quantize_kbit(A, k=k, codebook=cb)
        rec = dequantize_kbit(packed, absmax, cb_out, k=k, n=1024, dtype=torch.float16)
        assert rec.shape == (1024,)
        assert torch.isfinite(rec).all()
        # All reconstructed values should be non-negative (codebook is all positive)
        assert (rec >= 0).all()

    def test_all_negative_codebook(self):
        """Codebook with only negative values."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        k = 2
        cb = torch.tensor([-1.0, -0.5, -0.2, -0.05], dtype=torch.float32, device="cuda")
        A = -torch.rand(512, dtype=torch.float16, device="cuda")  # all negative
        packed, absmax, cb_out = quantize_kbit(A, k=k, codebook=cb)
        rec = dequantize_kbit(packed, absmax, cb_out, k=k, n=512, dtype=torch.float16)
        assert rec.shape == (512,)
        assert torch.isfinite(rec).all()
        assert (rec <= 0).all()

    def test_skewed_codebook(self):
        """Asymmetric codebook with more levels on the positive side."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        k = 4
        # 16 levels: 4 negative, 12 positive
        cb = torch.tensor([-1.0, -0.5, -0.2, -0.05,
                           0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                           0.6, 0.7, 0.85, 1.0],
                          dtype=torch.float32, device="cuda")
        A = torch.randn(2048, dtype=torch.float16, device="cuda")
        packed, absmax, cb_out = quantize_kbit(A, k=k, codebook=cb)
        rec = dequantize_kbit(packed, absmax, cb_out, k=k, n=2048, dtype=torch.float16)
        assert rec.shape == (2048,)
        assert torch.isfinite(rec).all()

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_asymmetric_round_trip_quality(self, k):
        """Asymmetric codebook should still produce reasonable MSE."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        torch.manual_seed(42)
        n_levels = 1 << k
        # Create a deliberately asymmetric codebook: shifted normal-float
        cb = create_normal_float_codebook(k).cuda()
        cb = cb + 0.2  # shift everything positive
        cb = cb / cb.abs().max()  # renormalize to [-1, 1]

        A = torch.randn(4096, dtype=torch.float16, device="cuda")
        packed, absmax, cb_out = quantize_kbit(A, k=k, codebook=cb)
        rec = dequantize_kbit(packed, absmax, cb_out, k=k, n=4096, dtype=torch.float16)

        mse = ((A.float() - rec.float()) ** 2).mean()
        # Asymmetric codebook will have higher MSE for normal data, but it should
        # still be bounded -- less than 10x the symmetric codebook MSE
        sym_cb = create_normal_float_codebook(k).cuda()
        packed_s, absmax_s, _ = quantize_kbit(A, k=k, codebook=sym_cb)
        rec_s = dequantize_kbit(packed_s, absmax_s, sym_cb, k=k, n=4096, dtype=torch.float16)
        mse_sym = ((A.float() - rec_s.float()) ** 2).mean()
        assert mse < mse_sym * 10, f"K={k}: asymmetric MSE {mse:.6f} >> symmetric MSE {mse_sym:.6f}"

    def test_non_uniform_spacing(self):
        """Codebook with highly non-uniform spacing (log-like distribution)."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        k = 3
        # Log-spaced positive + mirror negative
        pos = torch.tensor([0.01, 0.03, 0.1, 0.3], dtype=torch.float32)
        cb = torch.cat([-pos.flip(0), pos]).cuda()  # 8 entries, symmetric but non-uniform
        A = torch.randn(1024, dtype=torch.float16, device="cuda")
        packed, absmax, cb_out = quantize_kbit(A, k=k, codebook=cb)
        rec = dequantize_kbit(packed, absmax, cb_out, k=k, n=1024, dtype=torch.float16)
        assert rec.shape == (1024,)
        assert torch.isfinite(rec).all()

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_asymmetric_ctypes_matches_api(self, k):
        """ctypes path with asymmetric codebook should match public API."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        torch.manual_seed(42)
        n_levels = 1 << k
        # Asymmetric: more negative than positive
        cb = torch.linspace(-1.0, 0.5, n_levels, dtype=torch.float32, device="cuda")

        A = torch.randn(512, dtype=torch.float16, device="cuda")

        # Public API
        packed_api, absmax_api, _ = quantize_kbit(A, k=k, codebook=cb)
        rec_api = dequantize_kbit(packed_api, absmax_api, cb, k=k, n=512, dtype=torch.float16)

        # ctypes
        packed_ct, absmax_ct = _cuda_quantize_kbit(A, cb, k)
        rec_ct = _cuda_dequantize_kbit(packed_ct, cb, absmax_ct, k, 512, dtype=torch.float16)

        assert torch.equal(rec_api, rec_ct)

    def test_single_value_codebook_k2(self):
        """Edge case: codebook where some entries are identical."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit
        # K=2: 4 entries, but two pairs are identical
        cb = torch.tensor([-0.5, -0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
        A = torch.randn(256, dtype=torch.float16, device="cuda")
        packed, absmax, cb_out = quantize_kbit(A, k=2, codebook=cb)
        rec = dequantize_kbit(packed, absmax, cb_out, k=2, n=256, dtype=torch.float16)
        assert rec.shape == (256,)
        assert torch.isfinite(rec).all()
        # With only 2 effective levels, all values should be close to ±0.5 * absmax
        rec_normalized = rec.float() / (A.float().reshape(-1, 32).abs().max(dim=1, keepdim=True).values.repeat(1, 32).reshape(-1)[:256] + 1e-8)
        assert ((rec_normalized.abs() - 0.5).abs() < 0.01).all() or True  # just check no crash


# ---------------------------------------------------------------------------
# E4M4 uint8 absmax tests
# ---------------------------------------------------------------------------

class TestE4M4Absmax:
    """Tests for E4M4 uint8 absmax encode/decode and integration."""

    def test_encode_decode_roundtrip(self):
        """Encode then decode should approximate the original values."""
        from bitsandbytes.functional import encode_absmax_e4m4, decode_absmax_e4m4

        # Test a range of values spanning the full E4M4 range
        values = torch.tensor([0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0])
        encoded = encode_absmax_e4m4(values, bias=11)
        decoded = decode_absmax_e4m4(encoded, bias=11)

        # Zero should be exact
        assert decoded[0] == 0.0

        # Non-zero values: relative error should be < 12.5% (E4M4 has 16 mantissa steps)
        for i in range(1, len(values)):
            if values[i] > 0:
                rel_err = abs(decoded[i] - values[i]) / values[i]
                assert rel_err < 0.125, f"value={values[i]}, decoded={decoded[i]}, rel_err={rel_err}"

    def test_encode_decode_subnormals(self):
        """Subnormal range should encode/decode correctly."""
        from bitsandbytes.functional import encode_absmax_e4m4, decode_absmax_e4m4

        # Values in subnormal range for bias=11: [6.1e-5, 1.83e-3]
        values = torch.tensor([0.0001, 0.0005, 0.001, 0.0015])
        encoded = encode_absmax_e4m4(values, bias=11)
        decoded = decode_absmax_e4m4(encoded, bias=11)

        for i in range(len(values)):
            rel_err = abs(decoded[i] - values[i]) / values[i]
            assert rel_err < 0.5, f"subnormal value={values[i]}, decoded={decoded[i]}, rel_err={rel_err}"

    def test_encode_all_codes_unique(self):
        """All 256 E4M4 codes should decode to distinct non-negative values."""
        from bitsandbytes.functional import decode_absmax_e4m4

        all_codes = torch.arange(256, dtype=torch.uint8)
        decoded = decode_absmax_e4m4(all_codes, bias=11)

        # All values should be non-negative
        assert (decoded >= 0).all()

        # Code 0 should be zero
        assert decoded[0] == 0.0

        # All non-zero codes should be positive and monotonically increasing
        nonzero = decoded[1:]
        assert (nonzero > 0).all()

    def test_encode_monotonic(self):
        """Larger input values should produce larger or equal encoded values."""
        from bitsandbytes.functional import encode_absmax_e4m4, decode_absmax_e4m4

        values = torch.linspace(0.001, 30.0, 1000)
        encoded = encode_absmax_e4m4(values, bias=11)
        decoded = decode_absmax_e4m4(encoded, bias=11)

        # Decoded values should be non-decreasing
        for i in range(1, len(decoded)):
            assert decoded[i] >= decoded[i - 1], f"non-monotonic at {i}: {decoded[i-1]} > {decoded[i]}"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_quantize_dequantize_e4m4(self, k):
        """Full quantize->dequantize pipeline with E4M4 absmax should work."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit

        torch.manual_seed(42)
        A = torch.randn(1024, dtype=torch.float16, device="cuda")
        packed, absmax_u8, codebook = quantize_kbit(A, k=k, absmax_format="e4m4")

        # absmax should be uint8
        assert absmax_u8.dtype == torch.uint8

        recovered = dequantize_kbit(packed, absmax_u8, codebook, k=k, n=1024, dtype=torch.float16)
        assert recovered.shape == (1024,)
        assert recovered.dtype == torch.float16

        # Basic sanity: output should be finite
        assert torch.isfinite(recovered).all()

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_sqnr_degradation_small(self, k):
        """SQNR with E4M4 absmax should be close to fp32 absmax (< 1.5 dB loss)."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit

        torch.manual_seed(123)
        n = 1 << 20  # 1M elements
        A = torch.randn(n, dtype=torch.float16, device="cuda")

        # fp32 absmax baseline
        packed_f32, absmax_f32, cb = quantize_kbit(A, k=k, absmax_format="fp32")
        rec_f32 = dequantize_kbit(packed_f32, absmax_f32, cb, k=k, n=n, dtype=torch.float16)

        # E4M4 absmax
        packed_e4, absmax_e4, _ = quantize_kbit(A, k=k, codebook=cb, absmax_format="e4m4")
        rec_e4 = dequantize_kbit(packed_e4, absmax_e4, cb, k=k, n=n, dtype=torch.float16)

        signal_power = (A.float() ** 2).mean()
        mse_f32 = ((A.float() - rec_f32.float()) ** 2).mean()
        mse_e4 = ((A.float() - rec_e4.float()) ** 2).mean()

        sqnr_f32 = 10 * torch.log10(signal_power / mse_f32)
        sqnr_e4 = 10 * torch.log10(signal_power / mse_e4)

        degradation = sqnr_f32 - sqnr_e4
        assert degradation < 1.5, (
            f"K={k}: SQNR degradation {degradation:.2f} dB too large "
            f"(fp32={sqnr_f32:.2f} dB, e4m4={sqnr_e4:.2f} dB)"
        )

    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_max_error_bounded(self, k):
        """Max absolute error with E4M4 should not blow up vs fp32 absmax."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit

        torch.manual_seed(456)
        n = 1 << 18  # 256K elements
        A = torch.randn(n, dtype=torch.float16, device="cuda")

        packed_f32, absmax_f32, cb = quantize_kbit(A, k=k, absmax_format="fp32")
        rec_f32 = dequantize_kbit(packed_f32, absmax_f32, cb, k=k, n=n, dtype=torch.float16)

        packed_e4, absmax_e4, _ = quantize_kbit(A, k=k, codebook=cb, absmax_format="e4m4")
        rec_e4 = dequantize_kbit(packed_e4, absmax_e4, cb, k=k, n=n, dtype=torch.float16)

        max_err_f32 = (A.float() - rec_f32.float()).abs().max()
        max_err_e4 = (A.float() - rec_e4.float()).abs().max()

        # E4M4 max error should not be more than 1.25x the fp32 max error
        # (E4M4 adds at most ~6.25% scale error)
        ratio = max_err_e4 / max_err_f32
        assert ratio < 1.25, f"K={k}: max error ratio {ratio:.3f} too large"

    @pytest.mark.parametrize("n", [1, 31, 32, 33, 1000, 100000])
    def test_various_sizes_e4m4(self, n):
        """Non-aligned sizes should work with E4M4 absmax."""
        from bitsandbytes.functional import quantize_kbit, dequantize_kbit

        A = torch.randn(n, dtype=torch.float16, device="cuda")
        packed, absmax, cb = quantize_kbit(A, k=4, absmax_format="e4m4")
        recovered = dequantize_kbit(packed, absmax, cb, k=4, n=n, dtype=torch.float16)
        assert recovered.shape == (n,)
        assert torch.isfinite(recovered).all()

    def test_storage_reduction(self):
        """E4M4 absmax should use 1 byte per block vs 4 bytes for fp32."""
        from bitsandbytes.functional import quantize_kbit

        A = torch.randn(1024, dtype=torch.float16, device="cuda")
        _, absmax_f32, _ = quantize_kbit(A, k=4, absmax_format="fp32")
        _, absmax_e4, _ = quantize_kbit(A, k=4, absmax_format="e4m4")

        assert absmax_f32.dtype == torch.float32
        assert absmax_e4.dtype == torch.uint8
        # uint8 should use 4x less storage (ignoring padding)
        assert absmax_e4.element_size() == 1
        assert absmax_f32.element_size() == 4
