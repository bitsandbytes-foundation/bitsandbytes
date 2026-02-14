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


def _cuda_test_pack_unpack(indices, k):
    """Call ctest_pack_unpack_k{k} kernel."""
    lib = _get_lib()
    n = indices.numel()
    recovered = torch.zeros_like(indices)
    fn = getattr(lib, f"ctest_pack_unpack_k{k}")
    fn(_get_ptr(indices), _get_ptr(recovered), ct.c_int(n))
    torch.cuda.synchronize()
    return recovered


def _cuda_test_pack_write(indices, k):
    """Call ctest_pack_write_k{k} kernel. Returns packed uint32 tensor."""
    lib = _get_lib()
    n = indices.numel()
    num_blocks = (n + 31) // 32
    # Allocate packed output with K extra padding words
    packed = torch.zeros(num_blocks * k + k, dtype=torch.int32, device=indices.device)
    fn = getattr(lib, f"ctest_pack_write_k{k}")
    fn(_get_ptr(indices), _get_ptr(packed), ct.c_int(n))
    torch.cuda.synchronize()
    return packed[:num_blocks * k]  # trim padding


def _cuda_test_read_unpack(packed, k, n, device="cuda"):
    """Call ctest_read_unpack_k{k} kernel. Returns uint8 indices."""
    lib = _get_lib()
    num_blocks = (n + 31) // 32
    # Pad packed buffer with K extra words for safe out-of-bounds reads
    packed_padded = torch.zeros(num_blocks * k + k, dtype=torch.int32, device=device)
    packed_padded[:packed.numel()] = packed
    indices_out = torch.zeros(num_blocks * 32, dtype=torch.uint8, device=device)
    fn = getattr(lib, f"ctest_read_unpack_k{k}")
    fn(_get_ptr(packed_padded), _get_ptr(indices_out), ct.c_int(n))
    torch.cuda.synchronize()
    return indices_out[:n]


def _cuda_test_codebook_lookup(indices, codebook, k):
    """Call ctest_codebook_lookup_k{k} kernel. Returns float32 values."""
    lib = _get_lib()
    n = indices.numel()
    out = torch.zeros(n, dtype=torch.float32, device=indices.device)
    fn = getattr(lib, f"ctest_codebook_lookup_k{k}")
    fn(_get_ptr(indices), _get_ptr(codebook), _get_ptr(out), ct.c_int(n))
    torch.cuda.synchronize()
    return out


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
    """Call cdequantize_kbit_{tname}_k{k}. Returns output tensor."""
    lib = _get_lib()
    tname = _dtype_to_tname(dtype)
    num_blocks = (n + 31) // 32
    # Pad buffers
    packed_padded = torch.zeros(num_blocks * k + k, dtype=torch.int32, device=packed.device)
    packed_padded[:packed.numel()] = packed
    absmax_padded = torch.zeros(num_blocks + 1, dtype=torch.float32, device=packed.device)
    absmax_padded[:absmax.numel()] = absmax
    out = torch.zeros(num_blocks * 32, dtype=dtype, device=packed.device)
    fn = getattr(lib, f"cdequantize_kbit_{tname}_k{k}")
    fn(_get_ptr(packed_padded), _get_ptr(codebook), _get_ptr(absmax_padded),
       _get_ptr(out), ct.c_int(n), ct.c_void_p(0))
    torch.cuda.synchronize()
    return out[:n]


# ===========================================================================
# CUDA Tests
# ===========================================================================

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@requires_cuda
class TestStage1PackUnpackCUDA:
    """Stage 1: Pack/unpack in-warp round-trip on CUDA."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_round_trip(self, k):
        n = 128
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8, device="cuda")
        recovered = _cuda_test_pack_unpack(indices, k)
        assert (indices == recovered).all()

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    @pytest.mark.parametrize("n", [32, 64, 33, 1])
    def test_various_sizes(self, k, n):
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8, device="cuda")
        recovered = _cuda_test_pack_unpack(indices, k)
        assert (indices == recovered).all()


@requires_cuda
class TestStage2PackMemoryCUDA:
    """Stage 2: Pack-write / read-unpack persistent format on CUDA."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_round_trip(self, k):
        n = 128
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8, device="cuda")
        packed = _cuda_test_pack_write(indices, k)
        recovered = _cuda_test_read_unpack(packed, k, n)
        assert (indices == recovered).all()

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_packed_size(self, k):
        n = 128
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8, device="cuda")
        packed = _cuda_test_pack_write(indices, k)
        num_blocks = (n + 31) // 32
        assert packed.numel() == num_blocks * k

    @pytest.mark.parametrize("n", [1, 31, 32, 33, 64, 65, 1000])
    def test_non_aligned_sizes(self, n):
        k = 3
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8, device="cuda")
        packed = _cuda_test_pack_write(indices, k)
        recovered = _cuda_test_read_unpack(packed, k, n)
        assert (indices == recovered).all()

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_matches_python_ref(self, k):
        """CUDA packed output should match Python reference packing."""
        n = 64
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8, device="cuda")
        packed_cuda = _cuda_test_pack_write(indices, k)
        packed_ref = pack_kbit_ref(indices.cpu(), k)
        # Compare (both are int32, may differ in sign interpretation)
        assert ((packed_cuda.cpu().int() & 0xFFFFFFFF) == (packed_ref.int() & 0xFFFFFFFF)).all(), (
            f"CUDA packed:\n{packed_cuda.cpu()}\nRef packed:\n{packed_ref}"
        )


@requires_cuda
class TestStage3CodebookLookupCUDA:
    """Stage 3: Codebook shuffle lookup on CUDA."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_exact_lookup(self, k):
        """Shuffle lookup must produce exact codebook values."""
        cb = create_normal_float_codebook(k).cuda()
        n = 128
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8, device="cuda")
        result = _cuda_test_codebook_lookup(indices, cb, k)
        expected = cb[indices.long()]
        assert torch.equal(result, expected), f"max diff: {(result - expected).abs().max()}"

    @pytest.mark.parametrize("n", [1, 31, 32, 33, 1000])
    def test_various_sizes(self, n):
        k = 3
        cb = create_normal_float_codebook(k).cuda()
        indices = torch.randint(0, 1 << k, (n,), dtype=torch.uint8, device="cuda")
        result = _cuda_test_codebook_lookup(indices, cb, k)
        expected = cb[indices.long()]
        assert torch.equal(result, expected)


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
    def test_indices_match_ref(self, k):
        """CUDA quantized indices should match Python reference exactly."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k)
        A = torch.randn(256, dtype=torch.float16)
        # Python reference
        ref_indices, ref_absmax = quantize_kbit_ref(A.float(), cb)
        # CUDA
        packed, absmax = _cuda_quantize_kbit(A.cuda(), cb.cuda(), k)
        # Unpack CUDA output using test kernel
        cuda_indices = _cuda_test_read_unpack(packed, k, A.numel())
        assert (cuda_indices.cpu() == ref_indices).all(), (
            f"Mismatch at indices: {(cuda_indices.cpu() != ref_indices).nonzero()}"
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
        # Should be very close (float16 rounding may cause minor diffs)
        assert torch.allclose(recovered.cpu().float(), ref_recovered.float(), atol=1e-3), (
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
        """Round-trip error should be within analytical bounds."""
        torch.manual_seed(42)
        cb = create_normal_float_codebook(k).cuda()
        A = torch.randn(4096, dtype=torch.float32, device="cuda")
        packed, absmax = _cuda_quantize_kbit(A, cb, k)
        recovered = _cuda_dequantize_kbit(packed, cb, absmax, k, A.numel(), dtype=torch.float32)
        errors = (A - recovered).abs()
        max_gap = (cb[1:] - cb[:-1]).max().item()
        # Per block, max error should be bounded
        for i in range(absmax.numel()):
            block_bound = max_gap / 2 * absmax[i].item() + 1e-6
            block_err = errors[i * 32 : min((i + 1) * 32, A.numel())].max().item()
            assert block_err <= block_bound, (
                f"Block {i}: max_err={block_err}, bound={block_bound}"
            )
