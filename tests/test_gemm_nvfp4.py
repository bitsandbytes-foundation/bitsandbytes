"""Test NVFP4 GEMM kernel on SM_120 (Blackwell consumer GPUs).

Tests the block-scaled mma.sync GEMM kernel via ctypes.
Uses the CUDA quantize/dequantize kernels to prepare inputs,
ensuring the data format matches what the hardware expects.
"""

import ctypes
import os

import pytest
import torch


def get_lib():
    """Load the bitsandbytes CUDA library."""
    lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bitsandbytes")
    for suffix in ["cuda131", "cuda130"]:
        lib_path = os.path.join(lib_dir, f"libbitsandbytes_{suffix}.so")
        if os.path.exists(lib_path):
            return ctypes.cdll.LoadLibrary(lib_path)
    raise RuntimeError(f"Could not find bitsandbytes CUDA library in {lib_dir}")


def cuda_quantize_nvfp4(x, tensor_scale=None):
    """Quantize using the CUDA kernel (same as test_nvfp4.py)."""
    lib = get_lib()
    n = x.numel()
    assert n % 16 == 0
    if tensor_scale is None:
        tensor_scale = x.abs().max().item()
    packed = torch.zeros(n // 2, dtype=torch.uint8, device=x.device)
    block_scales = torch.zeros(n // 16, dtype=torch.uint8, device=x.device)
    if x.dtype == torch.float16:
        func = lib.cquantize_nvfp4_fp16
    elif x.dtype == torch.bfloat16:
        func = lib.cquantize_nvfp4_bf16
    else:
        func = lib.cquantize_nvfp4_fp32
    func(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(block_scales.data_ptr()),
        ctypes.c_float(tensor_scale),
        ctypes.c_int(n),
    )
    torch.cuda.synchronize()
    return packed, block_scales, tensor_scale


def cuda_dequantize_nvfp4(packed, block_scales, tensor_scale, n, dtype=torch.float32):
    """Dequantize using the CUDA kernel."""
    lib = get_lib()
    output = torch.zeros(n, dtype=dtype, device=packed.device)
    if dtype == torch.float16:
        func = lib.cdequantize_nvfp4_fp16
    elif dtype == torch.bfloat16:
        func = lib.cdequantize_nvfp4_bf16
    else:
        func = lib.cdequantize_nvfp4_fp32
    func(
        ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(block_scales.data_ptr()),
        ctypes.c_float(tensor_scale),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(n),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return output


def cuda_gemm_nvfp4(A_packed, B_packed, A_scales, B_scales, M, N, K):
    """Run GEMM using the CUDA kernel."""
    lib = get_lib()
    D_out = torch.zeros(M, N, dtype=torch.float32, device=A_packed.device)
    lib.cgemm_nvfp4(
        ctypes.c_void_p(A_packed.data_ptr()),
        ctypes.c_void_p(B_packed.data_ptr()),
        ctypes.c_void_p(A_scales.data_ptr()),
        ctypes.c_void_p(B_scales.data_ptr()),
        ctypes.c_void_p(D_out.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )
    torch.cuda.synchronize()
    return D_out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGemmNVFP4:
    """Test NVFP4 GEMM kernel correctness."""

    def test_identity_scales_single_tile(self):
        """All 1.0 values, scale 1.0 -> output = K (for m16n8k64)."""
        lib = get_lib()
        M, N, K = 16, 8, 64
        # E2M1 code for 1.0: magnitude index 2, sign 0 -> code 0x2
        # Pack: byte = (0x2) | (0x2 << 4) = 0x22
        A_packed = torch.full((M * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        B_packed = torch.full((N * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        # UE4M3 scale 1.0: exponent=7 (2^0), mantissa=0 -> code = 0x38
        A_scales = torch.full((M * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")
        B_scales = torch.full((N * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")

        D = cuda_gemm_nvfp4(A_packed, B_packed, A_scales, B_scales, M, N, K)
        expected = 64.0
        assert torch.allclose(D, torch.full((M, N), expected, device="cuda")), (
            f"Expected all {expected}, got min={D.min():.1f} max={D.max():.1f}"
        )

    def test_multi_k_tiles(self):
        """K > 64: verify K-loop accumulation works."""
        M, N, K = 16, 8, 128
        A_packed = torch.full((M * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        B_packed = torch.full((N * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        A_scales = torch.full((M * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")
        B_scales = torch.full((N * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")

        D = cuda_gemm_nvfp4(A_packed, B_packed, A_scales, B_scales, M, N, K)
        expected = float(K)
        assert torch.allclose(D, torch.full((M, N), expected, device="cuda")), (
            f"Expected all {expected}, got min={D.min():.1f} max={D.max():.1f}"
        )

    def test_multi_mn_tiles(self):
        """M and N > single tile: verify output tiling works."""
        M, N, K = 32, 16, 64
        A_packed = torch.full((M * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        B_packed = torch.full((N * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        A_scales = torch.full((M * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")
        B_scales = torch.full((N * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")

        D = cuda_gemm_nvfp4(A_packed, B_packed, A_scales, B_scales, M, N, K)
        expected = float(K)
        assert torch.allclose(D, torch.full((M, N), expected, device="cuda")), (
            f"Expected all {expected}, got min={D.min():.1f} max={D.max():.1f}"
        )

    def test_varied_values(self):
        """Test with non-uniform FP4 values and scale=1.0.

        A is all 2.0 (E2M1 code 0x4), B is all 0.5 (E2M1 code 0x1).
        Each output element = sum_{k=0}^{K-1} (2.0 * 0.5) = K.
        """
        M, N, K = 16, 8, 64
        # 2.0 = magnitude index 4, code = 0x4
        # Pack: byte = (0x4) | (0x4 << 4) = 0x44
        A_packed = torch.full((M * K // 2,), 0x44, dtype=torch.uint8, device="cuda")
        # 0.5 = magnitude index 1, code = 0x1
        # Pack: byte = (0x1) | (0x1 << 4) = 0x11
        B_packed = torch.full((N * K // 2,), 0x11, dtype=torch.uint8, device="cuda")
        A_scales = torch.full((M * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")
        B_scales = torch.full((N * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")

        D = cuda_gemm_nvfp4(A_packed, B_packed, A_scales, B_scales, M, N, K)
        expected = 2.0 * 0.5 * K  # = 64
        assert torch.allclose(D, torch.full((M, N), expected, device="cuda")), (
            f"Expected all {expected}, got min={D.min():.1f} max={D.max():.1f}"
        )

    def test_with_block_scales(self):
        """Test that block scales are applied correctly.

        A is all 1.0 with scale 2.0, B is all 1.0 with scale 3.0.
        Each output = sum(1.0*2.0 * 1.0*3.0) = 6.0 * K.
        """
        M, N, K = 16, 8, 64
        A_packed = torch.full((M * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        B_packed = torch.full((N * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        # UE4M3 for 2.0: exponent=8 (bias=7, 2^1=2), mantissa=0 -> code = (8<<3)|0 = 0x40
        A_scales = torch.full((M * (K // 16),), 0x40, dtype=torch.uint8, device="cuda")
        # UE4M3 for 3.0: exponent=8 (2^1=2), mantissa=4 (1 + 4/8 = 1.5, 2*1.5=3)
        # code = (8<<3)|4 = 0x44
        B_scales = torch.full((N * (K // 16),), 0x44, dtype=torch.uint8, device="cuda")

        D = cuda_gemm_nvfp4(A_packed, B_packed, A_scales, B_scales, M, N, K)
        expected = 1.0 * 2.0 * 1.0 * 3.0 * K  # = 384
        print(f"Block scales test: expected={expected}, got first element={D[0,0].item():.1f}")
        assert torch.allclose(D, torch.full((M, N), expected, device="cuda"), rtol=0.01), (
            f"Expected all {expected}, got min={D.min():.1f} max={D.max():.1f}"
        )

    def test_random_data_cuda_quantize(self):
        """Test GEMM with CUDA-quantized random data.

        Uses the CUDA quantize/dequantize kernels to prepare inputs,
        ensuring perfect format compatibility with the GEMM kernel.
        """
        torch.manual_seed(42)
        M, N, K = 16, 8, 64

        # Generate random data
        A_float = torch.randn(M, K, dtype=torch.float32, device="cuda")
        B_float = torch.randn(N, K, dtype=torch.float32, device="cuda")  # B is N x K (TN layout)

        # Quantize with CUDA kernels
        A_flat = A_float.reshape(-1)
        B_flat = B_float.reshape(-1)
        A_packed, A_scales, A_ts = cuda_quantize_nvfp4(A_flat)
        B_packed, B_scales, B_ts = cuda_quantize_nvfp4(B_flat)

        # Dequantize to get ground truth values
        A_deq = cuda_dequantize_nvfp4(A_packed, A_scales, A_ts, M * K).reshape(M, K)
        B_deq = cuda_dequantize_nvfp4(B_packed, B_scales, B_ts, N * K).reshape(N, K)

        # Reference: matmul on dequantized values
        D_ref = A_deq @ B_deq.T  # M x N

        # GEMM kernel (output doesn't include tensor scales)
        D_kernel = cuda_gemm_nvfp4(A_packed, B_packed, A_scales, B_scales, M, N, K)

        # Scale by tensor scales
        D_out = D_kernel * A_ts * B_ts

        # Compare
        abs_err = (D_out - D_ref).abs()
        max_err = abs_err.max().item()
        mean_err = abs_err.mean().item()
        ref_mag = D_ref.abs().mean().item()

        print(f"CUDA-quantized random test (M={M}, N={N}, K={K}):")
        print(f"  Reference mean magnitude: {ref_mag:.4f}")
        print(f"  Max abs error: {max_err:.4f}")
        print(f"  Mean abs error: {mean_err:.4f}")
        if ref_mag > 0:
            rel_err = mean_err / ref_mag
            print(f"  Mean relative error: {rel_err:.4f}")
            # Error should be small — both use the same quantized data
            # The only error source is the register layout mapping
            assert rel_err < 0.5, f"Relative error {rel_err:.4f} too large"

        print(f"  Output[0,:4]: {D_out[0,:4].tolist()}")
        print(f"  Reference[0,:4]: {D_ref[0,:4].tolist()}")

    def test_random_data_larger(self):
        """Test GEMM with CUDA-quantized data on a larger matrix (multiple tiles)."""
        torch.manual_seed(123)
        M, N, K = 32, 16, 128

        A_float = torch.randn(M, K, dtype=torch.float32, device="cuda")
        B_float = torch.randn(N, K, dtype=torch.float32, device="cuda")

        A_packed, A_scales, A_ts = cuda_quantize_nvfp4(A_float.reshape(-1))
        B_packed, B_scales, B_ts = cuda_quantize_nvfp4(B_float.reshape(-1))

        A_deq = cuda_dequantize_nvfp4(A_packed, A_scales, A_ts, M * K).reshape(M, K)
        B_deq = cuda_dequantize_nvfp4(B_packed, B_scales, B_ts, N * K).reshape(N, K)

        D_ref = A_deq @ B_deq.T
        D_kernel = cuda_gemm_nvfp4(A_packed, B_packed, A_scales, B_scales, M, N, K)
        D_out = D_kernel * A_ts * B_ts

        abs_err = (D_out - D_ref).abs()
        ref_mag = D_ref.abs().mean().item()
        mean_err = abs_err.mean().item()
        max_err = abs_err.max().item()

        print(f"Larger random test (M={M}, N={N}, K={K}):")
        print(f"  Reference mean magnitude: {ref_mag:.4f}")
        print(f"  Max abs error: {max_err:.4f}")
        print(f"  Mean abs error: {mean_err:.4f}")
        if ref_mag > 0:
            rel_err = mean_err / ref_mag
            print(f"  Mean relative error: {rel_err:.4f}")
            assert rel_err < 0.5, f"Relative error {rel_err:.4f} too large"

    def _run_gemm_test(self, M, N, K, seed=42):
        """Helper: quantize random data, run GEMM, compare against reference."""
        torch.manual_seed(seed)
        A_float = torch.randn(M, K, dtype=torch.float32, device="cuda")
        B_float = torch.randn(N, K, dtype=torch.float32, device="cuda")

        A_packed, A_scales, A_ts = cuda_quantize_nvfp4(A_float.reshape(-1))
        B_packed, B_scales, B_ts = cuda_quantize_nvfp4(B_float.reshape(-1))

        A_deq = cuda_dequantize_nvfp4(A_packed, A_scales, A_ts, M * K).reshape(M, K)
        B_deq = cuda_dequantize_nvfp4(B_packed, B_scales, B_ts, N * K).reshape(N, K)

        D_ref = A_deq @ B_deq.T
        D_kernel = cuda_gemm_nvfp4(A_packed, B_packed, A_scales, B_scales, M, N, K)
        D_out = D_kernel * A_ts * B_ts

        abs_err = (D_out - D_ref).abs()
        ref_mag = D_ref.abs().mean().item()
        mean_err = abs_err.mean().item()
        max_err = abs_err.max().item()

        if ref_mag > 0:
            rel_err = mean_err / ref_mag
        else:
            rel_err = mean_err

        return rel_err, max_err, mean_err, ref_mag

    def test_gemm_medium(self):
        """Medium matrices (128x128x128) — multiple tiles in all dimensions."""
        rel_err, max_err, mean_err, ref_mag = self._run_gemm_test(128, 128, 128)
        print(f"Medium (128x128x128): rel_err={rel_err:.6f}, max_err={max_err:.4f}")
        assert rel_err < 0.01, f"Relative error {rel_err:.6f} too large"

    def test_gemm_large(self):
        """Larger matrices (256x256x256)."""
        rel_err, max_err, mean_err, ref_mag = self._run_gemm_test(256, 256, 256)
        print(f"Large (256x256x256): rel_err={rel_err:.6f}, max_err={max_err:.4f}")
        assert rel_err < 0.01, f"Relative error {rel_err:.6f} too large"

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (16, 8, 128),   # Single M/N tile, multi K
            (48, 24, 64),   # M,N not multiples of tile (16,8)
            (32, 8, 192),   # K not multiple of 64 (3 K-tiles)
            (80, 40, 64),   # Larger non-aligned M,N
        ],
        ids=["16x8x128", "48x24x64", "32x8x192", "80x40x64"],
    )
    def test_gemm_various_shapes(self, M, N, K):
        """Test various matrix shapes including non-tile-aligned."""
        rel_err, max_err, mean_err, ref_mag = self._run_gemm_test(M, N, K)
        print(f"Shape ({M}x{N}x{K}): rel_err={rel_err:.6f}, ref_mag={ref_mag:.4f}")
        assert rel_err < 0.01, f"Relative error {rel_err:.6f} too large for {M}x{N}x{K}"

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 128, 64),     # Single row (batch=1 inference)
            (8, 128, 64),     # Small batch
            (32, 128, 128),   # Medium batch
        ],
        ids=["1x128x64", "8x128x64", "32x128x128"],
    )
    def test_gemm_tall_skinny(self, M, N, K):
        """Test tall/skinny shapes typical of LLM inference."""
        rel_err, max_err, mean_err, ref_mag = self._run_gemm_test(M, N, K)
        print(f"Tall/skinny ({M}x{N}x{K}): rel_err={rel_err:.6f}, ref_mag={ref_mag:.4f}")
        assert rel_err < 0.01, f"Relative error {rel_err:.6f} too large for {M}x{N}x{K}"


class TestGemmNVFP4Output:
    """Test GEMM with NVFP4 output (layer chaining) via Python API."""

    def test_gemm_nvfp4_output_basic(self):
        """GEMM with NVFP4 output: quantize → GEMM → quantize output → dequantize → compare."""
        from bitsandbytes.functional import (
            dequantize_nvfp4,
            gemm_nvfp4_to_nvfp4,
            quantize_nvfp4,
        )

        torch.manual_seed(42)
        M, N, K = 32, 32, 64

        A_float = torch.randn(M, K, dtype=torch.float32, device="cuda")
        B_float = torch.randn(N, K, dtype=torch.float32, device="cuda")

        # Quantize inputs
        A_packed, A_state = quantize_nvfp4(A_float)
        B_packed, B_state = quantize_nvfp4(B_float)

        # GEMM with NVFP4 output
        out_packed, out_state = gemm_nvfp4_to_nvfp4(A_packed, A_state, B_packed, B_state)

        # Dequantize output
        D_deq = dequantize_nvfp4(out_packed, out_state, out_dtype=torch.float32)

        # Reference: dequantize inputs → matmul
        A_deq = dequantize_nvfp4(A_packed, A_state, out_dtype=torch.float32)
        B_deq = dequantize_nvfp4(B_packed, B_state, out_dtype=torch.float32)
        D_ref = A_deq @ B_deq.T

        # NVFP4 output adds a second layer of quantization error
        ref_mag = D_ref.abs().mean().item()
        mean_err = (D_deq - D_ref).abs().mean().item()
        rel_err = mean_err / ref_mag if ref_mag > 0 else mean_err

        print(f"GEMM NVFP4 output (M={M}, N={N}, K={K}):")
        print(f"  Reference magnitude: {ref_mag:.4f}")
        print(f"  Mean abs error: {mean_err:.4f}")
        print(f"  Relative error: {rel_err:.4f}")
        print(f"  Output shape: {D_deq.shape}")

        assert D_deq.shape == (M, N), f"Wrong shape: {D_deq.shape}"
        # Double quantization error: once for inputs, once for output
        assert rel_err < 0.5, f"Relative error {rel_err:.4f} too large"

    def test_gemm_nvfp4_output_alpha(self):
        """GEMM with alpha scaling and NVFP4 output."""
        from bitsandbytes.functional import (
            dequantize_nvfp4,
            gemm_nvfp4,
            gemm_nvfp4_to_nvfp4,
            quantize_nvfp4,
        )

        torch.manual_seed(123)
        M, N, K = 16, 16, 64
        alpha = 2.5

        A_float = torch.randn(M, K, dtype=torch.float32, device="cuda")
        B_float = torch.randn(N, K, dtype=torch.float32, device="cuda")

        A_packed, A_state = quantize_nvfp4(A_float)
        B_packed, B_state = quantize_nvfp4(B_float)

        # GEMM without alpha (FP32 output)
        D_fp32 = gemm_nvfp4(A_packed, A_state, B_packed, B_state)

        # GEMM with alpha and NVFP4 output
        out_packed, out_state = gemm_nvfp4_to_nvfp4(
            A_packed, A_state, B_packed, B_state, alpha=alpha
        )
        D_nvfp4 = dequantize_nvfp4(out_packed, out_state, out_dtype=torch.float32)

        # Reference: alpha * FP32 output
        D_ref = D_fp32 * alpha

        # Verify alpha is reflected in the output (within NVFP4 quantization error)
        ref_mag = D_ref.abs().mean().item()
        mean_err = (D_nvfp4 - D_ref).abs().mean().item()
        rel_err = mean_err / ref_mag if ref_mag > 0 else mean_err

        print(f"Alpha test (alpha={alpha}): rel_err={rel_err:.4f}")
        assert rel_err < 0.5, f"Relative error {rel_err:.4f} too large"

    def test_gemm_nvfp4_output_non_aligned_N(self):
        """GEMM with NVFP4 output where N is not a multiple of 16."""
        from bitsandbytes.functional import (
            dequantize_nvfp4,
            gemm_nvfp4_to_nvfp4,
            quantize_nvfp4,
        )

        torch.manual_seed(77)
        M, N, K = 16, 24, 64  # N=24, not multiple of 16

        A_float = torch.randn(M, K, dtype=torch.float32, device="cuda")
        B_float = torch.randn(N, K, dtype=torch.float32, device="cuda")

        A_packed, A_state = quantize_nvfp4(A_float)
        B_packed, B_state = quantize_nvfp4(B_float)

        out_packed, out_state = gemm_nvfp4_to_nvfp4(A_packed, A_state, B_packed, B_state)
        D_deq = dequantize_nvfp4(out_packed, out_state, out_dtype=torch.float32)

        # Reference
        A_deq = dequantize_nvfp4(A_packed, A_state, out_dtype=torch.float32)
        B_deq = dequantize_nvfp4(B_packed, B_state, out_dtype=torch.float32)
        D_ref = A_deq @ B_deq.T

        assert D_deq.shape == (M, N), f"Wrong shape: {D_deq.shape}"
        ref_mag = D_ref.abs().mean().item()
        mean_err = (D_deq - D_ref).abs().mean().item()
        rel_err = mean_err / ref_mag if ref_mag > 0 else mean_err
        print(f"Non-aligned N test ({M}x{N}x{K}): rel_err={rel_err:.4f}")
        assert rel_err < 0.5, f"Relative error {rel_err:.4f} too large"


class TestNVFP4QuantStateSerialization:
    """Test NVFP4QuantState save/load."""

    def test_state_dict_round_trip(self):
        """Serialize and deserialize NVFP4QuantState."""
        from bitsandbytes.functional import NVFP4QuantState, dequantize_nvfp4, quantize_nvfp4

        torch.manual_seed(42)
        x = torch.randn(256, dtype=torch.float32, device="cuda")
        packed, state = quantize_nvfp4(x)

        # Serialize
        sd = state.state_dict()
        assert "packed_data" in sd
        assert "block_scales" in sd
        assert "tensor_scale" in sd
        assert "shape" in sd
        assert "dtype" in sd

        # Deserialize
        state2 = NVFP4QuantState.from_state_dict(sd, device="cuda")

        # Verify fields match
        assert torch.equal(state.packed_data, state2.packed_data)
        assert torch.equal(state.block_scales, state2.block_scales)
        assert state.tensor_scale == state2.tensor_scale
        assert state.shape == state2.shape
        assert state.dtype == state2.dtype
        assert state.rotated == state2.rotated

        # Verify dequantization produces same result
        out1 = dequantize_nvfp4(packed, state, out_dtype=torch.float32)
        out2 = dequantize_nvfp4(state2.packed_data, state2, out_dtype=torch.float32)
        assert torch.equal(out1, out2), "Dequantized outputs differ after serialization"

    def test_state_dict_save_load_file(self):
        """Save to file and reload."""
        import tempfile

        from bitsandbytes.functional import NVFP4QuantState, quantize_nvfp4

        torch.manual_seed(99)
        x = torch.randn(128, dtype=torch.float16, device="cuda")
        _, state = quantize_nvfp4(x)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(state.state_dict(), f.name)
            loaded = torch.load(f.name, weights_only=False)
            state2 = NVFP4QuantState.from_state_dict(loaded, device="cuda")

        assert torch.equal(state.packed_data, state2.packed_data)
        assert state.tensor_scale == state2.tensor_scale
        assert state.dtype == state2.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
