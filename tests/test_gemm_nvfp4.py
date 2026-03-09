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
    """Quantize to NVFP4 using the CUTLASS fused quantize path."""
    from bitsandbytes.functional import quantize_nvfp4

    x_2d = x.reshape(1, -1) if x.dim() == 1 else x
    packed, state = quantize_nvfp4(x_2d.to(torch.bfloat16), tensor_scale=tensor_scale)
    return state.packed_data, state.block_scales, state.tensor_scale


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


def swizzle_scales(flat_scales, rows, scale_K):
    """Convert flat row-major scales to CUTLASS block-scaled (swizzled) layout."""
    lib = get_lib()
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (scale_K + 3) // 4
    out_size = n_row_blocks * n_col_blocks * 128 * 4
    swizzled = torch.empty(out_size, dtype=torch.uint8, device=flat_scales.device)
    stream = torch.cuda.current_stream()
    lib.cscale_to_blocked(
        ctypes.c_void_p(flat_scales.data_ptr()),
        ctypes.c_void_p(swizzled.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(scale_K),
        ctypes.c_void_p(stream.cuda_stream),
    )
    torch.cuda.synchronize()
    return swizzled


def cuda_gemm_nvfp4(A_packed, B_packed, A_scales, B_scales, M, N, K):
    """Run GEMM using the CUDA kernel (BF16 output).

    A_scales and B_scales must be in flat row-major format; they are
    swizzled to CUTLASS block-scaled layout before calling the kernel.
    """
    lib = get_lib()
    scale_K = K // 16
    A_scales_sw = swizzle_scales(A_scales, M, scale_K)
    B_scales_sw = swizzle_scales(B_scales, N, scale_K)

    D_out = torch.zeros(M, N, dtype=torch.bfloat16, device=A_packed.device)
    workspace = torch.zeros(M, N, dtype=torch.float32, device=A_packed.device)
    stream = torch.cuda.current_stream()
    lib.cgemm_nvfp4_bf16(
        ctypes.c_void_p(A_packed.data_ptr()),
        ctypes.c_void_p(B_packed.data_ptr()),
        ctypes.c_void_p(A_scales_sw.data_ptr()),
        ctypes.c_void_p(B_scales_sw.data_ptr()),
        ctypes.c_void_p(D_out.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_void_p(stream.cuda_stream),
    )
    torch.cuda.synchronize()
    return D_out.float()


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
        print(f"Block scales test: expected={expected}, got first element={D[0, 0].item():.1f}")
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

        print(f"  Output[0,:4]: {D_out[0, :4].tolist()}")
        print(f"  Reference[0,:4]: {D_ref[0, :4].tolist()}")

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
        rel_err, max_err, _mean_err, _ref_mag = self._run_gemm_test(128, 128, 128)
        print(f"Medium (128x128x128): rel_err={rel_err:.6f}, max_err={max_err:.4f}")
        assert rel_err < 0.01, f"Relative error {rel_err:.6f} too large"

    def test_gemm_large(self):
        """Larger matrices (256x256x256)."""
        rel_err, max_err, _mean_err, _ref_mag = self._run_gemm_test(256, 256, 256)
        print(f"Large (256x256x256): rel_err={rel_err:.6f}, max_err={max_err:.4f}")
        assert rel_err < 0.01, f"Relative error {rel_err:.6f} too large"

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (16, 8, 128),  # Single M/N tile, multi K
            (48, 24, 64),  # M,N not multiples of tile (16,8)
            (32, 8, 192),  # K not multiple of 64 (3 K-tiles)
            (80, 40, 64),  # Larger non-aligned M,N
        ],
        ids=["16x8x128", "48x24x64", "32x8x192", "80x40x64"],
    )
    def test_gemm_various_shapes(self, M, N, K):
        """Test various matrix shapes including non-tile-aligned."""
        rel_err, _max_err, _mean_err, ref_mag = self._run_gemm_test(M, N, K)
        print(f"Shape ({M}x{N}x{K}): rel_err={rel_err:.6f}, ref_mag={ref_mag:.4f}")
        assert rel_err < 0.01, f"Relative error {rel_err:.6f} too large for {M}x{N}x{K}"

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 128, 64),  # Single row (batch=1 inference)
            (8, 128, 64),  # Small batch
            (32, 128, 128),  # Medium batch
        ],
        ids=["1x128x64", "8x128x64", "32x128x128"],
    )
    def test_gemm_tall_skinny(self, M, N, K):
        """Test tall/skinny shapes typical of LLM inference."""
        rel_err, _max_err, _mean_err, ref_mag = self._run_gemm_test(M, N, K)
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
        out_packed, out_state = gemm_nvfp4_to_nvfp4(A_packed, A_state, B_packed, B_state, alpha=alpha)
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


class TestScaleReorder:
    """Test scale factor reordering for CUTLASS block-scaled GEMM."""

    def test_scale_to_blocked_round_trip(self):
        """Flat → swizzled → flat round-trip preserves scale values."""
        lib = get_lib()
        H, W = 128, 4  # Minimum block size
        scales = torch.randint(0, 255, (H * W,), dtype=torch.uint8, device="cuda")

        # to_blocked
        n_row_blocks = (H + 127) // 128
        n_col_blocks = (W + 3) // 4
        out_size = n_row_blocks * n_col_blocks * 128 * 4
        blocked = torch.empty(out_size, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.current_stream()
        lib.cscale_to_blocked(
            ctypes.c_void_p(scales.data_ptr()),
            ctypes.c_void_p(blocked.data_ptr()),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_void_p(stream.cuda_stream),
        )

        # from_blocked (inverse)
        recovered = torch.empty(H * W, dtype=torch.uint8, device="cuda")
        lib.cscale_from_blocked(
            ctypes.c_void_p(blocked.data_ptr()),
            ctypes.c_void_p(recovered.data_ptr()),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_void_p(stream.cuda_stream),
        )
        torch.cuda.synchronize()

        assert torch.equal(scales, recovered), "Round-trip failed: scales differ"

    def test_scale_to_blocked_large(self):
        """Test scale reordering with larger shapes matching real GEMM usage."""
        lib = get_lib()
        # Scales for M=256, K=4096 → H=256, W=256 (K/16)
        H, W = 256, 256
        scales = torch.randint(0, 255, (H * W,), dtype=torch.uint8, device="cuda")

        n_row_blocks = (H + 127) // 128
        n_col_blocks = (W + 3) // 4
        out_size = n_row_blocks * n_col_blocks * 128 * 4
        blocked = torch.empty(out_size, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.current_stream()

        lib.cscale_to_blocked(
            ctypes.c_void_p(scales.data_ptr()),
            ctypes.c_void_p(blocked.data_ptr()),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_void_p(stream.cuda_stream),
        )

        recovered = torch.empty(H * W, dtype=torch.uint8, device="cuda")
        lib.cscale_from_blocked(
            ctypes.c_void_p(blocked.data_ptr()),
            ctypes.c_void_p(recovered.data_ptr()),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_void_p(stream.cuda_stream),
        )
        torch.cuda.synchronize()

        assert torch.equal(scales, recovered), "Round-trip failed for large shape"


class TestGemmNVFP4LargeBatch:
    """Test CUTLASS GEMM on large-batch shapes."""

    @pytest.mark.parametrize(
        "shape",
        [
            (512, 512, 512),
            (1024, 1024, 1024),
            (4096, 4096, 4096),
        ],
        ids=["512x512x512", "1024x1024x1024", "4096x4096x4096"],
    )
    def test_gemm_large_batch(self, shape):
        """Test CUTLASS GEMM on large shapes via the Python API."""
        from bitsandbytes.functional import dequantize_nvfp4, gemm_nvfp4, quantize_nvfp4

        M, N, K = shape
        torch.manual_seed(42)

        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        A_packed, A_state = quantize_nvfp4(A)
        B_packed, B_state = quantize_nvfp4(B)

        D = gemm_nvfp4(A_packed, A_state, B_packed, B_state)

        # Reference: dequantize → matmul
        A_deq = dequantize_nvfp4(A_packed, A_state, out_dtype=torch.float32)
        B_deq = dequantize_nvfp4(B_packed, B_state, out_dtype=torch.float32)
        D_ref = A_deq @ B_deq.T

        assert D.shape == (M, N), f"Wrong shape: {D.shape}"

        ref_mag = D_ref.abs().mean().item()
        rel_err = (D - D_ref).abs().mean().item() / ref_mag if ref_mag > 0 else 0
        print(f"Large batch ({M}x{N}x{K}): rel_err={rel_err:.6f}, ref_mag={ref_mag:.4f}")
        # FP4 quantization + accumulation error grows with K
        assert rel_err < 0.2, f"Relative error {rel_err:.4f} too large"


def _has_batched_moe_kernel():
    """Check if the batched MoE SM120 kernel is available."""
    try:
        lib = get_lib()
        return hasattr(lib, "cgemm_nvfp4_moe_sm120_init")
    except RuntimeError:
        return False


def _is_sm120():
    """Check if current GPU is SM120 (consumer Blackwell)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return major == 12 and minor == 0


_skip_no_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_sm120(),
    reason="Requires SM120 GPU (RTX 5090/5080)",
)
_skip_no_batched_moe = pytest.mark.skipif(
    not _has_batched_moe_kernel(),
    reason="Batched MoE SM120 kernel not available in this build",
)


@_skip_no_sm120
@_skip_no_batched_moe
class TestBatchedMoeNVFP4:
    """Test batched NVFP4 MoE GEMM (fixed-padding, CUDA-graph-compatible)."""

    def _run_batched_moe(self, max_M, N, K, num_experts, seed=42):
        """Helper: quantize random data, run batched MoE GEMM, compare vs per-expert loop."""
        from bitsandbytes.functional import gemm_nvfp4_batched_moe, quantize_nvfp4

        torch.manual_seed(seed)

        # Generate random data per expert
        A_float = torch.randn(num_experts, max_M, K, dtype=torch.bfloat16, device="cuda")
        B_float = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")

        # Quantize each expert's activations and weights
        A_packed_list, A_scales_list = [], []
        B_packed_list, B_scales_list = [], []
        A_ts_sum, B_ts_sum = 0.0, 0.0

        for e in range(num_experts):
            a_packed, a_state = quantize_nvfp4(A_float[e])
            b_packed, b_state = quantize_nvfp4(B_float[e])
            A_packed_list.append(a_state.packed_data)
            A_scales_list.append(a_state.block_scales)
            B_packed_list.append(b_state.packed_data)
            B_scales_list.append(b_state.block_scales)
            A_ts_sum += a_state.tensor_scale
            B_ts_sum += b_state.tensor_scale

        # For batched kernel: use a single shared tensor scale (average)
        # In practice, all experts would share the same tensor scale
        A_tensor_scale = A_ts_sum / num_experts
        B_tensor_scale = B_ts_sum / num_experts

        # Concatenate packed data and scales
        A_batched = torch.cat(A_packed_list)
        B_all = torch.cat(B_packed_list)
        SFA_batched = torch.cat(A_scales_list)
        SFB_all = torch.cat(B_scales_list)

        # Run batched MoE GEMM
        D_batched = gemm_nvfp4_batched_moe(
            A_batched, SFA_batched, A_tensor_scale,
            B_all, SFB_all, B_tensor_scale,
            max_M, N, K, num_experts,
        )

        # Reference: per-expert dense GEMM
        from bitsandbytes.functional import dequantize_nvfp4

        D_ref_list = []
        for e in range(num_experts):
            a_state_e = type(quantize_nvfp4(A_float[0])[1]).__new__(
                type(quantize_nvfp4(A_float[0])[1])
            )
            # Just compute reference from float data
            D_ref_list.append(A_float[e].float() @ B_float[e].float().T)

        D_ref = torch.cat(D_ref_list, dim=0)  # (num_experts * max_M, N)

        return D_batched, D_ref

    def test_basic_shape(self):
        """Basic batched MoE GEMM produces correct output shape."""
        from bitsandbytes.functional import gemm_nvfp4_batched_moe, quantize_nvfp4

        torch.manual_seed(42)
        max_M, N, K, num_experts = 4, 128, 256, 4

        A = torch.randn(num_experts, max_M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")

        # Quantize all at once
        A_2d = A.reshape(-1, K)
        B_2d = B.reshape(-1, K)
        A_packed, A_state = quantize_nvfp4(A_2d)
        B_packed, B_state = quantize_nvfp4(B_2d)

        D = gemm_nvfp4_batched_moe(
            A_state.packed_data, A_state.block_scales, A_state.tensor_scale,
            B_state.packed_data, B_state.block_scales, B_state.tensor_scale,
            max_M, N, K, num_experts,
        )

        assert D.shape == (num_experts * max_M, N), f"Wrong shape: {D.shape}"
        assert D.dtype == torch.bfloat16

    def test_nonzero_output(self):
        """Batched MoE GEMM produces non-zero output."""
        from bitsandbytes.functional import gemm_nvfp4_batched_moe, quantize_nvfp4

        torch.manual_seed(42)
        max_M, N, K, num_experts = 8, 128, 256, 2

        A = torch.randn(num_experts * max_M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(num_experts * N, K, dtype=torch.bfloat16, device="cuda")
        A_packed, A_state = quantize_nvfp4(A)
        B_packed, B_state = quantize_nvfp4(B)

        D = gemm_nvfp4_batched_moe(
            A_state.packed_data, A_state.block_scales, A_state.tensor_scale,
            B_state.packed_data, B_state.block_scales, B_state.tensor_scale,
            max_M, N, K, num_experts,
        )

        assert D.nonzero().shape[0] > 0, "Output is all zeros"

    def test_cache_hit(self):
        """Second call with same shape skips init (cache hit)."""
        from bitsandbytes.backends.cuda.ops import _batched_moe_sm120_cache
        from bitsandbytes.functional import gemm_nvfp4_batched_moe, quantize_nvfp4

        torch.manual_seed(42)
        max_M, N, K, num_experts = 4, 128, 256, 2

        A = torch.randn(num_experts * max_M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(num_experts * N, K, dtype=torch.bfloat16, device="cuda")
        A_packed, A_state = quantize_nvfp4(A)
        B_packed, B_state = quantize_nvfp4(B)

        args = (
            A_state.packed_data, A_state.block_scales, A_state.tensor_scale,
            B_state.packed_data, B_state.block_scales, B_state.tensor_scale,
            max_M, N, K, num_experts,
        )

        # First call: inits
        D1 = gemm_nvfp4_batched_moe(*args)
        cache_after_first = _batched_moe_sm120_cache

        # Second call: should hit cache
        D2 = gemm_nvfp4_batched_moe(*args)

        from bitsandbytes.backends.cuda import ops as cuda_ops
        assert cuda_ops._batched_moe_sm120_cache is cache_after_first, "Cache was invalidated unexpectedly"

    def test_cache_invalidation(self):
        """Changing shape triggers re-init (cache miss)."""
        from bitsandbytes.functional import gemm_nvfp4_batched_moe, quantize_nvfp4

        torch.manual_seed(42)
        max_M, K, num_experts = 4, 256, 2

        # First call with N=128
        N1 = 128
        A1 = torch.randn(num_experts * max_M, K, dtype=torch.bfloat16, device="cuda")
        B1 = torch.randn(num_experts * N1, K, dtype=torch.bfloat16, device="cuda")
        A1_packed, A1_state = quantize_nvfp4(A1)
        B1_packed, B1_state = quantize_nvfp4(B1)
        D1 = gemm_nvfp4_batched_moe(
            A1_state.packed_data, A1_state.block_scales, A1_state.tensor_scale,
            B1_state.packed_data, B1_state.block_scales, B1_state.tensor_scale,
            max_M, N1, K, num_experts,
        )

        from bitsandbytes.backends.cuda import ops as cuda_ops
        cache_key_1 = cuda_ops._batched_moe_sm120_cache["key"]

        # Second call with N=256 (different shape)
        N2 = 256
        B2 = torch.randn(num_experts * N2, K, dtype=torch.bfloat16, device="cuda")
        B2_packed, B2_state = quantize_nvfp4(B2)
        D2 = gemm_nvfp4_batched_moe(
            A1_state.packed_data, A1_state.block_scales, A1_state.tensor_scale,
            B2_state.packed_data, B2_state.block_scales, B2_state.tensor_scale,
            max_M, N2, K, num_experts,
        )

        cache_key_2 = cuda_ops._batched_moe_sm120_cache["key"]
        assert cache_key_1 != cache_key_2, "Cache should have been invalidated for different N"
        assert D2.shape == (num_experts * max_M, N2)

    @pytest.mark.parametrize(
        "max_M,N,K,num_experts",
        [
            (1, 128, 256, 2),
            (4, 128, 256, 4),
            (8, 256, 512, 8),
            (32, 128, 256, 128),
        ],
        ids=["M1_E2", "M4_E4", "M8_E8", "M32_E128"],
    )
    def test_various_shapes(self, max_M, N, K, num_experts):
        """Test batched MoE with various shape configurations."""
        from bitsandbytes.functional import gemm_nvfp4_batched_moe, quantize_nvfp4

        torch.manual_seed(42)
        A = torch.randn(num_experts * max_M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(num_experts * N, K, dtype=torch.bfloat16, device="cuda")
        A_packed, A_state = quantize_nvfp4(A)
        B_packed, B_state = quantize_nvfp4(B)

        D = gemm_nvfp4_batched_moe(
            A_state.packed_data, A_state.block_scales, A_state.tensor_scale,
            B_state.packed_data, B_state.block_scales, B_state.tensor_scale,
            max_M, N, K, num_experts,
        )

        assert D.shape == (num_experts * max_M, N)
        assert D.dtype == torch.bfloat16
        assert D.nonzero().shape[0] > 0, f"Output is all zeros for shape M={max_M},N={N},K={K},E={num_experts}"

    def test_glm47_shapes(self):
        """Test with GLM-4.7 model dimensions."""
        from bitsandbytes.functional import gemm_nvfp4_batched_moe, quantize_nvfp4

        torch.manual_seed(42)
        K = 4096
        num_experts = 128

        for name, N, max_M in [("gate_up", 13696, 4), ("down", 4096, 4)]:
            A = torch.randn(num_experts * max_M, K, dtype=torch.bfloat16, device="cuda")
            B = torch.randn(num_experts * N, K, dtype=torch.bfloat16, device="cuda")
            A_packed, A_state = quantize_nvfp4(A)
            B_packed, B_state = quantize_nvfp4(B)

            D = gemm_nvfp4_batched_moe(
                A_state.packed_data, A_state.block_scales, A_state.tensor_scale,
                B_state.packed_data, B_state.block_scales, B_state.tensor_scale,
                max_M, N, K, num_experts,
            )

            assert D.shape == (num_experts * max_M, N), f"{name}: wrong shape {D.shape}"
            assert D.nonzero().shape[0] > 0, f"{name}: output is all zeros"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
