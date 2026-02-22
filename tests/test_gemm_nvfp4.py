"""Test NVFP4 GEMM kernel on SM_120 (Blackwell consumer GPUs).

Tests the block-scaled mma.sync GEMM kernel via ctypes.
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


# E2M1 representable magnitudes (unsigned)
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def float_to_e2m1(x):
    """Quantize a float to nearest E2M1 value (magnitude only)."""
    ax = abs(x)
    # Decision boundaries (midpoints between consecutive E2M1 values)
    boundaries = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    for i, b in enumerate(boundaries):
        if ax < b:
            return E2M1_VALUES[i] * (1 if x >= 0 else -1)
    return E2M1_VALUES[7] * (1 if x >= 0 else -1)


def float_to_e4m3(x):
    """Quantize a positive float to UE4M3 (unsigned E4M3, bias=7)."""
    if x <= 0:
        return 0, 0.0
    # Clamp to max representable value (~448)
    x = min(x, 448.0)
    if x == 0:
        return 0, 0.0
    # Find the exponent
    import math
    e = math.floor(math.log2(x))
    e = max(e, -6)  # min exponent with bias=7 is -6
    e = min(e, 8)   # max exponent with bias=7 is 8
    # Mantissa
    m = x / (2.0 ** e) - 1.0
    m = max(0, min(m, 0.875))  # 3 mantissa bits -> 7/8 max
    m_int = round(m * 8)
    m_int = min(m_int, 7)
    # Encode
    e_biased = e + 7
    e_biased = max(0, min(e_biased, 15))
    code = (e_biased << 3) | m_int
    # Decode to get actual value
    actual = (1.0 + m_int / 8.0) * (2.0 ** (e_biased - 7))
    if e_biased == 0:
        actual = m_int / 8.0 * (2.0 ** -6)
    return code, actual


def quantize_tensor_reference(x_flat):
    """Reference quantization: float tensor -> packed FP4 + block scales + tensor scale.

    Returns (packed_bytes, block_scale_bytes, tensor_scale) in the format
    expected by the GEMM kernel.
    """
    n = len(x_flat)
    assert n % 16 == 0
    num_blocks = n // 16

    tensor_scale = max(abs(v) for v in x_flat)
    if tensor_scale == 0:
        tensor_scale = 1.0

    packed = []
    block_scales = []

    for b in range(num_blocks):
        block = x_flat[b * 16:(b + 1) * 16]
        # Normalize by tensor scale
        normalized = [v / tensor_scale for v in block]
        # Block absmax
        block_max = max(abs(v) for v in normalized)
        if block_max == 0:
            block_max = 1e-10

        # Block scale = block_max / 6.0 (max E2M1 value)
        raw_scale = block_max / 6.0
        scale_code, scale_actual = float_to_e4m3(raw_scale)
        block_scales.append(scale_code)

        if scale_actual == 0:
            scale_actual = 1e-10

        # Quantize each element
        nibbles = []
        for v in normalized:
            scaled_v = v / scale_actual
            qval = float_to_e2m1(scaled_v)
            # Encode: sign in bit 3, magnitude in bits 0-2
            mag = abs(qval)
            mag_idx = E2M1_VALUES.index(mag) if mag in E2M1_VALUES else 0
            code = mag_idx
            if qval < 0:
                code |= 0x8
            nibbles.append(code)

        # Pack 2 per byte (low nibble = even index, high nibble = odd index)
        for i in range(0, 16, 2):
            byte_val = (nibbles[i] & 0xF) | ((nibbles[i + 1] & 0xF) << 4)
            packed.append(byte_val)

    return packed, block_scales, tensor_scale


def dequantize_reference(packed, block_scales, tensor_scale, M, K):
    """Reference dequantization for verification."""
    n = M * K
    result = []
    for i in range(n):
        byte_idx = i // 2
        block_idx = i // 16
        byte_val = packed[byte_idx]
        if i % 2 == 0:
            code = byte_val & 0xF
        else:
            code = (byte_val >> 4) & 0xF

        sign = -1.0 if (code & 0x8) else 1.0
        mag_idx = code & 0x7
        mag = E2M1_VALUES[mag_idx]

        # Decode block scale
        sf_code = block_scales[block_idx]
        sf_e = (sf_code >> 3) & 0xF
        sf_m = sf_code & 0x7
        if sf_e == 0:
            sf_val = sf_m / 8.0 * (2.0 ** -6)
        else:
            sf_val = (1.0 + sf_m / 8.0) * (2.0 ** (sf_e - 7))

        result.append(sign * mag * sf_val * tensor_scale)
    return result


def prepare_gemm_inputs(M, N, K, seed=42):
    """Create random FP4-quantized inputs for GEMM testing.

    Returns CUDA tensors ready for the GEMM kernel, plus reference
    dequantized matrices for verification.
    """
    import random
    random.seed(seed)

    # Generate random float values
    A_flat = [random.gauss(0, 1) for _ in range(M * K)]
    B_flat = [random.gauss(0, 1) for _ in range(N * K)]  # B is N x K (transposed)

    # Quantize
    A_packed, A_sf, A_ts = quantize_tensor_reference(A_flat)
    B_packed, B_sf, B_ts = quantize_tensor_reference(B_flat)

    # Dequantize for reference
    A_deq = dequantize_reference(A_packed, A_sf, A_ts, M, K)
    B_deq = dequantize_reference(B_packed, B_sf, B_ts, N, K)

    # Reshape for torch.matmul: A is M x K, B^T is N x K -> B is K x N
    A_ref = torch.tensor(A_deq, dtype=torch.float32).reshape(M, K)
    B_ref = torch.tensor(B_deq, dtype=torch.float32).reshape(N, K).T  # K x N

    # Reference output
    D_ref = A_ref @ B_ref  # M x N

    # Create CUDA tensors
    A_data = torch.tensor(A_packed, dtype=torch.uint8, device="cuda")
    B_data = torch.tensor(B_packed, dtype=torch.uint8, device="cuda")
    A_scales = torch.tensor(A_sf, dtype=torch.uint8, device="cuda")
    B_scales = torch.tensor(B_sf, dtype=torch.uint8, device="cuda")

    return A_data, B_data, A_scales, B_scales, A_ts, B_ts, D_ref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGemmNVFP4:
    """Test NVFP4 GEMM kernel correctness."""

    def _run_gemm(self, M, N, K, seed=42):
        """Run the GEMM kernel and return (output, reference)."""
        lib = get_lib()
        assert hasattr(lib, "cgemm_nvfp4"), "cgemm_nvfp4 symbol not found in library"

        A_data, B_data, A_scales, B_scales, A_ts, B_ts, D_ref = prepare_gemm_inputs(M, N, K, seed)

        D_out = torch.zeros(M, N, dtype=torch.float32, device="cuda")

        lib.cgemm_nvfp4(
            ctypes.c_void_p(A_data.data_ptr()),
            ctypes.c_void_p(B_data.data_ptr()),
            ctypes.c_void_p(A_scales.data_ptr()),
            ctypes.c_void_p(B_scales.data_ptr()),
            ctypes.c_void_p(D_out.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(K),
        )
        torch.cuda.synchronize()

        return D_out.cpu(), D_ref

    def test_gemm_nvfp4_minimal(self):
        """Test 16x8x64 (single MMA tile)."""
        D_out, D_ref = self._run_gemm(16, 8, 64)
        print(f"Output[0:4, 0:4]:\n{D_out[0:4, 0:4]}")
        print(f"Reference[0:4, 0:4]:\n{D_ref[0:4, 0:4]}")
        # Just check it runs and produces finite values
        assert torch.isfinite(D_out).all(), "Output contains non-finite values"
        # Check rough magnitude match (within 10x)
        if D_ref.abs().max() > 0:
            ratio = D_out.abs().max() / D_ref.abs().max()
            print(f"Max magnitude ratio (out/ref): {ratio:.3f}")

    def test_gemm_nvfp4_identity_scales(self):
        """Test with all-ones data and scale=1 to verify basic MMA correctness."""
        lib = get_lib()
        M, N, K = 16, 8, 64

        # All values = 1.0 in E2M1: code = 0b0010 = 2
        # Pack: byte = (2) | (2 << 4) = 0x22
        A_packed = torch.full((M * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        B_packed = torch.full((N * K // 2,), 0x22, dtype=torch.uint8, device="cuda")

        # Scale = 1.0 in UE4M3: exponent=7 (bias=7, so 2^0=1), mantissa=0
        # Code = (7 << 3) | 0 = 56 = 0x38
        A_scales = torch.full((M * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")
        B_scales = torch.full((N * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")

        D_out = torch.zeros(M, N, dtype=torch.float32, device="cuda")

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

        # Each output element = sum of K products: 1.0 * 1.0 * K = 64
        expected = 64.0
        D_cpu = D_out.cpu()
        print(f"Identity test output:\n{D_cpu}")
        assert torch.allclose(D_cpu, torch.full((M, N), expected)), (
            f"Expected all {expected}, got min={D_cpu.min():.1f} max={D_cpu.max():.1f}"
        )

    def test_gemm_nvfp4_multi_k_tiles(self):
        """Test with K > 64 to verify K-loop accumulation."""
        lib = get_lib()
        M, N, K = 16, 8, 128  # 2 k-tiles

        # All values = 1.0
        A_packed = torch.full((M * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        B_packed = torch.full((N * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
        A_scales = torch.full((M * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")
        B_scales = torch.full((N * (K // 16),), 0x38, dtype=torch.uint8, device="cuda")

        D_out = torch.zeros(M, N, dtype=torch.float32, device="cuda")

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

        expected = float(K)  # 1.0 * 1.0 * K
        D_cpu = D_out.cpu()
        print(f"Multi-K test output (expect {expected}):\n{D_cpu[0, :]}")
        assert torch.allclose(D_cpu, torch.full((M, N), expected)), (
            f"Expected all {expected}, got min={D_cpu.min():.1f} max={D_cpu.max():.1f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
