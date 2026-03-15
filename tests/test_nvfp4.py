"""Tests for NVFP4 (E2M1) dequantization kernel.

Tests the NVFP4 dequantize kernel via ctypes calls to the C library.
The quantize path uses the CUTLASS fused kernel (tested in test_fused_quantize.py).
"""

import ctypes
import os

import pytest
import torch


def get_lib():
    """Load the bitsandbytes CUDA library."""
    lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bitsandbytes")
    # Try cuda131 first (built from nvcc 13.1), fall back to cuda130
    for suffix in ["cuda131", "cuda130"]:
        lib_path = os.path.join(lib_dir, f"libbitsandbytes_{suffix}.so")
        if os.path.exists(lib_path):
            return ctypes.cdll.LoadLibrary(lib_path)
    raise RuntimeError(f"Could not find bitsandbytes CUDA library in {lib_dir}")


def dequantize_nvfp4(packed, block_scales, tensor_scale, n, dtype=torch.float16):
    """Dequantize NVFP4 packed data back to FP16/BF16/FP32."""
    lib = get_lib()
    output = torch.zeros(n, dtype=dtype, device=packed.device)

    if dtype == torch.float16:
        func = lib.cdequantize_nvfp4_fp16
    elif dtype == torch.bfloat16:
        func = lib.cdequantize_nvfp4_bf16
    elif dtype == torch.float32:
        func = lib.cdequantize_nvfp4_fp32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    func(
        ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(block_scales.data_ptr()),
        ctypes.c_float(tensor_scale),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(n),
        ctypes.c_void_p(0),  # default stream
    )
    torch.cuda.synchronize()
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestNVFP4Dequant:
    """Test the E2M1 dequantization kernel with known packed values."""

    def test_dequant_known_codes(self):
        """Verify dequantize produces correct values for known E2M1 codes."""
        # E2M1 magnitude table: code -> value
        # 0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
        # Sign bit is bit 3: code 8-15 are negative versions of 0-7

        # Pack 16 values (one block): codes 0-7 positive, then 8-15 negative
        # Each byte holds two 4-bit codes: low nibble first
        packed_bytes = [
            0x10,  # codes 0, 1 -> 0.0, 0.5
            0x32,  # codes 2, 3 -> 1.0, 1.5
            0x54,  # codes 4, 5 -> 2.0, 3.0
            0x76,  # codes 6, 7 -> 4.0, 6.0
            0x98,  # codes 8, 9 -> -0.0, -0.5
            0xBA,  # codes 10, 11 -> -1.0, -1.5
            0xDC,  # codes 12, 13 -> -2.0, -3.0
            0xFE,  # codes 14, 15 -> -4.0, -6.0
        ]
        packed = torch.tensor(packed_bytes, dtype=torch.uint8, device="cuda")

        # UE4M3 scale 1.0: exponent=7 (2^0), mantissa=0 -> code = 0x38
        block_scales = torch.tensor([0x38], dtype=torch.uint8, device="cuda")
        tensor_scale = 1.0

        y = dequantize_nvfp4(packed, block_scales, tensor_scale, 16, dtype=torch.float32)
        expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

        for i, (exp, got) in enumerate(zip(expected, y.tolist())):
            assert abs(exp - got) < 0.01, f"Code {i}: expected {exp}, got {got}"

    def test_dequant_with_tensor_scale(self):
        """Verify tensor scale is applied correctly."""
        # All 1.0 values: code 2, packed as 0x22
        packed = torch.tensor([0x22] * 8, dtype=torch.uint8, device="cuda")
        block_scales = torch.tensor([0x38], dtype=torch.uint8, device="cuda")  # scale 1.0
        tensor_scale = 5.0

        y = dequantize_nvfp4(packed, block_scales, tensor_scale, 16, dtype=torch.float32)
        # Each value should be 1.0 * 1.0 (block) * 5.0 (tensor) = 5.0
        assert torch.allclose(y, torch.full((16,), 5.0, device="cuda"), atol=0.1)

    def test_dequant_with_block_scale(self):
        """Verify block scale is applied correctly."""
        # All 1.0 values: code 2, packed as 0x22
        packed = torch.tensor([0x22] * 8, dtype=torch.uint8, device="cuda")
        # UE4M3 for 2.0: exponent=8 (bias=7, 2^1=2), mantissa=0 -> code = 0x40
        block_scales = torch.tensor([0x40], dtype=torch.uint8, device="cuda")
        tensor_scale = 1.0

        y = dequantize_nvfp4(packed, block_scales, tensor_scale, 16, dtype=torch.float32)
        # Each value should be 1.0 * 2.0 (block) * 1.0 (tensor) = 2.0
        assert torch.allclose(y, torch.full((16,), 2.0, device="cuda"), atol=0.1)
