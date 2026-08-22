
import pytest
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import get_available_devices

@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
@pytest.mark.parametrize("blocksize", [64, 128])
def test_dequantize_4bit_1d_shape(device, dtype, quant_type, blocksize):
    """
    Regression test for issue #2047: 
    CPU dequantize_4bit returns shape (1, n) for even-length 1-D inputs.
    """
    if device == "hpu":
        pytest.skip("Skipping on HPU")
        
    input_size = 256
    shape = (input_size,)
    
    # Create dummy quantized data
    n = input_size
    blocks = -(n // -blocksize)
    
    # 4-bit packed data
    A = torch.randint(0, 255, (n // 2,), dtype=torch.uint8, device=device)
    absmax = torch.randn((blocks,), dtype=torch.float32, device=device)
    
    # Call dequantize_4bit through the torch op (which calls our kernel)
    out = torch.ops.bitsandbytes.dequantize_4bit.default(
        A, absmax, blocksize, quant_type, shape, dtype
    )
    
    assert out.shape == shape, f"Expected shape {shape}, got {out.shape} on {device}"
    assert out.dtype == dtype
    assert out.device.type == torch.device(device).type
