import torch
import pytest
import bitsandbytes.functional as F


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_kbit_roundtrip_basic(k, dtype):
    """Test basic round-trip quantization/dequantization for k-bit."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = 'cuda'
    blocksize = 32  # Only supported blocksize
    shape = (128,)  # Simple 1D tensor
    
    # Create test tensor
    torch.manual_seed(42)
    A = torch.randn(shape, device=device, dtype=dtype)
    
    # Test k-bit quantization/dequantization
    C_kbit, S_kbit = F.quantize_blockwise_kbit(A, k=k, blocksize=blocksize)
    A_dequant = F.dequantize_blockwise_kbit(C_kbit, k=k, quant_state=S_kbit)
    
    # Shape checks
    assert A_dequant.shape == A.shape, f"Shape mismatch: {A_dequant.shape} vs {A.shape}"
    assert A_dequant.dtype == A.dtype, f"Dtype mismatch: {A_dequant.dtype} vs {A.dtype}"
    
    # Quantized tensor should be smaller (packed)
    elements_per_word = 32 // k
    expected_packed_words = (A.numel() + elements_per_word - 1) // elements_per_word
    expected_packed_bytes = expected_packed_words * 4
    assert C_kbit.numel() == expected_packed_bytes, f"Packed size mismatch: {C_kbit.numel()} vs {expected_packed_bytes}"
    
    # Error should be reasonable for k-bit quantization
    error = (A - A_dequant).abs().mean().item()
    max_expected_error = 0.5 / (2**(k-1))  # Rough estimate based on bit width
    assert error < max_expected_error * 10, f"Error too high for {k}-bit: {error} > {max_expected_error * 10}"


@pytest.mark.parametrize("shape", [(96,), (64, 32), (8, 8, 16)])
def test_kbit_shapes(shape):
    """Test k-bit quantization with different tensor shapes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = 'cuda'
    k = 4  # 4-bit quantization
    blocksize = 32
    
    # Create test tensor
    torch.manual_seed(42)
    A = torch.randn(shape, device=device, dtype=torch.float32)
    
    # Test quantization/dequantization
    C_kbit, S_kbit = F.quantize_blockwise_kbit(A, k=k, blocksize=blocksize)
    A_dequant = F.dequantize_blockwise_kbit(C_kbit, k=k, quant_state=S_kbit)
    
    # Shape should be preserved
    assert A_dequant.shape == shape, f"Shape not preserved: {A_dequant.shape} vs {shape}"
    
    # Quantization state should store original shape
    assert S_kbit.shape == shape, f"QuantState shape mismatch: {S_kbit.shape} vs {shape}"


def test_kbit_vs_linear_map():
    """Compare k-bit quantization with linear map quantization for correctness."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = 'cuda'
    k = 4
    blocksize = 32
    shape = (128,)
    
    # Create test tensor with known range
    torch.manual_seed(42)
    A = torch.randn(shape, device=device, dtype=torch.float32) * 0.5  # Smaller range for better quantization
    
    # Test k-bit quantization with linear map
    C_kbit, S_kbit = F.quantize_blockwise_kbit(A, k=k, blocksize=blocksize)
    A_kbit_dequant = F.dequantize_blockwise_kbit(C_kbit, k=k, quant_state=S_kbit)
    
    # Test regular blockwise quantization with same linear map
    code = F.create_linear_map(signed=True, total_bits=k).to(device)
    C_regular, S_regular = F.quantize_blockwise(A, blocksize=blocksize, code=code)
    A_regular_dequant = F.dequantize_blockwise(C_regular, S_regular)
    
    # Both should have similar error characteristics
    error_kbit = (A - A_kbit_dequant).abs().mean().item()
    error_regular = (A - A_regular_dequant).abs().mean().item()
    
    # K-bit should be competitive (within 2x error)
    assert error_kbit < error_regular * 2, f"K-bit error much higher: {error_kbit} vs {error_regular}"


def test_kbit_blocksize_validation():
    """Test that only blocksize=32 is supported."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = 'cuda'
    A = torch.randn((64,), device=device, dtype=torch.float32)
    
    # blocksize=32 should work
    try:
        C, S = F.quantize_blockwise_kbit(A, k=4, blocksize=32)
        A_dequant = F.dequantize_blockwise_kbit(C, k=4, quant_state=S, blocksize=32)
        assert A_dequant.shape == A.shape
    except Exception as e:
        pytest.fail(f"blocksize=32 should work: {e}")
    
    # Other blocksizes should raise NotImplementedError
    for invalid_blocksize in [64, 128, 256, 512, 1024, 2048, 4096]:
        with pytest.raises(NotImplementedError):
            F.quantize_blockwise_kbit(A, k=4, blocksize=invalid_blocksize)


def test_kbit_bit_packing():
    """Test that bit packing produces expected output sizes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = 'cuda'
    
    test_cases = [
        # (k, input_elements, expected_packed_words)
        (2, 64, 4),   # 32/2=16 elements per word, 64/16=4 words
        (3, 60, 6),   # 32/3=10 elements per word, 60/10=6 words
        (4, 64, 8),   # 32/4=8 elements per word, 64/8=8 words
        (5, 60, 10),  # 32/5=6 elements per word, 60/6=10 words
        (8, 64, 16),  # 32/8=4 elements per word, 64/4=16 words
    ]
    
    for k, n_elements, expected_words in test_cases:
        A = torch.randn((n_elements,), device=device, dtype=torch.float32)
        C, S = F.quantize_blockwise_kbit(A, k=k, blocksize=32)
        
        expected_bytes = expected_words * 4  # uint32 = 4 bytes
        assert C.numel() == expected_bytes, f"k={k}, n={n_elements}: expected {expected_bytes} bytes, got {C.numel()}"


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7, 8])
def test_kbit_error_bounds(k):
    """Test that quantization error is within reasonable bounds for each k."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = 'cuda'
    blocksize = 32
    
    # Test with different ranges to see quantization behavior
    for scale in [0.1, 0.5, 1.0, 2.0]:
        torch.manual_seed(42)
        A = torch.randn((96,), device=device, dtype=torch.float32) * scale
        
        C_kbit, S_kbit = F.quantize_blockwise_kbit(A, k=k, blocksize=blocksize)
        A_dequant = F.dequantize_blockwise_kbit(C_kbit, k=k, quant_state=S_kbit)
        
        # Calculate relative error
        rel_error = ((A - A_dequant).abs() / (A.abs() + 1e-8)).mean().item()
        
        # Error bounds based on bit width (rough estimates)
        max_rel_error = {
            2: 0.4,   # 2-bit is very coarse
            3: 0.2,   # 3-bit
            4: 0.1,   # 4-bit
            5: 0.05,  # 5-bit
            6: 0.03,  # 6-bit
            7: 0.02,  # 7-bit
            8: 0.01   # 8-bit
        }
        
        assert rel_error < max_rel_error[k], f"k={k}, scale={scale}: rel_error {rel_error} > {max_rel_error[k]}"


def test_kbit_cpu_not_implemented():
    """Test that CPU backend raises NotImplementedError."""
    A = torch.randn((32,), dtype=torch.float32)  # CPU tensor
    
    with pytest.raises(NotImplementedError, match="K-bit quantization is not implemented for CPU backend"):
        F.quantize_blockwise_kbit(A, k=4, blocksize=32)


if __name__ == "__main__":
    # Run basic tests if executed directly
    if torch.cuda.is_available():
        print("Running basic k-bit quantization tests...")
        test_kbit_roundtrip_basic(4, torch.float32)
        test_kbit_shapes((64, 32))
        test_kbit_blocksize_validation()
        test_kbit_bit_packing()
        print("✓ All basic tests passed!")
    else:
        print("CUDA not available, skipping tests")