import torch
import bitsandbytes.functional as F
import numpy as np

def test_kbit_quantization_correctness():
    """
    Test if k-bit quantization binary search is working correctly by:
    1. Comparing with reference 8-bit quantization using same codebook values
    2. Manually unpacking k-bit values to verify they represent same choices
    """
    
    device = 'cuda'
    k = 3  # Test with 3-bit first
    ref_blocksize = 64  # For reference quantization (must be supported)
    kbit_blocksize = 32  # For k-bit quantization (only 32 supported)
    
    # Create a tensor where every 32 values have the same absolute maximum
    # This makes blocksize 32 and 64 produce the same scaling factors
    base_values = torch.tensor([-0.8, -0.4, 0.0, 0.4, 0.8], device=device)  # Simple pattern
    
    # Create 64 elements: two groups of 32 with same absmax in each group
    group1 = base_values.repeat(7)[:32]  # First 32 elements
    group2 = base_values.repeat(7)[:32]  # Second 32 elements (identical pattern)
    A = torch.cat([group1, group2])
    
    print(f"Created test tensor with repeated pattern:")
    print(f"  First 8: {A[:8].tolist()}")
    print(f"  Last 8: {A[-8:].tolist()}")
    print(f"  Absmax group 1 (0:32): {A[:32].abs().max():.4f}")
    print(f"  Absmax group 2 (32:64): {A[32:].abs().max():.4f}")
    print(f"  Overall: min={A.min():.4f}, max={A.max():.4f}")
    
    # Step 1: Get reference quantization with original (scattered) codebook
    original_codebook = F.create_linear_map(signed=True, total_bits=k).to(device)
    print(f"\nOriginal codebook non-zero indices: {torch.nonzero(original_codebook != 0).flatten()}")
    print(f"Original codebook non-zero values: {original_codebook[original_codebook != 0]}")
    
    # Reference quantization (this should work correctly)
    C_ref, S_ref = F.quantize_blockwise(A, blocksize=ref_blocksize, code=original_codebook)
    
    # Step 2: Get my k-bit quantization with compacted codebook  
    C_kbit, S_kbit = F.quantize_blockwise_kbit(A, k=k, blocksize=kbit_blocksize)
    
    print(f"\nCompacted codebook first 8 values: {S_kbit.code[:8]}")
    
    # Step 3: Create mapping between original and compacted codebook indices
    # Find where each compacted value appears in the original codebook
    original_nonzero_indices = torch.nonzero(original_codebook != 0).flatten()
    compacted_to_original_map = {}
    
    for compact_idx in range(len(S_kbit.code[:2**k])):
        compact_value = S_kbit.code[compact_idx]
        # Find this value in original codebook
        matches = torch.where(torch.abs(original_codebook - compact_value) < 1e-6)[0]
        if len(matches) > 0:
            compacted_to_original_map[compact_idx] = matches[0].item()
            print(f"  Compact[{compact_idx}]={compact_value:.4f} -> Original[{matches[0].item()}]")
    
    # Step 4: Manually unpack k-bit values from my quantized tensor
    print(f"\nManual unpacking of k-bit tensor:")
    print(f"Packed tensor shape: {C_kbit.shape} (should be {(64*k+31)//32 * 4} bytes)")
    
    # Convert packed bytes to uint32 words
    packed_uint32 = C_kbit.reshape(-1, 4)  # Group into uint32 words
    packed_words = []
    for i in range(packed_uint32.shape[0]):
        # Reconstruct uint32 from 4 uint8 bytes (little endian)
        word = (packed_uint32[i, 0].item() + 
                (packed_uint32[i, 1].item() << 8) + 
                (packed_uint32[i, 2].item() << 16) + 
                (packed_uint32[i, 3].item() << 24))
        packed_words.append(word)
    
    print(f"Reconstructed {len(packed_words)} uint32 words")
    
    # Extract k-bit values manually
    elements_per_word = 32 // k
    extracted_indices = []
    
    for word_idx, packed_word in enumerate(packed_words):
        for bit_pos in range(elements_per_word):
            element_idx = word_idx * elements_per_word + bit_pos
            if element_idx < A.numel():
                # Extract k bits starting at position (bit_pos * k)
                bit_offset = bit_pos * k
                mask = (1 << k) - 1  # k-bit mask
                extracted_idx = (packed_word >> bit_offset) & mask
                extracted_indices.append(extracted_idx)
    
    print(f"Extracted {len(extracted_indices)} k-bit indices")
    print(f"First 10 extracted indices: {extracted_indices[:10]}")
    
    # Step 5: Map extracted indices back to original codebook space for comparison
    mapped_indices = []
    for compact_idx in extracted_indices:
        if compact_idx in compacted_to_original_map:
            mapped_indices.append(compacted_to_original_map[compact_idx])
        else:
            mapped_indices.append(-1)  # Invalid index
    
    print(f"First 10 mapped indices: {mapped_indices[:10]}")
    print(f"First 10 reference indices: {C_ref[:10].tolist()}")
    
    # Step 6: Compare if they represent the same quantization choices
    matches = 0
    mismatches = 0
    
    print(f"\nDetailed comparison (all 64 elements):")
    for i in range(min(64, len(mapped_indices))):
        ref_idx = C_ref[i].item()
        my_mapped_idx = mapped_indices[i]
        ref_value = f"{original_codebook[ref_idx].item():.3f}" if ref_idx < len(original_codebook) else "OOB"
        my_value = f"{original_codebook[my_mapped_idx].item():.3f}" if my_mapped_idx >= 0 and my_mapped_idx < len(original_codebook) else "INVALID"
        
        match = (ref_idx == my_mapped_idx)
        if match:
            matches += 1
        else:
            mismatches += 1
            
        print(f"  [{i:2d}]: ref_idx={ref_idx:3d} ({ref_value:>6s}) vs my_idx={my_mapped_idx:3d} ({my_value:>6s}) {'✓' if match else '✗'}")
    
    total_compared = min(len(mapped_indices), A.numel())
    accuracy = matches / total_compared if total_compared > 0 else 0
    
    print(f"\nQuantization accuracy: {matches}/{total_compared} = {accuracy:.3f}")
    
    if accuracy > 0.95:
        print("✅ Binary search appears to be working correctly!")
    else:
        print("❌ Binary search has issues - quantization choices don't match reference")
        
    return accuracy

if __name__ == "__main__":
    accuracy = test_kbit_quantization_correctness()