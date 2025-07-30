import torch
import bitsandbytes.functional as F

k = 3
device = 'cuda'

print("Testing codebook compaction:")

# Original codebook
full_code = F.create_linear_map(signed=True, total_bits=k).to(device)
print(f'Original full codebook:')
print(f'  First 8: {full_code[:8]}')
print(f'  Last 8: {full_code[-8:]}')

# Find non-zero values
non_zero_mask = full_code != 0
non_zero_values = full_code[non_zero_mask]
print(f'Non-zero values: {non_zero_values}')
print(f'Count: {len(non_zero_values)}, expected: {2**k}')

# Manual compaction
code = torch.zeros(256, device=device, dtype=torch.float32)
code[:len(non_zero_values)] = non_zero_values

# Add zero if missing
if len(non_zero_values) < 2**k:
    code[len(non_zero_values)] = 0.0
    print(f'Added zero at position {len(non_zero_values)}')

print(f'Compacted codebook first 8: {code[:8]}')

# Test with actual quantization to see what happens
A = torch.randn((32,), device=device, dtype=torch.float32)
C, S = F.quantize_blockwise_kbit(A, k=k, blocksize=32)
print(f'QuantState codebook first 8: {S.code[:8]}')