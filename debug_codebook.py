import torch
import bitsandbytes.functional as F

print("Analyzing codebook creation for different k values:")

for k in [2, 3, 4, 5, 6, 7, 8]:
    code = F.create_linear_map(signed=True, total_bits=k)
    non_zero_indices = torch.nonzero(code, as_tuple=False).flatten()
    non_zero_values = code[non_zero_indices]
    
    print(f'\nk={k} (should have {2**k} values):')
    print(f'  Non-zero indices: {non_zero_indices.tolist()[:10]}')  # First 10
    print(f'  Non-zero values: {non_zero_values.tolist()[:10]}')   # First 10
    print(f'  Count: {len(non_zero_indices)} (expected: {2**k})')
    print(f'  Range: [{non_zero_values.min():.4f}, {non_zero_values.max():.4f}]')
    
    if len(non_zero_indices) != 2**k:
        print(f'  ❌ MISMATCH: Expected {2**k} non-zero values, got {len(non_zero_indices)}')
    else:
        print(f'  ✅ Correct count')