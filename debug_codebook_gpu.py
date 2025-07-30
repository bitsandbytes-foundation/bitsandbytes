import torch
import bitsandbytes.functional as F

# Test with CUDA
print("Testing codebook on GPU:")
k = 3
device = 'cuda'

# Create codebook and move to GPU
code_cpu = F.create_linear_map(signed=True, total_bits=k)
code_gpu = code_cpu.to(device)

print(f'CPU codebook first 8 values: {code_cpu[:8]}')
print(f'GPU codebook first 8 values: {code_gpu[:8]}')

# The issue might be that the codebook creation is correct,
# but my quantization kernel is only looking at indices 0 to 2^k-1
# Let's check what's at those indices
print(f'\nIndices 0 to {2**k-1} (what kernel should see):')
for i in range(2**k):
    if i < len(code_gpu):
        print(f'  code[{i}] = {code_gpu[i]:.6f}')
    else:
        print(f'  code[{i}] = OUT_OF_BOUNDS')

# Find where the actual values are
non_zero_indices = torch.nonzero(code_gpu, as_tuple=False).flatten()
print(f'\nActual non-zero indices: {non_zero_indices}')
print(f'Actual non-zero values: {code_gpu[non_zero_indices]}')

# The problem is clear now: for k=3, the kernel looks at indices 0-7,
# but the values are spread across indices [0,1,2,252,253,254,255]
# So indices 3,4,5,6,7 are all zeros!