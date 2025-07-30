import torch
import bitsandbytes.functional as F

device = 'cuda'
torch.manual_seed(42)
A = torch.randn((64,), device=device, dtype=torch.float32)
k = 3

print(f'Input tensor A: min={A.min():.4f}, max={A.max():.4f}, mean={A.mean():.4f}')

# Check codebook
code = F.create_linear_map(signed=True, total_bits=k).to(device)
print(f'Codebook for k={k}: {code[:8]}')  # First 2^k values
non_zero_count = (code != 0).sum()
expected_count = 2**k
print(f'Non-zero entries: {non_zero_count}, should be {expected_count}')

# Quantize and check intermediate results
C, S = F.quantize_blockwise_kbit(A, k=k, blocksize=32)
expected_size = (64*k+31)//32 * 4
print(f'Quantized tensor C shape: {C.shape}, expected: {expected_size}')
print(f'Absmax: {S.absmax}')

A_dequant = F.dequantize_blockwise_kbit(C, k=k, quant_state=S)
print(f'Dequantized tensor: min={A_dequant.min():.4f}, max={A_dequant.max():.4f}')

error = (A - A_dequant).abs().mean().item()
print(f'Reconstruction error: {error:.4f}')

# Check some specific values
print('First 8 values:')
for i in range(8):
    print(f'  {i}: orig={A[i]:.4f}, dequant={A_dequant[i]:.4f}, diff={abs(A[i]-A_dequant[i]):.4f}')

# Let's also check if the issue is with the binary search bounds
print(f'\nCodebook analysis for k={k}:')
for i in range(2**k):
    print(f'  code[{i}] = {code[i]:.4f}')