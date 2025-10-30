import torch
import bitsandbytes as bnb
A = torch.randn(128, device='mps', dtype=torch.float16)
q, state = bnb.functional.quantize_4bit(A, quant_type='nf4')
A2 = bnb.functional.dequantize_4bit(q, quant_state=state)
print('diff', float((A2 - A).abs().mean().cpu()), flush=True)
