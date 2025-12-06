from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1b-Instruct")
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1b-Instruct", device_map="mps", quantization_config=quantization_config)
print("model.device:", model.device)
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # or whatever entry function you have

import torch
import bitsandbytes as bnb
A = torch.randn(2048, device='mps', dtype=torch.float16)
q, absmax = torch.ops.bitsandbytes.quantize_4bit(A, 64, 'nf4', torch.uint8)
print('q.shape:', q.shape, q.dtype)
print('absmax.shape:', absmax.shape, absmax.dtype)
B = torch.ops.bitsandbytes.dequantize_4bit(q, absmax, 64, 'nf4', A.shape, A.dtype)
print('ok', float((A-B).abs().max()))

# import torch, bitsandbytes as bnb

# torch.manual_seed(0)
# A = torch.randn(256, device="mps", dtype=torch.float16)

# q, absmax = torch.ops.bitsandbytes.quantize_4bit(A, 64, "nf4", torch.uint8)
# B_native = torch.ops.bitsandbytes.dequantize_4bit(q, absmax, 64, "nf4", A.shape, A.dtype)

# # CPU reference (uses the default implementation, then move back to MPS)
# B_ref = torch.ops.bitsandbytes.dequantize_4bit.default(
#     q.cpu(), absmax.cpu(), 64, "nf4", A.shape, A.dtype
# ).to("mps")

# print("A[:8]      ", A[:8].cpu())
# print("B_native[:8]", B_native[:8].cpu())
# print("B_ref[:8]  ", B_ref[:8].cpu())
# print("max |A-B_native|:", float((A - B_native).abs().max()))
# print("max |A-B_ref|   :", float((A - B_ref).abs().max()))

# diff = (B_native - B_ref).cpu()
# print("B_native shape:", B_native.shape)
# print("B_ref shape:", B_ref.shape)
# print("max |B_native - B_ref|:", float(diff.abs().max()))
# print("first 16 diffs:", diff[:16])

# q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_4bit.default(
#     A.cpu(), 64, "nf4", torch.uint8
# )

# print("q identical? ", torch.equal(q.cpu(), q_cpu))
# print("absmax max diff:", float((absmax.cpu() - absmax_cpu).abs().max()))
# print("q_mps[:8]:", q.view(-1)[:8].cpu())
# print("q_cpu[:8]:", q_cpu.view(-1)[:8])
# print("absmax_mps[:4]:", absmax[:4].cpu())
# print("absmax_cpu[:4]:", absmax_cpu[:4])