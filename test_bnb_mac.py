# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1b-Instruct")
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1b-Instruct", device_map="mps", quantization_config=quantization_config, dtype=torch.float16)
# print("model.device:", model.device)
# prompt = "Hello, how are you?"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # or whatever entry function you have

import torch
import bitsandbytes as bnb
from torch.profiler import profile, ProfilerActivity
from torch.mps.profiler import metal_capture

_NF4_QUANT_TABLE = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
    device="xpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cpu",  # Only cpu/xpu use this table for now.
)
_FP4_QUANT_TABLE = torch.tensor(
    [
        0.0000,
        0.0052,
        0.6667,
        1.0000,
        0.3333,
        0.5000,
        0.1667,
        0.2500,
        0.0000,
        -0.0052,
        -0.6667,
        -1.0000,
        -0.3333,
        -0.5000,
        -0.1667,
        -0.2500,
    ],
    dtype=torch.float32,
    device="xpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cpu",  # Only cpu/xpu use this table for now.
)
CODE = {"nf4": _NF4_QUANT_TABLE, "fp4": _FP4_QUANT_TABLE}

def _dequantize_4bit_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: "Sequence[int]",
    dtype: torch.dtype,
) -> torch.Tensor:
    # Enable non uint8 dtype
    if A.dtype != torch.uint8:
        A = A.view(torch.uint8)
    A = A.reshape(-1)
    # Map nf4 to [-1, 1]
    out_dq = torch.empty(A.size(0) * 2, dtype=torch.int32, device=A.device)
    n = out_dq.numel()
    out_dq[1::2] = A & 0xF
    out_dq[::2] = A >> 4
    # code is fp32, cast to dtype to avoid the mismatch issue
    code = CODE[quant_type].to(dtype).to(A.device)
    out_dq = code[out_dq]

    # Apply scales
    if out_dq.numel() != n:
        assert out_dq.numel() == n + 1
        out_dq = torch.narrow(out_dq, 0, 0, n)
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    rem = n % blocksize
    has_rem = rem > 0

    out = torch.empty(shape, dtype=dtype, device=A.device).reshape(-1)
    if has_rem:
        out[: n - rem] = (out_dq[: n - rem].view(-1, blocksize) * absmax[: blocks - has_rem].view(-1, 1)).reshape(-1)
        out[n - rem :] = out_dq[n - rem :] * absmax[-1]
    else:
        out = out_dq.view(-1, blocksize) * absmax.view(-1, 1)

    out = out.reshape(-1, *shape[1:]).to(dtype)

    return out
def run_once():
    # A = torch.randn(2048, device="mps", dtype=torch.float16)
    # q, absmax = torch.ops.bitsandbytes.quantize_4bit(A, 64, "nf4", torch.uint8)
    out = torch.empty(2048*2, device="mps", dtype=torch.float32)
    q = torch.randint(0, 255, (2048,), device="mps", dtype=torch.uint8)
    absmax = torch.randn(64, device="mps", dtype=torch.float32)
    print("q.shape:", q.shape, q.dtype)
    print("absmax.shape:", absmax.shape, absmax.dtype)
    B = torch.ops.bitsandbytes.dequantize_4bit(q, absmax, 64, "nf4", out.shape, out.dtype)
    # B_ref = _dequantize_4bit_impl(q, absmax, 64, "nf4", out.shape, out.dtype)
    # print("ok", float((B - B_ref).abs().max()))
    # torch.mps.synchronize()
    # print("B.shape:", B.shape, B.dtype)
    # print("ok", float((A - B).abs().max()))

run_once()
trace_path = "bnb_mps_capture_11.gputrace"

with metal_capture(trace_path):
    with profile(
        activities=[],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for i in range(10):
            run_once()
            torch.mps.synchronize()
            print(f"iteration {i} done")

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
print(f"Metal capture saved to: {trace_path}")

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

# import torch, bitsandbytes as bnb, time

# torch.manual_seed(0)
# A = torch.randn(4096 * 4096, device="mps", dtype=torch.float16)
# blocksize = 64

# q, absmax = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, "nf4", torch.uint8)

# torch.mps.synchronize()
# t0 = time.perf_counter()
# torch.ops.bitsandbytes.dequantize_4bit(q, absmax, blocksize, "nf4", A.shape, A.dtype)
# torch.mps.synchronize()
# dt = time.perf_counter() - t0
# print(f"Dequant time: {dt*1000:.2f} ms for {A.numel()/1e6:.1f}M elements")