import argparse

import torch
import torch.utils.benchmark as benchmark
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name", default="meta-llama/Llama-3.1-8B-Instruct", required=False, type=str, help="model_name"
)
parser.add_argument("--quant_type", default="int8", type=str, help="quant type", choices=["int8", "nf4", "fp4"])
parser.add_argument("--device_map", default="cpu", type=str, help="device_map", choices=["cpu", "xpu", "cuda"])
args = parser.parse_args()

model_name = args.model_name
device_map = args.device_map
if args.quant_type == "int8":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map=device_map, quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to(quantized_model.device)

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))


# benchmark the performance
def benchmark_fn(f, *args, **kwargs):
    # Manual warmup
    for _ in range(2):
        f(*args, **kwargs)

    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return t0.blocked_autorange().mean


MAX_NEW_TOKENS = 100

quantized_model_latency = benchmark_fn(quantized_model.generate, **input_ids, max_new_tokens=MAX_NEW_TOKENS)

bf16_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.bfloat16)
bf16_model_latency = benchmark_fn(bf16_model.generate, **input_ids, max_new_tokens=MAX_NEW_TOKENS)

print(f"bnb model latency: {quantized_model_latency:.3f}")
print(f"bf16 model latency: {bf16_model_latency:.3f}")
print(f"BNB vs. bf16 model speed-up: {(bf16_model_latency / quantized_model_latency):.3f}")

print(f"BNB model memory: {(quantized_model.get_memory_footprint() / 1024 / 1024 / 1024):.3f} GB")
print(f"bf16 model memory: {(bf16_model.get_memory_footprint() / 1024 / 1024 / 1024):.3f} GB")
print(
    f"BNB vs. bf16 model memory ratio: {(bf16_model.get_memory_footprint() / quantized_model.get_memory_footprint()):.3f}"
)
