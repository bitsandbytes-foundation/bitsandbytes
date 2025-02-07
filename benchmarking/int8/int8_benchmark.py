"""
Basic benchmark for text generation.

Usage: python benchmarking/int8/int8_benchmark.py
"""

import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MAX_NEW_TOKENS = 128
model_name = "meta-llama/Llama-3.1-8B"

text = "Below is a question. I need an answer.\n\nExplain machine learning: "
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer([text] * 8, return_tensors="pt").input_ids.to(0)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    ),
    attn_implementation="sdpa",
    torch_dtype=torch.float16,
)

print(model)

# warmup
print("Warmup...")
for i in range(3):
    generated_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)

print("Profiler starting...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_modules=True,
    with_stack=True,
) as prof:
    model.generate(input_ids, max_new_tokens=1)

print(
    prof.key_averages().table(
        sort_by="cpu_time_total",
        max_name_column_width=50,
        top_level_events_only=True,
        row_limit=50,
    )
)

torch.cuda.synchronize()


print("Generating...")
num = 0
time_1 = time.time()
for i in range(5):
    generated_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
    num += len(generated_ids[0])

print("=" * 40)
print(f"Example:\n{tokenizer.decode(generated_ids[0])}")
print("=" * 40)
print(f"Speed: {num/(time.time() - time_1)}token/s")
