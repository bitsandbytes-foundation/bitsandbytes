"""
Benchmark: Paged vs Non-Paged Optimizer GPU Memory Usage.

Demonstrates that paged optimizers significantly reduce GPU memory consumption
by storing optimizer states in CPU/GPU shared memory (USM) instead of pure GPU memory.

Usage:
    python benchmark_paged_memory.py
    python benchmark_paged_memory.py --hidden_size 2048 --num_layers 16
    python benchmark_paged_memory.py --device cuda  # also works on CUDA
"""

import argparse
import gc

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import bitsandbytes as bnb


def get_args():
    parser = argparse.ArgumentParser(description="Paged Optimizer Memory Benchmark")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=2752)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="xpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def get_torch_dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def get_accelerator(device_type):
    """Return the torch accelerator module (torch.cuda / torch.xpu)."""
    if device_type == "xpu":
        return torch.xpu
    return torch.cuda


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def create_model(args):
    """Create a LLaMA model from config (no download needed)."""
    config = LlamaConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.seq_len * 2,
    )
    dtype = get_torch_dtype(args.dtype)
    model = LlamaForCausalLM(config).to(dtype=dtype, device=args.device)
    return model


def make_batch(args):
    """Create a random batch of input_ids and labels."""
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=args.device)
    labels = input_ids.clone()
    return input_ids, labels


def cleanup(device_type):
    """Force cleanup of GPU memory."""
    gc.collect()
    acc = get_accelerator(device_type)
    acc.empty_cache()
    acc.synchronize()


def measure_training(args, optimizer_name, OptClass):
    """Run a few training steps and return peak GPU memory in bytes."""
    acc = get_accelerator(args.device)

    # Clean slate
    cleanup(args.device)
    acc.reset_peak_memory_stats()
    mem_before = acc.memory_allocated()

    # Create model
    model = create_model(args)
    acc.synchronize()
    mem_after_model = acc.memory_allocated()

    # Create optimizer
    optimizer = OptClass(model.parameters(), lr=2e-4)

    # Training steps
    model.train()
    for step in range(args.train_steps):
        input_ids, labels = make_batch(args)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step == 0:
            acc.synchronize()
            mem_after_first_step = acc.max_memory_allocated()

    acc.synchronize()
    peak_mem = acc.max_memory_allocated()

    # Count optimizer state size on GPU
    gpu_state_bytes = 0
    cpu_state_bytes = 0
    for param in model.parameters():
        state = optimizer.state.get(param, {})
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                nbytes = v.numel() * v.element_size()
                if v.device.type == args.device:
                    gpu_state_bytes += nbytes
                else:
                    cpu_state_bytes += nbytes

    # Cleanup
    del optimizer, model
    cleanup(args.device)

    return {
        "name": optimizer_name,
        "peak_mem": peak_mem,
        "mem_model": mem_after_model - mem_before,
        "mem_first_step": mem_after_first_step,
        "gpu_state_bytes": gpu_state_bytes,
        "cpu_state_bytes": cpu_state_bytes,
    }


def fmt_mb(nbytes):
    return f"{nbytes / 1024**2:.1f} MB"


def fmt_gb(nbytes):
    return f"{nbytes / 1024**3:.2f} GB"


def main():
    args = get_args()

    device_type = args.device
    if device_type == "xpu":
        assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU not available!"
    elif device_type == "cuda":
        assert torch.cuda.is_available(), "CUDA not available!"

    # Print config
    model_tmp = create_model(args)
    n_params = count_params(model_tmp)
    del model_tmp
    cleanup(device_type)

    print("=" * 85)
    print("  Paged vs Non-Paged Optimizer: GPU Memory Benchmark (32-bit & 8-bit)")
    print("=" * 85)
    print(f"  Device:       {device_type}")
    print(f"  Dtype:        {args.dtype}")
    print(f"  Model:        LLaMA (hidden={args.hidden_size}, layers={args.num_layers}, heads={args.num_heads})")
    print(f"  Parameters:   {n_params:,} ({fmt_mb(n_params * (2 if args.dtype != 'fp32' else 4))})")
    print(f"  Batch:        {args.batch_size} x {args.seq_len}")
    print(f"  Train steps:  {args.train_steps}")
    expected_state = n_params * 4 * 2  # fp32, 2 states (exp_avg + exp_avg_sq)
    expected_state_8bit = n_params * 1 * 2  # int8, 2 states
    print(f"  Expected optimizer state size (32-bit): {fmt_mb(expected_state)}")
    print(f"  Expected optimizer state size (8-bit):  {fmt_mb(expected_state_8bit)}")
    print("=" * 85)

    # Define all optimizers to benchmark
    benchmarks = [
        ("AdamW", bnb.optim.AdamW),
        ("AdamW8bit", bnb.optim.AdamW8bit),
        ("PagedAdamW", bnb.optim.PagedAdamW),
        ("PagedAdamW8bit", bnb.optim.PagedAdamW8bit),
    ]

    results = []
    for i, (name, OptClass) in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] Running {name}...")
        r = measure_training(args, name, OptClass)
        print(f"  Peak GPU memory: {fmt_mb(r['peak_mem'])}")
        print(f"  Optimizer state on GPU: {fmt_mb(r['gpu_state_bytes'])}")
        print(f"  Optimizer state on CPU: {fmt_mb(r['cpu_state_bytes'])}")
        results.append(r)

    # --- Comparison ---
    col_width = 16
    header_names = [r["name"] for r in results]
    baseline_peak = results[0]["peak_mem"]

    print("\n" + "=" * 85)
    print("  RESULTS")
    print("=" * 85)
    print(f"  {'':30s}" + "".join(f"  {n:>{col_width}s}" for n in header_names))
    print(f"  {'-' * 30}" + "".join(f"  {'-' * col_width}" for _ in results))
    for label, key in [
        ("Peak GPU Memory", "peak_mem"),
        ("Optimizer State on GPU", "gpu_state_bytes"),
        ("Optimizer State on CPU (USM)", "cpu_state_bytes"),
    ]:
        print(f"  {label:30s}" + "".join(f"  {fmt_mb(r[key]):>{col_width}s}" for r in results))
    print(f"  {'-' * 30}" + "".join(f"  {'-' * col_width}" for _ in results))
    # Show savings vs baseline (AdamW)
    savings_row = []
    for r in results:
        saved = baseline_peak - r["peak_mem"]
        pct = (saved / baseline_peak) * 100 if baseline_peak > 0 else 0
        savings_row.append(f"{fmt_mb(saved)} ({pct:.1f}%)" if saved > 0 else "baseline")
    print(f"  {'GPU Memory Saved vs AdamW':30s}" + "".join(f"  {s:>{col_width}s}" for s in savings_row))
    print("=" * 85)

    for r in results[1:]:
        saved = baseline_peak - r["peak_mem"]
        if saved > 0:
            pct = (saved / baseline_peak) * 100
            print(f"\n  >>> {r['name']} saved {fmt_mb(saved)} GPU memory ({pct:.1f}% reduction vs AdamW)")

    print()


if __name__ == "__main__":
    main()


# python benchmark_paged_memory.py
# =====================================================================================
#   RESULTS
# =====================================================================================
#                                              AdamW         AdamW8bit        PagedAdamW    PagedAdamW8bit
#   ------------------------------  ----------------  ----------------  ----------------  ----------------
#   Peak GPU Memory                        2524.7 MB         1287.4 MB          861.3 MB          867.8 MB
#   Optimizer State on GPU                 1658.2 MB          421.3 MB            0.2 MB            6.8 MB
#   Optimizer State on CPU (USM)              0.0 MB            0.0 MB         1658.0 MB          414.5 MB
#   ------------------------------  ----------------  ----------------  ----------------  ----------------
#   GPU Memory Saved vs AdamW               baseline  1237.4 MB (49.0%)  1663.5 MB (65.9%)  1657.0 MB (65.6%)
# =====================================================================================

#   >>> AdamW8bit saved 1237.4 MB GPU memory (49.0% reduction vs AdamW)

#   >>> PagedAdamW saved 1663.5 MB GPU memory (65.9% reduction vs AdamW)

#   >>> PagedAdamW8bit saved 1657.0 MB GPU memory (65.6% reduction vs AdamW)
