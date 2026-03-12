"""
Benchmark: Paged vs Non-Paged Optimizer GPU Memory Usage.

Demonstrates that paged optimizers significantly reduce GPU memory consumption
by storing optimizer states in CPU/GPU shared memory (USM) instead of pure GPU memory.

Usage:
    python tests/benchmark_paged_memory.py
    python tests/benchmark_paged_memory.py --hidden_size 2048 --num_layers 16
    python tests/benchmark_paged_memory.py --device cuda  # also works on CUDA
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


def measure_training(args, optimizer_name, paged):
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
    if paged:
        OptClass = bnb.optim.PagedAdamW
    else:
        OptClass = bnb.optim.AdamW
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

    print("=" * 65)
    print("  Paged vs Non-Paged Optimizer: GPU Memory Benchmark")
    print("=" * 65)
    print(f"  Device:       {device_type}")
    print(f"  Dtype:        {args.dtype}")
    print(f"  Model:        LLaMA (hidden={args.hidden_size}, layers={args.num_layers}, heads={args.num_heads})")
    print(f"  Parameters:   {n_params:,} ({fmt_mb(n_params * (2 if args.dtype != 'fp32' else 4))})")
    print(f"  Batch:        {args.batch_size} x {args.seq_len}")
    print(f"  Train steps:  {args.train_steps}")
    expected_state = n_params * 4 * 2  # fp32, 2 states (exp_avg + exp_avg_sq)
    print(f"  Expected optimizer state size: {fmt_mb(expected_state)}")
    print("=" * 65)

    # --- Run non-paged ---
    print("\n[1/2] Running AdamW (non-paged)...")
    r_normal = measure_training(args, "AdamW", paged=False)
    print(f"  Peak GPU memory: {fmt_mb(r_normal['peak_mem'])}")
    print(f"  Optimizer state on GPU: {fmt_mb(r_normal['gpu_state_bytes'])}")
    print(f"  Optimizer state on CPU: {fmt_mb(r_normal['cpu_state_bytes'])}")

    # --- Run paged ---
    print("\n[2/2] Running PagedAdamW (paged)...")
    r_paged = measure_training(args, "PagedAdamW", paged=True)
    print(f"  Peak GPU memory: {fmt_mb(r_paged['peak_mem'])}")
    print(f"  Optimizer state on GPU: {fmt_mb(r_paged['gpu_state_bytes'])}")
    print(f"  Optimizer state on CPU: {fmt_mb(r_paged['cpu_state_bytes'])}")

    # --- Comparison ---
    saved = r_normal["peak_mem"] - r_paged["peak_mem"]
    pct = (saved / r_normal["peak_mem"]) * 100 if r_normal["peak_mem"] > 0 else 0

    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  {'':30s} {'AdamW':>12s}  {'PagedAdamW':>12s}")
    print(f"  {'-'*30} {'-'*12}  {'-'*12}")
    print(f"  {'Peak GPU Memory':30s} {fmt_mb(r_normal['peak_mem']):>12s}  {fmt_mb(r_paged['peak_mem']):>12s}")
    print(f"  {'Optimizer State on GPU':30s} {fmt_mb(r_normal['gpu_state_bytes']):>12s}  {fmt_mb(r_paged['gpu_state_bytes']):>12s}")
    print(f"  {'Optimizer State on CPU (USM)':30s} {fmt_mb(r_normal['cpu_state_bytes']):>12s}  {fmt_mb(r_paged['cpu_state_bytes']):>12s}")
    print(f"  {'-'*30} {'-'*12}  {'-'*12}")
    print(f"  {'GPU Memory Saved':30s} {fmt_mb(saved):>12s}  ({pct:.1f}%)")
    print("=" * 65)

    if saved > 0:
        print(f"\n  >>> PagedAdamW saved {fmt_mb(saved)} GPU memory ({pct:.1f}% reduction)")
        print(f"  >>> Optimizer states moved to shared memory (USM), freeing GPU VRAM")
    else:
        print("\n  NOTE: No memory saving detected. Model may be too small to observe the difference.")

    print()


if __name__ == "__main__":
    main()


# python benchmark_paged_memory.py
# =================================================================
#   RESULTS
# =================================================================
#                                         AdamW    PagedAdamW
#   ------------------------------ ------------  ------------
#   Peak GPU Memory                   2524.7 MB      861.3 MB
#   Optimizer State on GPU            1658.2 MB        0.2 MB
#   Optimizer State on CPU (USM)         0.0 MB     1658.0 MB
#   ------------------------------ ------------  ------------
#   GPU Memory Saved                  1663.5 MB  (65.9%)
# =================================================================

#   >>> PagedAdamW saved 1663.5 MB GPU memory (65.9% reduction)
#   >>> Optimizer states moved to shared memory (USM), freeing GPU VRAM
