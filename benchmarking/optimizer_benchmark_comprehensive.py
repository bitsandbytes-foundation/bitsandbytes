"""
Comprehensive Optimizer Benchmark: 8-bit vs 32-bit vs Paged

Benchmarks:
  - CUDA / XPU: AdamW (32-bit) vs AdamW8bit vs PagedAdamW vs PagedAdamW8bit
  - CPU:        PyTorch AdamW (32-bit) vs bnb AdamW8bit

Measures:
  - Peak memory usage (GPU only, CPU uses system RAM)
  - Optimizer state memory (GPU vs CPU breakdown)
  - Per-step training time (with warmup + device synchronization)
  - Total training time

Usage:
    # Auto-detect device (prefers xpu > cuda > cpu)
    python benchmarking/optimizer_benchmark_comprehensive.py

    # CUDA
    python benchmarking/optimizer_benchmark_comprehensive.py --device cuda

    # XPU
    python benchmarking/optimizer_benchmark_comprehensive.py --device xpu

    # CPU (bind to NUMA node 0 for stable benchmarks)
    numactl --cpunodebind=0 --membind=0 python benchmarking/optimizer_benchmark_comprehensive.py --device cpu

    # Custom model (default: Qwen/Qwen2.5-1.5B-Instruct)
    python benchmarking/optimizer_benchmark_comprehensive.py --device cuda --model meta-llama/Llama-3.2-1B

    # Use fp32 (default is bf16)
    python benchmarking/optimizer_benchmark_comprehensive.py --device cuda --dtype fp32

Requires: pip install transformers
"""

import argparse
import gc
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import bitsandbytes as bnb

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def get_args():
    parser = argparse.ArgumentParser(description="Comprehensive Optimizer Benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda, xpu, or cpu (default: cuda)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--train_steps", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Training dtype (default: bf16)")
    return parser.parse_args()


def detect_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_torch_dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def get_accelerator(device_type):
    if device_type == "xpu":
        return torch.xpu
    if device_type == "cuda":
        return torch.cuda
    return None


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def fmt_mb(nbytes):
    return f"{nbytes / 1024**2:.1f} MB"


def fmt_sec(t):
    return f"{t:.3f}s"


def create_model(args):
    """Create a causal LM from HuggingFace config (random weights, no download of weights)."""
    config = AutoConfig.from_pretrained(args.model)
    dtype = get_torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model = model.to(device=args.device)
    return model, config.vocab_size


def cleanup(device_type):
    gc.collect()
    acc = get_accelerator(device_type)
    if acc is not None:
        acc.empty_cache()
        acc.synchronize()


def measure_optimizer_state(model, optimizer, device_type):
    """Count optimizer state bytes on device vs CPU."""
    device_bytes = 0
    cpu_bytes = 0
    for param in model.parameters():
        state = optimizer.state.get(param, {})
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                nbytes = v.numel() * v.element_size()
                if v.device.type == device_type:
                    device_bytes += nbytes
                elif v.device.type == "cpu":
                    cpu_bytes += nbytes
    return device_bytes, cpu_bytes


def run_benchmark_gpu(args, name, OptimizerClass):
    """Benchmark an optimizer on GPU (CUDA or XPU). Returns metrics dict."""
    acc = get_accelerator(args.device)
    cleanup(args.device)
    acc.reset_peak_memory_stats()

    mem_before = acc.memory_allocated()

    model, vocab_size = create_model(args)
    acc.synchronize()
    mem_after_model = acc.memory_allocated()

    optimizer = OptimizerClass(model.parameters(), lr=2e-4)
    model.train()

    step_times = []
    total_steps = args.warmup_steps + args.train_steps

    for step in range(total_steps):
        input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=args.device)
        labels = input_ids.clone()

        acc.synchronize()
        t0 = time.perf_counter()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc.synchronize()
        t1 = time.perf_counter()

        if step >= args.warmup_steps:
            step_times.append(t1 - t0)

    acc.synchronize()
    peak_mem = acc.max_memory_allocated()
    device_state, cpu_state = measure_optimizer_state(model, optimizer, args.device)

    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    total_time = sum(step_times)

    del optimizer, model
    cleanup(args.device)

    return {
        "name": name,
        "peak_mem": peak_mem,
        "model_mem": mem_after_model - mem_before,
        "device_state": device_state,
        "cpu_state": cpu_state,
        "avg_step_time": avg_step_time,
        "total_time": total_time,
    }


def run_benchmark_cpu(args, name, OptimizerClass, is_pytorch=False):
    """Benchmark an optimizer on CPU. Returns metrics dict."""
    import resource

    cleanup("cpu")

    model, vocab_size = create_model(args)
    optimizer = OptimizerClass(model.parameters(), lr=2e-4)

    model.train()

    step_times = []
    total_steps = args.warmup_steps + args.train_steps

    for step in range(total_steps):
        input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len))
        labels = input_ids.clone()

        # Ensure all async ops are done before timing
        torch.cpu.synchronize()
        t0 = time.perf_counter()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cpu.synchronize()
        t1 = time.perf_counter()
        if step >= args.warmup_steps:
            step_times.append(t1 - t0)

    # Measure optimizer state size
    state_bytes = 0
    for param in model.parameters():
        state = optimizer.state.get(param, {})
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state_bytes += v.numel() * v.element_size()

    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    total_time = sum(step_times)

    # RSS (resident set size) from OS
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: KB -> bytes

    del optimizer, model
    cleanup("cpu")

    return {
        "name": name,
        "state_bytes": state_bytes,
        "avg_step_time": avg_step_time,
        "total_time": total_time,
        "rss_bytes": rss_bytes,
    }


def print_gpu_results(results, args):
    baseline = results[0]
    col_w = 18
    names = [r["name"] for r in results]

    print("\n" + "=" * 100)
    print(f"  RESULTS ({args.device.upper()})")
    print("=" * 100)

    # Header
    print(f"  {'Metric':35s}" + "".join(f"  {n:>{col_w}s}" for n in names))
    print(f"  {'-' * 35}" + "".join(f"  {'-' * col_w}" for _ in results))

    # Memory rows
    for label, key in [
        ("Peak GPU Memory", "peak_mem"),
        ("Model Memory", "model_mem"),
        ("Optimizer State on GPU", "device_state"),
        ("Optimizer State on CPU", "cpu_state"),
    ]:
        print(f"  {label:35s}" + "".join(f"  {fmt_mb(r[key]):>{col_w}s}" for r in results))

    print(f"  {'-' * 35}" + "".join(f"  {'-' * col_w}" for _ in results))

    # Savings
    base_peak = baseline["peak_mem"]
    savings = []
    for r in results:
        saved = base_peak - r["peak_mem"]
        pct = (saved / base_peak) * 100 if base_peak > 0 else 0
        if saved > 0:
            savings.append(f"{fmt_mb(saved)} ({pct:.1f}%)")
        else:
            savings.append(f"{fmt_mb(base_peak)} (100%)")
    print(f"  {'GPU Mem Saved vs baseline':35s}" + "".join(f"  {s:>{col_w}s}" for s in savings))

    base_time = baseline["avg_step_time"]
    speedups = []
    for r in results:
        if base_time > 0 and r["avg_step_time"] > 0:
            ratio = base_time / r["avg_step_time"]
            speedups.append(f"{ratio:.2f}x")
        else:
            speedups.append("N/A")
    print(f"  {'Speed vs baseline':35s}" + "".join(f"  {s:>{col_w}s}" for s in speedups))

    print("=" * 100)

    print("\nKey Takeaways:")
    for r in results[1:]:
        saved = base_peak - r["peak_mem"]
        if saved > 0:
            pct = (saved / base_peak) * 100
            print(f"  - {r['name']}: saves {fmt_mb(saved)} GPU memory ({pct:.1f}% reduction)")
        speedup = base_time / r["avg_step_time"] if r["avg_step_time"] > 0 else 0
        if speedup > 1.05:
            print(f"    {speedup:.2f}x faster per step")
        elif speedup < 0.95:
            print(f"    {1/speedup:.2f}x slower per step (tradeoff for memory savings)")


def print_cpu_results(results, args):
    baseline = results[0]
    col_w = 20

    names = [r["name"] for r in results]
    print("\n" + "=" * 80)
    print("  RESULTS (CPU)")
    print("=" * 80)

    print(f"  {'Metric':30s}" + "".join(f"  {n:>{col_w}s}" for n in names))
    print(f"  {'-' * 30}" + "".join(f"  {'-' * col_w}" for _ in results))

    print(f"  {'Optimizer State Size':30s}" + "".join(f"  {fmt_mb(r['state_bytes']):>{col_w}s}" for r in results))

    print(f"  {'-' * 30}" + "".join(f"  {'-' * col_w}" for _ in results))

    # State size savings
    base_state = baseline["state_bytes"]
    savings = []
    for r in results:
        saved = base_state - r["state_bytes"]
        pct = (saved / base_state) * 100 if base_state > 0 else 0
        if saved > 0:
            savings.append(f"{fmt_mb(saved)} ({pct:.1f}%)")
        else:
            savings.append("baseline")
    print(f"  {'State Size Saved':30s}" + "".join(f"  {s:>{col_w}s}" for s in savings))

    base_time = baseline["avg_step_time"]
    speedups = []
    for r in results:
        if base_time > 0 and r["avg_step_time"] > 0:
            ratio = base_time / r["avg_step_time"]
            speedups.append(f"{ratio:.2f}x")
        else:
            speedups.append("N/A")
    print(f"  {'Speed vs baseline':30s}" + "".join(f"  {s:>{col_w}s}" for s in speedups))
    print("=" * 80)

    print("\nKey Takeaways:")
    for r in results[1:]:
        saved = base_state - r["state_bytes"]
        if saved > 0:
            pct = (saved / base_state) * 100
            print(f"  - {r['name']}: saves {fmt_mb(saved)} optimizer state ({pct:.1f}% reduction)")
        speedup = base_time / r["avg_step_time"] if r["avg_step_time"] > 0 else 0
        if speedup > 1.05:
            print(f"    {speedup:.2f}x faster per step")
        elif speedup < 0.95:
            print(f"    {1/speedup:.2f}x slower per step (tradeoff for memory savings)")


def main():
    args = get_args()

    is_gpu = args.device in ("cuda", "xpu")

    # Validate device
    if args.device == "xpu":
        assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU not available!"
    elif args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available!"

    # Print configuration
    model_tmp, vocab_size = create_model(args)
    n_params = count_params(model_tmp)
    elem_size = 2 if args.dtype != "fp32" else 4
    del model_tmp
    cleanup(args.device)

    print("=" * 100)
    print("  Comprehensive Optimizer Benchmark")
    print("=" * 100)
    print(f"  Device:         {args.device}")
    print(f"  Dtype:          {args.dtype}")
    print(f"  Model:          {args.model}")
    print(f"  Parameters:     {n_params:,} ({fmt_mb(n_params * elem_size)} in {args.dtype})")
    print(f"  Batch:          {args.batch_size} x {args.seq_len}")
    print(f"  Warmup steps:   {args.warmup_steps}")
    print(f"  Measured steps: {args.train_steps}")
    expected_32bit = n_params * 4 * 2  # fp32, 2 states for AdamW
    expected_8bit = n_params * 1 * 2   # int8, 2 states
    print(f"  Expected optimizer state (32-bit): {fmt_mb(expected_32bit)}")
    print(f"  Expected optimizer state (8-bit):  {fmt_mb(expected_8bit)}")
    print("=" * 100)

    if is_gpu:
        # GPU benchmark: AdamW32bit vs AdamW8bit vs PagedAdamW vs PagedAdamW8bit
        benchmarks = [
            ("AdamW (32-bit)", bnb.optim.AdamW),
            ("AdamW8bit", bnb.optim.AdamW8bit),
            ("PagedAdamW", bnb.optim.PagedAdamW),
            ("PagedAdamW8bit", bnb.optim.PagedAdamW8bit),
        ]

        results = []
        for i, (name, OptClass) in enumerate(benchmarks, 1):
            print(f"\n[{i}/{len(benchmarks)}] Running {name}...")
            r = run_benchmark_gpu(args, name, OptClass)
            print(f"  Peak GPU memory: {fmt_mb(r['peak_mem'])}")
            print(f"  Optimizer state on GPU: {fmt_mb(r['device_state'])}")
            print(f"  Optimizer state on CPU: {fmt_mb(r['cpu_state'])}")
            print(f"  Avg step time: {fmt_sec(r['avg_step_time'])}")
            results.append(r)

        print_gpu_results(results, args)

    else:
        # CPU benchmark: PyTorch AdamW vs bnb AdamW8bit
        benchmarks = [
            ("PyTorch AdamW (32-bit)", torch.optim.AdamW, True),
            ("bnb AdamW8bit", bnb.optim.AdamW8bit, False),
        ]

        results = []
        for i, (name, OptClass, is_pytorch) in enumerate(benchmarks, 1):
            print(f"\n[{i}/{len(benchmarks)}] Running {name}...")
            r = run_benchmark_cpu(args, name, OptClass, is_pytorch=is_pytorch)
            print(f"  Optimizer state size: {fmt_mb(r['state_bytes'])}")
            print(f"  Avg step time: {fmt_sec(r['avg_step_time'])}")
            results.append(r)

        print_cpu_results(results, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
