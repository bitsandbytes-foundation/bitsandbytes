"""Benchmark NVFP4 MoE pipeline vs BF16 baseline on B200.

Compares:
  1. BF16 reference: per-expert torch.matmul in a Python loop
  2. NVFP4 MoE pipeline: 6-kernel fused pipeline (quantize → scatter → swizzle → GEMM → gather)

Reports latency (ms), speedup, and effective TFLOPS.
"""

import torch
import time
import sys


def bench_bf16_moe(x, weights, expert_offsets, num_experts, N, warmup=10, iters=50):
    """BF16 baseline: per-expert matmul in a Python loop."""
    total_tokens = expert_offsets[-1].item()
    out = torch.empty(total_tokens, N, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(warmup):
        for i in range(num_experts):
            s = expert_offsets[i].item()
            e = expert_offsets[i + 1].item()
            if e > s:
                out[s:e] = x[s:e] @ weights[i].T

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for i in range(num_experts):
            s = expert_offsets[i].item()
            e = expert_offsets[i + 1].item()
            if e > s:
                out[s:e] = x[s:e] @ weights[i].T
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000  # ms


def bench_nvfp4_moe(layer, x, expert_offsets, warmup=10, iters=50):
    """NVFP4 MoE pipeline benchmark."""
    # Warmup (first call triggers weight quantization + GEMM init)
    for _ in range(warmup):
        _ = layer(x, expert_offsets)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = layer(x, expert_offsets)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000  # ms


def run_config(num_experts, K, N, tokens_per_expert, warmup=10, iters=50):
    """Run both benchmarks for a given MoE configuration."""
    from bitsandbytes.nn.modules import LinearNVFP4MoE

    total_tokens = sum(tokens_per_expert)
    offsets = [0]
    for n in tokens_per_expert:
        offsets.append(offsets[-1] + n)
    expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    x = torch.randn(total_tokens, K, dtype=torch.bfloat16, device="cuda")

    # BF16 baseline weights
    bf16_weights = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda") * 0.02

    # NVFP4 layer
    layer = LinearNVFP4MoE(num_experts, K, N, bias=False)
    torch.nn.init.normal_(layer.weight.data, std=0.02)
    layer = layer.cuda()

    # Run benchmarks
    bf16_ms = bench_bf16_moe(x, bf16_weights, expert_offsets, num_experts, N, warmup, iters)
    nvfp4_ms = bench_nvfp4_moe(layer, x, expert_offsets, warmup, iters)

    # Compute FLOPS: 2 * total_tokens * K * N (matmul) * num_experts is wrong
    # Each expert processes its own tokens: sum(2 * tpe[i] * K * N)
    total_flops = sum(2 * t * K * N for t in tokens_per_expert)
    bf16_tflops = total_flops / (bf16_ms / 1000) / 1e12
    nvfp4_tflops = total_flops / (nvfp4_ms / 1000) / 1e12

    return {
        "bf16_ms": bf16_ms,
        "nvfp4_ms": nvfp4_ms,
        "speedup": bf16_ms / nvfp4_ms,
        "bf16_tflops": bf16_tflops,
        "nvfp4_tflops": nvfp4_tflops,
        "total_tokens": total_tokens,
        "total_flops": total_flops,
    }


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print()

    # GLM-4.7 (352B MoE) shapes from benchmarks/bench_moe_gemm_sm100.py
    # gate_up: K=4096, N=13696
    # down:    K=13696, N=4096
    configs = [
        # --- GLM-4.7 gate_up (K=4096, N=13696) ---
        {
            "name": "GLM4.7 gate_up 8e×8tok",
            "num_experts": 8,
            "K": 4096,
            "N": 13696,
            "tokens_per_expert": [8] * 8,
        },
        {
            "name": "GLM4.7 gate_up 8e×32tok",
            "num_experts": 8,
            "K": 4096,
            "N": 13696,
            "tokens_per_expert": [32] * 8,
        },
        {
            "name": "GLM4.7 gate_up 8e×64tok",
            "num_experts": 8,
            "K": 4096,
            "N": 13696,
            "tokens_per_expert": [64] * 8,
        },
        {
            "name": "GLM4.7 gate_up 8e×128tok",
            "num_experts": 8,
            "K": 4096,
            "N": 13696,
            "tokens_per_expert": [128] * 8,
        },
        {
            "name": "GLM4.7 gate_up 8e skewed",
            "num_experts": 8,
            "K": 4096,
            "N": 13696,
            "tokens_per_expert": [128, 64, 32, 16, 8, 4, 2, 1],
        },
        # --- GLM-4.7 down (K=13696, N=4096) ---
        {
            "name": "GLM4.7 down 8e×8tok",
            "num_experts": 8,
            "K": 13696,
            "N": 4096,
            "tokens_per_expert": [8] * 8,
        },
        {
            "name": "GLM4.7 down 8e×32tok",
            "num_experts": 8,
            "K": 13696,
            "N": 4096,
            "tokens_per_expert": [32] * 8,
        },
        {
            "name": "GLM4.7 down 8e×64tok",
            "num_experts": 8,
            "K": 13696,
            "N": 4096,
            "tokens_per_expert": [64] * 8,
        },
        {
            "name": "GLM4.7 down 8e×128tok",
            "num_experts": 8,
            "K": 13696,
            "N": 4096,
            "tokens_per_expert": [128] * 8,
        },
        {
            "name": "GLM4.7 down 8e skewed",
            "num_experts": 8,
            "K": 13696,
            "N": 4096,
            "tokens_per_expert": [128, 64, 32, 16, 8, 4, 2, 1],
        },
    ]

    print(f"{'Config':<45} {'BF16 (ms)':>10} {'NVFP4 (ms)':>11} {'Speedup':>8} {'BF16 TFLOPS':>12} {'NVFP4 TFLOPS':>13}")
    print("-" * 105)

    for cfg in configs:
        name = cfg.pop("name")
        tpe = cfg["tokens_per_expert"]
        result = run_config(**cfg, warmup=20, iters=100)
        total = result["total_tokens"]
        print(f"{name} ({total} tok) {' ' * max(0, 25 - len(name))}"
              f"{result['bf16_ms']:10.3f} {result['nvfp4_ms']:11.3f} "
              f"{result['speedup']:7.2f}x "
              f"{result['bf16_tflops']:11.2f} {result['nvfp4_tflops']:12.2f}")


if __name__ == "__main__":
    main()
