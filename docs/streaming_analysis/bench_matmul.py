#!/usr/bin/env python3
"""
RTX 4090 matmul benchmark for GLM-4.7 streaming simulation validation.

Measures actual BF16 matmul throughput using CUDA graphs for representative
layer dimensions. Compares measured TFLOPS against the simulation's assumption
of 50% peak utilization (= 82.5 TFLOPS out of 165 peak).

This lets us validate or correct the GPU_UTILIZATION parameter.
"""

import torch
import time
import sys


def benchmark_matmul(M, K, N, dtype=torch.bfloat16, warmup=20, iters=100,
                     use_cuda_graph=True, label=""):
    """
    Benchmark a single matmul: [M, K] @ [K, N] → [M, N].

    Returns measured TFLOPS (accounting for 2*M*K*N FLOPs per matmul).
    """
    device = "cuda"
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    C = torch.empty(M, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        torch.mm(A, B, out=C)
    torch.cuda.synchronize()

    if use_cuda_graph:
        # Capture CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.mm(A, B, out=C)

        # Warmup the graph
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()

        # Timed run
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        # Standard benchmark without graph
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            torch.mm(A, B, out=C)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)

    avg_ms = elapsed_ms / iters
    flops = 2 * M * K * N
    tflops = flops / (avg_ms * 1e-3) / 1e12

    return avg_ms, tflops


def main():
    if not torch.cuda.is_available():
        print("No CUDA device found.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # GLM-4.7 dimensions
    H = 5120        # hidden_size
    QD = 12288      # num_attention_heads * head_dim = 96 * 128
    KVD = 1024      # num_kv_heads * head_dim = 8 * 128
    SHARED_I = 12288  # shared_intermediate_size
    EXPERT_I = 1536   # moe_intermediate_size
    S = 1024        # seq_len

    # Representative projections in one transformer layer
    projections = [
        # (label, M_factor, K, N) — M = B * S
        ("Attn Q proj   [B*S, 5120] → [B*S, 12288]", H, QD),
        ("Attn K proj   [B*S, 5120] → [B*S, 1024]",  H, KVD),
        ("Attn V proj   [B*S, 5120] → [B*S, 1024]",  H, KVD),
        ("Attn O proj   [B*S, 12288] → [B*S, 5120]", QD, H),
        ("Shared gate   [B*S, 5120] → [B*S, 12288]", H, SHARED_I),
        ("Shared up     [B*S, 5120] → [B*S, 12288]", H, SHARED_I),
        ("Shared down   [B*S, 12288] → [B*S, 5120]", SHARED_I, H),
        ("Expert gate   [B*S, 5120] → [B*S, 1536]",  H, EXPERT_I),
        ("Expert up     [B*S, 5120] → [B*S, 1536]",  H, EXPERT_I),
        ("Expert down   [B*S, 1536] → [B*S, 5120]",  EXPERT_I, H),
    ]

    batch_sizes = [1, 2, 4, 8, 16, 32]

    print("=" * 100)
    print("BF16 MATMUL BENCHMARK — GLM-4.7 layer dimensions")
    print("Using CUDA graphs to minimize kernel launch overhead")
    print("=" * 100)
    print()

    # First: sweep batch sizes for total layer forward FLOP estimate
    print("--- Total layer forward pass (all projections) ---")
    print()
    hdr = f"{'B':>3s}  {'M=B*S':>7s}  {'Total ms':>9s}  {'TFLOPS':>8s}  {'Util%':>6s}  {'vs sim':>8s}"
    print(hdr)
    print("-" * len(hdr))

    for B in batch_sizes:
        M = B * S
        total_ms = 0
        total_flops = 0

        for label, K, N in projections:
            ms, _ = benchmark_matmul(M, K, N, use_cuda_graph=True)
            total_ms += ms
            total_flops += 2 * M * K * N

        # For routing experts: 8 active, each with gate+up+down
        # Expert projections are the last 3 in the list — multiply by 8
        for label, K, N in projections[-3:]:
            ms, _ = benchmark_matmul(M, K, N, use_cuda_graph=True)
            total_ms += ms * 7  # already counted once, add 7 more
            total_flops += 7 * 2 * M * K * N

        # Add attention QK^T and softmax*V (~5% of projection FLOPs)
        # These are memory-bound at small B, hard to benchmark precisely
        attn_flops = 2 * 2 * B * 96 * S * S * 128  # QK^T + score*V
        total_flops += attn_flops
        # Estimate attn time as proportional to flops at same utilization
        est_attn_ms = total_ms * (attn_flops / (total_flops - attn_flops)) if total_flops > attn_flops else 0
        total_ms += est_attn_ms

        measured_tflops = total_flops / (total_ms * 1e-3) / 1e12
        utilization = measured_tflops / 165 * 100

        # Compare with simulation: sim assumes 50% of 165 = 82.5 TFLOPS
        sim_time_ms = total_flops / (82.5e12) * 1000
        ratio = sim_time_ms / total_ms

        print(f"{B:>3d}  {M:>7d}  {total_ms:>8.2f}ms  {measured_tflops:>7.1f}T  "
              f"{utilization:>5.1f}%  {ratio:>6.2f}x")

    print()
    print("Util% = measured TFLOPS / 165 peak")
    print("vs sim = sim_predicted_time / actual_time (>1 = sim is conservative, <1 = sim is optimistic)")
    print()

    # Per-projection breakdown at B=8
    B = 8
    M = B * S
    print(f"--- Per-projection breakdown at B={B}, S={S}, M={M} ---")
    print()
    hdr2 = f"{'Projection':40s}  {'[M,K]→[M,N]':>16s}  {'ms':>7s}  {'TFLOPS':>8s}  {'Util%':>6s}"
    print(hdr2)
    print("-" * len(hdr2))

    for label, K, N in projections:
        ms, tflops = benchmark_matmul(M, K, N, use_cuda_graph=True)
        util = tflops / 165 * 100
        dims = f"[{M},{K}]x[{K},{N}]"
        print(f"{label:40s}  {dims:>20s}  "
              f"{ms:>6.3f}ms  {tflops:>7.1f}T  {util:>5.1f}%")

    print()

    # Graph vs no-graph comparison at B=8
    print(f"--- CUDA graph vs standard at B={B} ---")
    print()
    proj_label, K, N = projections[0]  # Q proj as representative
    ms_graph, tflops_graph = benchmark_matmul(M, K, N, use_cuda_graph=True)
    ms_no_graph, tflops_no_graph = benchmark_matmul(M, K, N, use_cuda_graph=False)
    print(f"Q proj [{M},{K}]→[{M},{N}]:")
    print(f"  With CUDA graph:    {ms_graph:.3f} ms, {tflops_graph:.1f} TFLOPS ({tflops_graph/165*100:.1f}% peak)")
    print(f"  Without CUDA graph: {ms_no_graph:.3f} ms, {tflops_no_graph:.1f} TFLOPS ({tflops_no_graph/165*100:.1f}% peak)")
    print(f"  Graph speedup: {ms_no_graph/ms_graph:.2f}x")
    print()

    # Summary recommendation
    print("=" * 60)
    print("RECOMMENDATION FOR SIMULATION PARAMETERS")
    print("=" * 60)
    print()
    # Use B=8 as representative (common micro-batch size)
    total_ms = 0
    total_flops = 0
    for label, K, N in projections:
        ms, _ = benchmark_matmul(M, K, N, use_cuda_graph=True)
        total_ms += ms
        total_flops += 2 * M * K * N
    for label, K, N in projections[-3:]:
        ms, _ = benchmark_matmul(M, K, N, use_cuda_graph=True)
        total_ms += ms * 7
        total_flops += 7 * 2 * M * K * N
    attn_flops = 2 * 2 * B * 96 * S * S * 128
    total_flops += attn_flops
    est_attn_ms = total_ms * (attn_flops / (total_flops - attn_flops))
    total_ms += est_attn_ms
    measured_tflops = total_flops / (total_ms * 1e-3) / 1e12
    utilization = measured_tflops / 165

    print(f"Measured effective utilization at B=8: {utilization*100:.1f}%")
    print(f"Simulation assumes: 50.0%")
    print()
    if utilization > 0.55:
        print(f"Simulation is CONSERVATIVE — real throughput is {utilization/0.5:.2f}x what sim predicts.")
        print(f"Consider increasing GPU_UTILIZATION to {utilization:.2f}")
    elif utilization < 0.45:
        print(f"Simulation is OPTIMISTIC — real throughput is {utilization/0.5:.2f}x what sim predicts.")
        print(f"Consider decreasing GPU_UTILIZATION to {utilization:.2f}")
    else:
        print(f"Simulation's 50% assumption is reasonable (measured {utilization*100:.1f}%).")

    print()
    print("NOTE: This benchmarks BF16 matmuls, not NF4 quantized matmuls.")
    print("NF4 dequant overhead typically reduces throughput by 10-30%.")
    print("True NF4 benchmark would require bitsandbytes quantization kernels.")


if __name__ == "__main__":
    main()
