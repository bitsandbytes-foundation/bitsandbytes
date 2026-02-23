"""CUDA event benchmark for kbit kernels — measures kernel-only latency.

Uses pre-allocated output buffers (out parameter) and CUDA events to
measure just the kernel execution time, excluding allocation overhead.

Output: same shape/k/M grid as bench_ncu.sh for direct comparison.

Usage:
    python benchmarks/bench_cuda_events.py                   # all kernels
    python benchmarks/bench_cuda_events.py --kernel mma      # MMA only
    python benchmarks/bench_cuda_events.py --kernel scalar    # scalar GEMV only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from bitsandbytes.functional import create_normal_float_codebook

WARMUP = 20
ITERS = 100

# Same shapes as ncu_driver.py
dense_shapes = [
    ("gateup", 2048, 5120),
    ("down", 5120, 2048),
    ("Q", 2048, 4096),
    ("O", 4096, 2048),
    ("KV", 2048, 512),
]

moe_shapes = [
    ("moe_gu", 2048, 512),
    ("moe_dn", 512, 2048),
]

k_bits_list = [2, 3, 4, 5]
NUM_EXPERTS = 8


def bench_kernel(fn, warmup=WARMUP, iters=ITERS):
    """Time a kernel using CUDA graph replay + events.

    Captures the kernel into a CUDA graph, then replays it to measure
    kernel-only latency without Python dispatch or launch overhead.
    """
    # Warmup (eager, to JIT compile etc.)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Capture into CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    # Time graph replays
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup the graph replay
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return (total_ms / iters) * 1000.0  # convert ms -> us


def prepare_dense_data(device):
    """Pre-quantize all dense shapes for all k values."""
    data = {}
    for name, K_dim, N in dense_shapes:
        for k in k_bits_list:
            codebook = create_normal_float_codebook(k, device=device)
            W = torch.randn(K_dim * N, device=device, dtype=torch.float32)
            packed_flat, absmax_flat = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
            packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(packed_flat, absmax_flat, K_dim, N, k)
            data[(name, k)] = (K_dim, N, packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook)
    return data


def prepare_moe_data(device):
    """Pre-quantize MoE expert weights."""
    data = {}
    for name, K_dim, N in moe_shapes:
        for k in k_bits_list:
            codebook = create_normal_float_codebook(k, device=device)
            packed_list, absmax_list = [], []
            for _ in range(NUM_EXPERTS):
                W = torch.randn(K_dim * N, device=device, dtype=torch.float32)
                pf, af = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
                pt, at = torch.ops.bitsandbytes.repack_kbit(pf, af, K_dim, N, k)
                packed_list.append(pt)
                absmax_list.append(at)
            B_packed_all = torch.cat(packed_list, dim=0)
            B_absmax_all = torch.cat(absmax_list, dim=0)
            data[(name, k)] = (K_dim, N, B_packed_all, B_absmax_all, codebook)
    return data


def bench_mma(data, m_vals, device):
    """Benchmark MMA GEMM kernel with out parameter."""
    print("\n=== MMA kernel (CUDA events) ===")
    print(f"{'shape':<8} {'k':>2} {'M':>2} {'avg_us':>10}")
    print("---")

    for name, K_dim, N in dense_shapes:
        for k in k_bits_list:
            K_dim, N, _, _, packed_tiled, absmax_tiled, codebook = data[(name, k)]
            for M in m_vals:
                A = torch.randn(M, K_dim, dtype=torch.float16, device=device)
                out = torch.empty(M, N, dtype=torch.float16, device=device)

                # Allocate workspace and tile_counters for the _ variant
                C_workspace = torch.zeros(M, N, dtype=torch.float32, device=device)
                # Upper bound on tile count
                TILE_M = 16 * max(1, min(4, (M + 15) // 16))
                TILE_N = 64 if M <= 16 and N % 64 == 0 else 128
                m_tiles = (M + TILE_M - 1) // TILE_M
                n_tiles = N // TILE_N
                tile_counters = torch.zeros(m_tiles * n_tiles, dtype=torch.int32, device=device)

                fn = lambda: torch.ops.bitsandbytes.kbit_gemm_prod_(
                    A,
                    packed_tiled,
                    absmax_tiled,
                    codebook,
                    K_dim,
                    N,
                    k,
                    1,
                    out,
                    C_workspace,
                    tile_counters,
                )
                avg_us = bench_kernel(fn)
                print(f"{name:<8} {k:>2} {M:>2} {avg_us:>10.2f}")


def bench_scalar(data, m_vals, device):
    """Benchmark scalar GEMV kernel with out parameter (tiled layout)."""
    m_vals = [m for m in m_vals if m <= 4]
    if not m_vals:
        print("\n=== Scalar GEMV (CUDA events) ===\n(no M<=4 values)")
        return

    print(f"\n=== Scalar GEMV M<={max(m_vals)} (CUDA events) ===")
    print(f"{'shape':<8} {'k':>2} {'M':>2} {'avg_us':>10}")
    print("---")

    for name, K_dim, N in dense_shapes:
        for k in k_bits_list:
            K_dim, N, _, _, packed_tiled, absmax_tiled, codebook = data[(name, k)]
            for M in m_vals:
                A = torch.randn(M, K_dim, dtype=torch.float16, device=device)
                out = torch.empty(M, N, dtype=torch.float16, device=device)

                fn = lambda: torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                    A,
                    packed_tiled,
                    absmax_tiled,
                    codebook,
                    K_dim,
                    N,
                    k,
                    out,
                )
                avg_us = bench_kernel(fn)
                print(f"{name:<8} {k:>2} {M:>2} {avg_us:>10.2f}")


def bench_grouped(moe_data, m_vals, device):
    """Benchmark grouped MMA kernel (MoE)."""
    print(f"\n=== Grouped MMA ({NUM_EXPERTS} experts, CUDA events) ===")
    print(f"{'shape':<8} {'k':>2} {'M':>2} {'avg_us':>10}")
    print("---")

    for name, K_dim, N in moe_shapes:
        for k in k_bits_list:
            K_dim, N, B_packed_all, B_absmax_all, codebook = moe_data[(name, k)]
            for M in m_vals:
                total_tokens = M * NUM_EXPERTS
                A_concat = torch.randn(total_tokens, K_dim, dtype=torch.float16, device=device)
                offsets = list(range(0, total_tokens + 1, M))
                expert_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)

                # Grouped GEMM doesn't have an _ variant yet — use the allocating version
                fn = lambda: torch.ops.bitsandbytes.kbit_grouped_gemm(
                    A_concat,
                    B_packed_all,
                    B_absmax_all,
                    codebook,
                    expert_offsets,
                    K_dim,
                    N,
                    k,
                    NUM_EXPERTS,
                    M,
                )
                avg_us = bench_kernel(fn)
                print(f"{name:<8} {k:>2} {M:>2} {avg_us:>10.2f}")


def main():
    parser = argparse.ArgumentParser(description="CUDA event kernel benchmark")
    parser.add_argument("--kernel", choices=["mma", "scalar", "grouped", "all"], default="all")
    parser.add_argument("--m-vals", default="1,2,3,4,5,6,7,8", help="Comma-separated M values")
    args = parser.parse_args()

    m_vals = [int(x) for x in args.m_vals.split(",")]
    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print(f"M values: {m_vals}")

    dense_data = prepare_dense_data(device)

    if args.kernel in ("mma", "all"):
        bench_mma(dense_data, m_vals, device)

    if args.kernel in ("scalar", "all"):
        bench_scalar(dense_data, m_vals, device)

    if args.kernel in ("grouped", "all"):
        moe_data = prepare_moe_data(device)
        bench_grouped(moe_data, m_vals, device)


if __name__ == "__main__":
    main()
