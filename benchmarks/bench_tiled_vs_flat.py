"""Benchmark tiled vs flat scalar GEMV with pre-allocated output buffers.

Measures kernel-only time by pre-allocating all buffers before the timing loop.
No allocations inside the measured region — fair comparison between flat and tiled.

Usage:
    python benchmarks/bench_tiled_vs_flat.py
    python benchmarks/bench_tiled_vs_flat.py --ncu   # NCU mode (single iteration)
"""

import argparse
import os
import sys

for p in [".", ".."]:
    if os.path.isfile(os.path.join(p, "bitsandbytes", "__init__.py")):
        sys.path.insert(0, os.path.abspath(p))
        break

import torch

from bitsandbytes.functional import create_normal_float_codebook

parser = argparse.ArgumentParser()
parser.add_argument("--ncu", action="store_true", help="NCU mode: single iteration, no timing")
parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
args = parser.parse_args()

SHAPES = [
    ("gateup", 2048, 5120),
    ("down", 5120, 2048),
    ("Q", 2048, 4096),
    ("KV", 2048, 512),
]
K_VALUES = [2, 3, 4, 5]
M_VALUES = [1, 2, 4]

print(f"{'shape':<8} {'K_dim':>5} {'N':>5} {'k':>2} {'M':>2}  {'flat_us':>8} {'tiled_us':>8} {'diff%':>7}")
print("-" * 60)

for name, K_dim, N in SHAPES:
    for k in K_VALUES:
        codebook = create_normal_float_codebook(k).cuda()
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")

        # Quantize and repack
        packed_flat, absmax_flat = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
        packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
            packed_flat, absmax_flat, K_dim, N, k
        )

        for M in M_VALUES:
            A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

            # Pre-allocate output buffers
            out_flat = torch.empty(M, N, dtype=torch.float16, device="cuda")
            out_tiled = torch.empty(M, N, dtype=torch.float16, device="cuda")

            if args.ncu:
                # NCU mode: single call each, profiler captures kernel time
                torch.ops.bitsandbytes.kbit_scalar_gemv.out(
                    A, packed_flat, absmax_flat, codebook, K_dim, N, k, out_flat
                )
                torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                    A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, out_tiled
                )
                print(f"{name:<8} {K_dim:>5} {N:>5} {k:>2} {M:>2}  {'ncu':>8} {'ncu':>8} {'ncu':>7}")
                continue

            # CUDA events timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # --- Flat ---
            for _ in range(args.warmup):
                torch.ops.bitsandbytes.kbit_scalar_gemv.out(
                    A, packed_flat, absmax_flat, codebook, K_dim, N, k, out_flat
                )
            torch.cuda.synchronize()

            start.record()
            for _ in range(args.iters):
                torch.ops.bitsandbytes.kbit_scalar_gemv.out(
                    A, packed_flat, absmax_flat, codebook, K_dim, N, k, out_flat
                )
            end.record()
            torch.cuda.synchronize()
            flat_us = start.elapsed_time(end) * 1000 / args.iters  # ms -> us

            # --- Tiled ---
            for _ in range(args.warmup):
                torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                    A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, out_tiled
                )
            torch.cuda.synchronize()

            start.record()
            for _ in range(args.iters):
                torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                    A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, out_tiled
                )
            end.record()
            torch.cuda.synchronize()
            tiled_us = start.elapsed_time(end) * 1000 / args.iters

            diff_pct = (tiled_us - flat_us) / flat_us * 100
            print(f"{name:<8} {K_dim:>5} {N:>5} {k:>2} {M:>2}  {flat_us:>8.1f} {tiled_us:>8.1f} {diff_pct:>+7.1f}%")

        # Correctness check (once per shape/k)
        assert torch.equal(out_flat, out_tiled) or torch.allclose(out_flat, out_tiled, rtol=0.05, atol=0.1), (
            f"MISMATCH {name} k={k}: max diff = {(out_flat - out_tiled).abs().max().item()}"
        )
