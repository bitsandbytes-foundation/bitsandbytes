"""Benchmark tiled vs flat scalar GEMV with pre-allocated output buffers.

Measures kernel-only time by pre-allocating all buffers before the timing loop.
No allocations inside the measured region — fair comparison between flat, tiled, and tiled v2.

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
parser.add_argument("--graph", action="store_true", help="Use CUDA graph replay for accurate kernel timing")
parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
parser.add_argument("--iters", type=int, default=100, help="Timed iterations per trial")
parser.add_argument("--trials", type=int, default=5, help="Number of trials for stddev (graph mode)")
args = parser.parse_args()

SHAPES = [
    ("gateup", 2048, 5120),
    ("down", 5120, 2048),
    ("Q", 2048, 4096),
    ("O", 4096, 2048),
    ("KV", 2048, 512),
]
K_VALUES = [2, 3, 4, 5]
M_VALUES = [1, 2, 4]

if args.graph:
    print(
        f"{'shape':<8} {'K_dim':>5} {'N':>5} {'k':>2} {'M':>2}"
        f"  {'flat_us':>8} {'±flat':>6}"
        f" {'tiled_us':>8} {'±tl':>4}"
        f" {'v2_us':>8} {'±v2':>4}"
        f" {'tl/fl%':>7} {'v2/fl%':>7}"
    )
    print("-" * 100)
else:
    print(
        f"{'shape':<8} {'K_dim':>5} {'N':>5} {'k':>2} {'M':>2}"
        f"  {'flat_us':>8} {'tiled_us':>8} {'v2_us':>8} {'tl/fl%':>7} {'v2/fl%':>7}"
    )
    print("-" * 75)

for name, K_dim, N in SHAPES:
    for k in K_VALUES:
        codebook = create_normal_float_codebook(k).cuda()
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")

        # Quantize and repack
        packed_flat, absmax_flat = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
        packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(packed_flat, absmax_flat, K_dim, N, k)

        for M in M_VALUES:
            A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

            # Pre-allocate output buffers
            out_flat = torch.empty(M, N, dtype=torch.float16, device="cuda")
            out_tiled = torch.empty(M, N, dtype=torch.float16, device="cuda")
            out_v2 = torch.empty(M, N, dtype=torch.float16, device="cuda")

            # v2 workspace
            n_tiles = N // 128
            C_workspace = torch.zeros(M, N, dtype=torch.float32, device="cuda")
            tile_counters = torch.zeros(n_tiles, dtype=torch.int32, device="cuda")

            if args.ncu:
                torch.ops.bitsandbytes.kbit_scalar_gemv.out(
                    A, packed_flat, absmax_flat, codebook, K_dim, N, k, out_flat
                )
                torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                    A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, out_tiled
                )
                torch.ops.bitsandbytes.kbit_scalar_gemv_v2_(
                    A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, out_v2, C_workspace, tile_counters
                )
                print(
                    f"{name:<8} {K_dim:>5} {N:>5} {k:>2} {M:>2}"
                    f"  {'ncu':>8} {'ncu':>8} {'ncu':>8} {'ncu':>7} {'ncu':>7}"
                )
                continue

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            def call_flat():
                torch.ops.bitsandbytes.kbit_scalar_gemv.out(
                    A, packed_flat, absmax_flat, codebook, K_dim, N, k, out_flat
                )

            def call_tiled():
                torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                    A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, out_tiled
                )

            def call_v2():
                torch.ops.bitsandbytes.kbit_scalar_gemv_v2_(
                    A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, out_v2, C_workspace, tile_counters
                )

            if args.graph:
                import statistics

                # CUDA graph replay — measures kernel-only time
                for fn in (call_flat, call_tiled, call_v2):
                    for _ in range(3):
                        fn()
                torch.cuda.synchronize()

                def bench_graph(fn, trials, iters):
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=s):
                            fn()
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(trials):
                        start.record()
                        for _ in range(iters):
                            g.replay()
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end) * 1000 / iters)
                    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0

                flat_us, flat_std = bench_graph(call_flat, args.trials, args.iters)
                tiled_us, tiled_std = bench_graph(call_tiled, args.trials, args.iters)
                v2_us, v2_std = bench_graph(call_v2, args.trials, args.iters)
            else:

                def bench_events(fn):
                    for _ in range(args.warmup):
                        fn()
                    torch.cuda.synchronize()
                    start.record()
                    for _ in range(args.iters):
                        fn()
                    end.record()
                    torch.cuda.synchronize()
                    return start.elapsed_time(end) * 1000 / args.iters

                flat_us = bench_events(call_flat)
                tiled_us = bench_events(call_tiled)
                v2_us = bench_events(call_v2)

            tl_pct = (tiled_us - flat_us) / flat_us * 100
            v2_pct = (v2_us - flat_us) / flat_us * 100
            if args.graph:
                print(
                    f"{name:<8} {K_dim:>5} {N:>5} {k:>2} {M:>2}"
                    f"  {flat_us:>8.1f} {flat_std:>5.1f}σ"
                    f" {tiled_us:>8.1f} {tiled_std:>3.1f}σ"
                    f" {v2_us:>8.1f} {v2_std:>3.1f}σ"
                    f" {tl_pct:>+7.1f}% {v2_pct:>+7.1f}%"
                )
            else:
                print(
                    f"{name:<8} {K_dim:>5} {N:>5} {k:>2} {M:>2}"
                    f"  {flat_us:>8.1f} {tiled_us:>8.1f} {v2_us:>8.1f} {tl_pct:>+7.1f}% {v2_pct:>+7.1f}%"
                )

        # Correctness check (once per shape/k)
        assert torch.equal(out_flat, out_tiled) or torch.allclose(out_flat, out_tiled, rtol=0.05, atol=0.1), (
            f"MISMATCH flat vs tiled {name} k={k}: max diff = {(out_flat - out_tiled).abs().max().item()}"
        )
        # v2 uses split-K so small FP diffs are expected
        assert torch.allclose(out_flat.float(), out_v2.float(), rtol=0.1, atol=1.0), (
            f"MISMATCH flat vs v2 {name} k={k}: max diff = {(out_flat.float() - out_v2.float()).abs().max().item()}"
        )
