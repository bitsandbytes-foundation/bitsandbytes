"""Benchmark for Hadamard rotation kernel and kbit M=1 pipeline.

Measures:
1. Rotation standalone: all block sizes x Qwen3 K values x M=1,4
2. Full pipeline (rotate + kbit_scalar_gemv_tiled): Qwen3 dense shapes at M=1, k=2..5
3. cuBLAS FP16 baseline: same shapes
4. Speedup table: pipeline vs cuBLAS

Timing methodology:
  CUDA graph capture + batched replay. Each measurement replays the graph
  INNER times within a single event-timed region, then divides. This
  amortizes the ~14 us per-replay overhead down to negligible levels,
  revealing true kernel execution times. Median of OUTER measurements.

Usage:
  python benchmarks/bench_hadamard.py
  python benchmarks/bench_hadamard.py --inner 1000 --outer 30   # higher accuracy
"""

import argparse
import sys

import torch

sys.path.insert(0, ".")
from scipy.stats import norm

from bitsandbytes import _ops  # noqa: F401
from bitsandbytes.functional import (
    hadamard_rotate,
    quantize_kbit,
)

ROTATION_BLOCK_SIZE = 64


def create_normal_float_codebook(k: int) -> torch.Tensor:
    n_levels = 1 << k
    quantiles = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    values = torch.tensor(norm.ppf(quantiles.numpy()), dtype=torch.float32)
    values = values / values.abs().max()
    return values.cuda()


def bench(fn, inner: int, outer: int) -> float:
    """Batched CUDA graph replay timing. Returns median us per iteration.

    Captures fn into a CUDA graph, then replays it `inner` times within a
    single CUDA event pair. The per-replay overhead (~14 us on RTX 4090)
    is amortized to ~14/inner us per iteration. Takes the median of `outer`
    such measurements.
    """
    for _ in range(30):
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()

    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(outer):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / inner)  # ms -> us/iter
    times.sort()
    return times[len(times) // 2]


def prepare_kbit_weights(K_dim, N, k):
    """Quantize random weights and repack for tiled access."""
    W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
    codebook = create_normal_float_codebook(k)
    packed, absmax, _ = quantize_kbit(W, k=k, codebook=codebook)
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(packed, absmax, K_dim, N, k)
    return packed_tiled, absmax_tiled, codebook


def bench_rotation_standalone(inner, outer):
    """Benchmark rotation kernel standalone across block sizes and shapes."""
    print("=" * 70)
    print("1. ROTATION STANDALONE")
    print("=" * 70)
    print(f"{'M':>4} {'K':>6} {'BS':>4} {'Time (us)':>10} {'BW (GB/s)':>10}")
    print("-" * 40)

    block_sizes = [32, 64, 128, 256]
    k_values = [512, 2048, 4096, 5120]
    m_values = [1, 4]

    for M in m_values:
        for K in k_values:
            for bs in block_sizes:
                A = torch.randn(M, K, dtype=torch.float16, device="cuda")
                t = bench(lambda: hadamard_rotate(A, block_size=bs), inner, outer)
                bw = 2 * A.numel() * 2 / (t / 1e6) / 1e9
                print(f"{M:>4} {K:>6} {bs:>4} {t:>10.3f} {bw:>10.1f}")
        print()


def bench_pipeline(inner, outer):
    """Benchmark full pipeline: rotate(A) + kbit_scalar_gemv."""
    print("=" * 70)
    print("2. FULL PIPELINE: rotate + kbit_scalar_gemv_tiled")
    print("=" * 70)
    print(f"{'M':>4} {'K':>6} {'N':>6} {'k':>2} {'Rotate(us)':>11} {'GEMV(us)':>9} {'Total(us)':>10} {'TFLOPS':>7}")
    print("-" * 65)

    shapes = [
        (1, 2048, 5120, "gate/up"),
        (1, 5120, 2048, "down"),
        (1, 2048, 4096, "Q proj"),
        (1, 4096, 2048, "O proj"),
        (1, 2048, 512, "KV proj"),
        (4, 2048, 5120, "gate/up M=4"),
        (4, 5120, 2048, "down M=4"),
    ]

    for k in [2, 3, 4, 5]:
        print(f"\n--- k={k} ---")
        for M, K_dim, N, label in shapes:
            packed_tiled, absmax_tiled, codebook = prepare_kbit_weights(K_dim, N, k)
            A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

            A_copy = A.clone()
            t_rot = bench(lambda: hadamard_rotate(A_copy, block_size=ROTATION_BLOCK_SIZE), inner, outer)

            out = torch.zeros(M, N, dtype=torch.float16, device="cuda")
            t_gemv = bench(
                lambda: torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                    A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, out
                ),
                inner,
                outer,
            )

            def pipeline():
                hadamard_rotate(A_copy, block_size=ROTATION_BLOCK_SIZE)
                torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                    A_copy, packed_tiled, absmax_tiled, codebook, K_dim, N, k, out
                )

            t_total = bench(pipeline, inner, outer)

            flops = 2 * M * K_dim * N
            tflops = flops / (t_total / 1e6) / 1e12
            print(
                f"{M:>4} {K_dim:>6} {N:>6} {k:>2} {t_rot:>11.3f} {t_gemv:>9.3f} "
                f"{t_total:>10.3f} {tflops:>7.3f}  {label}"
            )


def bench_cublas_baseline(inner, outer):
    """Benchmark cuBLAS FP16 GEMM for the same shapes."""
    print("\n" + "=" * 70)
    print("3. cuBLAS FP16 BASELINE")
    print("=" * 70)
    print(f"{'M':>4} {'K':>6} {'N':>6} {'Time(us)':>9} {'TFLOPS':>7}")
    print("-" * 40)

    shapes = [
        (1, 2048, 5120),
        (1, 5120, 2048),
        (1, 2048, 4096),
        (1, 4096, 2048),
        (1, 2048, 512),
        (4, 2048, 5120),
        (4, 5120, 2048),
    ]

    for M, K_dim, N in shapes:
        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        out = torch.empty(M, N, dtype=torch.float16, device="cuda")

        t = bench(lambda: torch.mm(A, W.t(), out=out), inner, outer)
        flops = 2 * M * K_dim * N
        tflops = flops / (t / 1e6) / 1e12
        print(f"{M:>4} {K_dim:>6} {N:>6} {t:>9.3f} {tflops:>7.3f}")


def bench_speedup_table(inner, outer):
    """Print a speedup comparison table: pipeline vs cuBLAS."""
    print("\n" + "=" * 70)
    print("4. SPEEDUP TABLE: Rot + kbit GEMV vs cuBLAS FP16")
    print("=" * 70)

    shapes = [
        (1, 2048, 5120, "gate/up"),
        (1, 5120, 2048, "down"),
        (1, 2048, 4096, "Q proj"),
        (1, 4096, 2048, "O proj"),
        (4, 2048, 5120, "gate/up M=4"),
        (4, 5120, 2048, "down M=4"),
    ]

    print(f"{'Shape':>20} {'k':>2} {'Pipeline(us)':>13} {'cuBLAS(us)':>11} {'Speedup':>8}")
    print("-" * 65)

    for k in [2, 3, 4, 5]:
        print(f"\n--- k={k} ---")
        for M, K_dim, N, label in shapes:
            packed_tiled, absmax_tiled, codebook = prepare_kbit_weights(K_dim, N, k)
            A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
            W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
            out = torch.zeros(M, N, dtype=torch.float16, device="cuda")
            A_copy = A.clone()

            def pipeline():
                hadamard_rotate(A_copy, block_size=ROTATION_BLOCK_SIZE)
                torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                    A_copy, packed_tiled, absmax_tiled, codebook, K_dim, N, k, out
                )

            t_pipe = bench(pipeline, inner, outer)
            t_cublas = bench(lambda: torch.mm(A, W.t(), out=out), inner, outer)

            speedup = t_cublas / t_pipe
            shape_str = f"{M}x{K_dim}x{N}"
            print(f"{shape_str:>20} {k:>2} {t_pipe:>13.3f} {t_cublas:>11.3f} {speedup:>7.2f}x  {label}")


def main():
    parser = argparse.ArgumentParser(description="Hadamard rotation + kbit M=1 pipeline benchmark")
    parser.add_argument("--inner", type=int, default=500, help="Graph replays per measurement (default: 500)")
    parser.add_argument("--outer", type=int, default=15, help="Measurements per benchmark (default: 15)")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Timing: batched graph replay ({args.inner} replays/measurement, median of {args.outer})")
    print()

    bench_rotation_standalone(args.inner, args.outer)
    bench_pipeline(args.inner, args.outer)
    bench_cublas_baseline(args.inner, args.outer)
    bench_speedup_table(args.inner, args.outer)


if __name__ == "__main__":
    main()
