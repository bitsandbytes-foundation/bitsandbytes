"""Comprehensive kbit kernel benchmark across VLM-relevant M values.

Compares all kbit kernel variants (with Hadamard rotation) against cuBLAS FP16
on Qwen3-Coder-Next 70B shapes at M values spanning decode through VLM image
prefill:

  M=1       autoregressive decode (single user)
  M=4       small batch decode / MoE expert tokens
  M=8,16    multi-user batched decode
  M=32,64   larger batch decode
  M=128+    VLM image token prefill (256-2880 patches per image)

Kernel dispatch (matching kbit_linear):
  M <= 4:   scalar GEMV (tiled layout)
  M 5-16:   fused dequant + MMA (tensor core)
  M > 16:   dequantize to fp16 + cuBLAS matmul

Timing methodology:
  CUDA graph capture + batched replay. Each measurement replays the graph
  INNER times within a single event-timed region, then divides. This
  amortizes the ~14 us per-replay overhead to ~2 us/iter, revealing true
  kernel execution times. Median of OUTER measurements is reported.

Usage:
  python benchmarks/bench_kbit_vlm.py
  python benchmarks/bench_kbit_vlm.py --inner 1000 --outer 30   # higher accuracy
  python benchmarks/bench_kbit_vlm.py --k 4                     # single k value
  python benchmarks/bench_kbit_vlm.py --m 1,4,8                 # subset of M values
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

# Qwen3-Coder-Next 70B dense layer shapes (K_dim, N, label)
SHAPES = [
    (2048, 5120, "gate_proj"),
    (5120, 2048, "down_proj"),
    (2048, 4096, "q_proj"),
    (4096, 2048, "o_proj"),
    (2048, 512, "kv_proj"),
]

# VLM-relevant M values
ALL_M_VALUES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

ALL_K_VALUES = [2, 3, 4, 5]

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
    # Warm up (uncaptured)
    for _ in range(30):
        fn()
    torch.cuda.synchronize()

    # Capture on a side stream
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

    # Warm up replay
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()

    # Timed measurements
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


def try_bench(fn, inner: int, outer: int):
    """bench() wrapped in a try/except — returns None on failure."""
    try:
        return bench(fn, inner, outer)
    except Exception:
        return None


def fmt(val):
    """Format a time value or None as a fixed-width string."""
    if val is None:
        return "    ---"
    return f"{val:>6.1f}"


def prepare_kbit_weights(K_dim, N, k, codebook):
    """Quantize random weights and repack to tiled layout."""
    W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
    packed, absmax, _ = quantize_kbit(W, k=k, codebook=codebook)
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(packed, absmax, K_dim, N, k)
    return packed, absmax, packed_tiled, absmax_tiled


def run_benchmarks(m_values, k_values, inner, outer):
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Timing: batched graph replay ({inner} replays/measurement, median of {outer})")
    print(f"Rotation: Hadamard block_size={ROTATION_BLOCK_SIZE}")
    print(f"M values: {m_values}")
    print(f"k values: {k_values}")
    print()

    for K_dim, N, layer_label in SHAPES:
        print()
        print("=" * 130)
        print(f"  {layer_label}  (K={K_dim}, N={N})    [all times in us]")
        print("=" * 130)

        for k in k_values:
            cb = create_normal_float_codebook(k)
            packed, absmax, packed_tiled, absmax_tiled = prepare_kbit_weights(K_dim, N, k, cb)

            print(f"\n  k={k}:")
            print(
                f"  {'M':>6} | {'cuBLAS':>8} | "
                f"{'Scalar':>8} {'Tiled':>8} {'MMA':>8} {'DQ+cuB':>8} | "
                f"{'R+Tiled':>8} {'R+MMA':>8} {'R+DQ+C':>8} | "
                f"{'best kbit':>12} {'speedup':>8}"
            )
            print(f"  {'-' * 118}")

            for M in m_values:
                # --- cuBLAS FP16 baseline ---
                A_cb = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
                W_cb = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
                out_cb = torch.empty(M, N, dtype=torch.float16, device="cuda")
                t_cublas = bench(lambda: torch.mm(A_cb, W_cb.t(), out=out_cb), inner, outer)

                A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
                A_rot = A.clone()

                t_sc = t_tiled = t_mma = t_dq = None
                t_rtiled = t_rmma = t_rdq = None

                # --- Scalar GEMV (M <= 4) ---
                if M <= 4:
                    out_s = torch.zeros(M, N, dtype=torch.float16, device="cuda")
                    t_sc = try_bench(
                        lambda: torch.ops.bitsandbytes.kbit_scalar_gemv(A, packed, absmax, cb, K_dim, N, k),
                        inner,
                        outer,
                    )
                    t_tiled = try_bench(
                        lambda: torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                            A, packed_tiled, absmax_tiled, cb, K_dim, N, k, out_s
                        ),
                        inner,
                        outer,
                    )

                    def pipe_tiled():
                        hadamard_rotate(A_rot, block_size=ROTATION_BLOCK_SIZE)
                        torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                            A_rot, packed_tiled, absmax_tiled, cb, K_dim, N, k, out_s
                        )

                    t_rtiled = try_bench(pipe_tiled, inner, outer)

                # --- MMA GEMM (M <= 64) ---
                if M <= 64:
                    TILE_M, TILE_N = 16, 64
                    m_tiles = (M + TILE_M - 1) // TILE_M
                    n_tiles = max(1, N // TILE_N)
                    out_mma = torch.zeros(M, N, dtype=torch.float16, device="cuda")
                    ws_mma = torch.zeros(M, N, dtype=torch.float32, device="cuda")
                    tc_mma = torch.zeros(m_tiles * n_tiles, dtype=torch.int32, device="cuda")

                    def run_mma():
                        ws_mma.zero_()
                        tc_mma.zero_()
                        torch.ops.bitsandbytes.kbit_gemm_prod_(
                            A, packed, absmax, cb, K_dim, N, k, 1, out_mma, ws_mma, tc_mma
                        )

                    t_mma = try_bench(run_mma, inner, outer)

                    def pipe_mma():
                        hadamard_rotate(A_rot, block_size=ROTATION_BLOCK_SIZE)
                        ws_mma.zero_()
                        tc_mma.zero_()
                        torch.ops.bitsandbytes.kbit_gemm_prod_(
                            A_rot, packed, absmax, cb, K_dim, N, k, 1, out_mma, ws_mma, tc_mma
                        )

                    t_rmma = try_bench(pipe_mma, inner, outer)

                # --- Dequant + cuBLAS (any M) ---
                dq_buf = torch.empty(N * K_dim, dtype=torch.float16, device="cuda")
                out_dq = torch.empty(M, N, dtype=torch.float16, device="cuda")

                def run_dq():
                    torch.ops.bitsandbytes.dequantize_kbit_tiled_(
                        packed_tiled, cb, absmax_tiled, k, K_dim, N, torch.float16, dq_buf
                    )
                    torch.mm(A, dq_buf.view(N, K_dim).t(), out=out_dq)

                t_dq = try_bench(run_dq, inner, outer)

                def pipe_dq():
                    hadamard_rotate(A_rot, block_size=ROTATION_BLOCK_SIZE)
                    torch.ops.bitsandbytes.dequantize_kbit_tiled_(
                        packed_tiled, cb, absmax_tiled, k, K_dim, N, torch.float16, dq_buf
                    )
                    torch.mm(A_rot, dq_buf.view(N, K_dim).t(), out=out_dq)

                t_rdq = try_bench(pipe_dq, inner, outer)

                # --- Find best with rotation ---
                candidates = {}
                if t_rtiled is not None:
                    candidates["R+Tiled"] = t_rtiled
                if t_rmma is not None:
                    candidates["R+MMA"] = t_rmma
                if t_rdq is not None:
                    candidates["R+DQ+C"] = t_rdq

                if candidates:
                    best_name = min(candidates, key=candidates.get)
                    best_time = candidates[best_name]
                    speedup = t_cublas / best_time
                    best_str = f"{best_time:>6.1f}({best_name:>6})"
                    sp_str = f"{speedup:>6.2f}x"
                else:
                    best_str = "       ---"
                    sp_str = "     ---"

                print(
                    f"  {M:>6} | {t_cublas:>7.1f}u | "
                    f"{fmt(t_sc)}u {fmt(t_tiled)}u {fmt(t_mma)}u {fmt(t_dq)}u | "
                    f"{fmt(t_rtiled)}u {fmt(t_rmma)}u {fmt(t_rdq)}u | "
                    f"{best_str} {sp_str}"
                )


def main():
    parser = argparse.ArgumentParser(description="kbit kernel benchmark (VLM M sweep)")
    parser.add_argument("--inner", type=int, default=500, help="Graph replays per measurement (default: 500)")
    parser.add_argument("--outer", type=int, default=15, help="Measurements per benchmark (default: 15)")
    parser.add_argument("--k", type=str, default=None, help="Comma-separated k values (default: 2,3,4,5)")
    parser.add_argument("--m", type=str, default=None, help="Comma-separated M values (default: 1,4,8,...,1024)")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k.split(",")] if args.k else ALL_K_VALUES
    m_values = [int(x) for x in args.m.split(",")] if args.m else ALL_M_VALUES

    run_benchmarks(m_values, k_values, args.inner, args.outer)


if __name__ == "__main__":
    main()
