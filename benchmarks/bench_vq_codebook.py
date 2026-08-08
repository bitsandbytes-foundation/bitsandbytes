"""Benchmark: VQ codebook production kernels.

Compares production VQ kernels against kbit baseline and cuBLAS:
1. vq_scalar_gemv_tiled (M=1-4, p=2 and p=4)
2. vq_gemm_prod MMA kernel (M=5-16, p=2 and p=4)
3. dequant+cuBLAS fallback (M=32, p=2 and p=4)
4. kbit_scalar_gemv_tiled (M=1-4, k=4 bit-plane baseline)
5. kbit_gemm_prod MMA (M=5-16, k=4)
6. cuBLAS fp16 (dense baseline)

Timing: CUDA graph capture + batched replay.
Output: JSON results + human-readable tables.

Usage:
  cd /path/to/bnb-kbit-gemm
  python benchmarks/bench_vq_codebook.py
  python benchmarks/bench_vq_codebook.py --inner 1000 --outer 30
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, ".")
from bitsandbytes import _ops  # noqa: F401
from bitsandbytes.functional import (
    create_normal_float_codebook,
    create_vq_codebook,
    quantize_kbit,
    quantize_vq,
    repack_vq,
)


# ---- Timing utility (CUDA graph replay, same as bench_hadamard.py) ----

def bench(fn, inner: int, outer: int) -> float:
    """Batched CUDA graph replay timing. Returns median us per iteration."""
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


# ---- Weight preparation ----

def prepare_vq_weights(K_dim, N, p, dtype=torch.float16):
    """Quantize and repack random weights using production VQ kernels."""
    dev = torch.device("cuda")
    codebook = create_vq_codebook(p, device=dev)
    W = torch.randn(N, K_dim, dtype=dtype, device=dev)
    packed_flat, absmax_flat, codebook = quantize_vq(W, p=p, codebook=codebook)
    packed_tiled, absmax_tiled = repack_vq(packed_flat, absmax_flat, K_dim, N, p)
    return packed_tiled, absmax_tiled, codebook, packed_flat, absmax_flat


def prepare_kbit_weights(K_dim, N, k=4):
    """Quantize and repack random weights using kbit kernels."""
    dev = torch.device("cuda")
    codebook = create_normal_float_codebook(k, device=dev)
    W = torch.randn(N, K_dim, dtype=torch.float16, device=dev)
    packed_flat, absmax_flat, codebook = quantize_kbit(W, k=k, codebook=codebook)
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
        packed_flat, absmax_flat, K_dim, N, k)
    return packed_tiled, absmax_tiled, codebook


# ---- Qwen3 shapes ----

QWEN3_SHAPES = [
    (2048, 5120, "gate/up"),
    (5120, 2048, "down"),
    (2048, 4096, "Q proj"),
    (4096, 2048, "O proj"),
    (2048, 512, "KV proj"),
]


# ---- Benchmark functions ----

def bench_scalar_gemv(inner, outer):
    """Benchmark scalar GEMV at M=1 for p=2, p=4, kbit k=4, and cuBLAS."""
    results = []

    print("=" * 80)
    print("SCALAR GEMV (M=1)")
    print("=" * 80)
    print(f"{'Method':>18} {'K':>5} {'N':>5} {'Time(us)':>9} {'TFLOPS':>7}  Label")
    print("-" * 65)

    for K_dim, N, label in QWEN3_SHAPES:
        M = 1
        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
        flops = 2 * M * K_dim * N

        # VQ p=2 (4 bits/wt) scalar GEMV
        pt2, at2, cb2, _, _ = prepare_vq_weights(K_dim, N, p=2)
        out2 = torch.zeros(M, N, dtype=torch.float16, device="cuda")
        t = bench(lambda: torch.ops.bitsandbytes.vq_scalar_gemv_tiled_(
            A, pt2, at2, cb2, K_dim, N, 2, out2), inner, outer)
        tflops = flops / (t / 1e6) / 1e12
        print(f"{'vq_p2':>18} {K_dim:>5} {N:>5} {t:>9.3f} {tflops:>7.3f}  {label}")
        results.append(dict(method="vq_p2", kernel="scalar", M=M, K=K_dim,
                            N=N, time_us=round(t, 3), tflops=round(tflops, 4), label=label))

        # VQ p=4 (2 bits/wt) scalar GEMV
        pt4, at4, cb4, _, _ = prepare_vq_weights(K_dim, N, p=4)
        out4 = torch.zeros(M, N, dtype=torch.float16, device="cuda")
        t = bench(lambda: torch.ops.bitsandbytes.vq_scalar_gemv_tiled_(
            A, pt4, at4, cb4, K_dim, N, 4, out4), inner, outer)
        tflops = flops / (t / 1e6) / 1e12
        print(f"{'vq_p4':>18} {K_dim:>5} {N:>5} {t:>9.3f} {tflops:>7.3f}  {label}")
        results.append(dict(method="vq_p4", kernel="scalar", M=M, K=K_dim,
                            N=N, time_us=round(t, 3), tflops=round(tflops, 4), label=label))

        # Kbit k=4 scalar GEMV baseline
        pt_k, at_k, cb_k = prepare_kbit_weights(K_dim, N, k=4)
        out_k = torch.zeros(M, N, dtype=torch.float16, device="cuda")
        t = bench(lambda: torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
            A, pt_k, at_k, cb_k, K_dim, N, 4, out_k), inner, outer)
        tflops = flops / (t / 1e6) / 1e12
        print(f"{'kbit_k4':>18} {K_dim:>5} {N:>5} {t:>9.3f} {tflops:>7.3f}  {label}")
        results.append(dict(method="kbit_k4", kernel="scalar", M=M, K=K_dim,
                            N=N, time_us=round(t, 3), tflops=round(tflops, 4), label=label))

        # cuBLAS fp16 baseline
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        out_c = torch.empty(M, N, dtype=torch.float16, device="cuda")
        t = bench(lambda: torch.mm(A, W.t(), out=out_c), inner, outer)
        tflops = flops / (t / 1e6) / 1e12
        print(f"{'cublas_fp16':>18} {K_dim:>5} {N:>5} {t:>9.3f} {tflops:>7.3f}  {label}")
        results.append(dict(method="cublas_fp16", kernel="dense", M=M, K=K_dim,
                            N=N, time_us=round(t, 3), tflops=round(tflops, 4), label=label))

        print()

    return results


def bench_mma(inner, outer):
    """Benchmark MMA kernels at M=5, 8, 16 for VQ (p=2, p=4), kbit k=4, and cuBLAS."""
    results = []

    m_vals = [5, 8, 16]
    # Use a subset of shapes for MMA
    mma_shapes = [
        (2048, 5120, "gate/up"),
        (5120, 2048, "down"),
        (2048, 4096, "Q proj"),
    ]

    print("=" * 80)
    print("MMA KERNEL (M=5,8,16)")
    print("=" * 80)
    print(f"{'Method':>18} {'M':>3} {'K':>5} {'N':>5} {'Time(us)':>9} {'TFLOPS':>7}  Label")
    print("-" * 68)

    for K_dim, N, label in mma_shapes:
        # Pre-quantize weights once
        pt2, at2, cb2, _, _ = prepare_vq_weights(K_dim, N, p=2)
        pt4, at4, cb4, _, _ = prepare_vq_weights(K_dim, N, p=4)
        pt_k, at_k, cb_k = prepare_kbit_weights(K_dim, N, k=4)
        W_dense = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")

        for M in m_vals:
            A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
            flops = 2 * M * K_dim * N

            # VQ p=2 MMA
            t = bench(lambda: torch.ops.bitsandbytes.vq_gemm_prod(
                A, pt2, at2, cb2, K_dim, N, 2, 1), inner, outer)
            tflops = flops / (t / 1e6) / 1e12
            print(f"{'vq_mma_p2':>18} {M:>3} {K_dim:>5} {N:>5} {t:>9.3f} {tflops:>7.3f}  {label}")
            results.append(dict(method="vq_mma_p2", kernel="mma", M=M, K=K_dim,
                                N=N, time_us=round(t, 3), tflops=round(tflops, 4), label=label))

            # VQ p=4 MMA
            t = bench(lambda: torch.ops.bitsandbytes.vq_gemm_prod(
                A, pt4, at4, cb4, K_dim, N, 4, 1), inner, outer)
            tflops = flops / (t / 1e6) / 1e12
            print(f"{'vq_mma_p4':>18} {M:>3} {K_dim:>5} {N:>5} {t:>9.3f} {tflops:>7.3f}  {label}")
            results.append(dict(method="vq_mma_p4", kernel="mma", M=M, K=K_dim,
                                N=N, time_us=round(t, 3), tflops=round(tflops, 4), label=label))

            # Kbit k=4 MMA
            t = bench(lambda: torch.ops.bitsandbytes.kbit_gemm_prod(
                A, pt_k, at_k, cb_k, K_dim, N, 4, 1), inner, outer)
            tflops = flops / (t / 1e6) / 1e12
            print(f"{'kbit_mma_k4':>18} {M:>3} {K_dim:>5} {N:>5} {t:>9.3f} {tflops:>7.3f}  {label}")
            results.append(dict(method="kbit_mma_k4", kernel="mma", M=M, K=K_dim,
                                N=N, time_us=round(t, 3), tflops=round(tflops, 4), label=label))

            # cuBLAS fp16
            out_c = torch.empty(M, N, dtype=torch.float16, device="cuda")
            t = bench(lambda: torch.mm(A, W_dense.t(), out=out_c), inner, outer)
            tflops = flops / (t / 1e6) / 1e12
            print(f"{'cublas_fp16':>18} {M:>3} {K_dim:>5} {N:>5} {t:>9.3f} {tflops:>7.3f}  {label}")
            results.append(dict(method="cublas_fp16", kernel="dense", M=M, K=K_dim,
                                N=N, time_us=round(t, 3), tflops=round(tflops, 4), label=label))

            print()

    return results


def bench_dequant_cublas(inner, outer):
    """Benchmark dequant+cuBLAS fallback at M=32 for VQ p=2 and p=4."""
    results = []
    M = 32

    dequant_shapes = [
        (2048, 5120, "gate/up"),
        (5120, 2048, "down"),
    ]

    print("=" * 80)
    print(f"DEQUANT + cuBLAS (M={M})")
    print("=" * 80)
    print(f"{'Method':>18} {'K':>5} {'N':>5} {'Time(us)':>9} {'TFLOPS':>7}  Label")
    print("-" * 65)

    for K_dim, N, label in dequant_shapes:
        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
        flops = 2 * M * K_dim * N

        for p, method_name in [(2, "vq_dequant_p2"), (4, "vq_dequant_p4")]:
            pt, at, cb, _, _ = prepare_vq_weights(K_dim, N, p=p)
            # Full vq_linear dispatch with workspace (M=32 goes through dequant+cuBLAS)
            from bitsandbytes.functional import vq_linear, vq_linear_workspace
            out_vq = torch.empty(M, N, dtype=torch.float16, device="cuda")
            ws = vq_linear_workspace(M, K_dim, N, p, torch.float16, torch.device("cuda"))
            t = bench(lambda: vq_linear(A, pt, at, cb, p, K_dim, N, out=out_vq, workspace=ws), inner, outer)
            tflops = flops / (t / 1e6) / 1e12
            print(f"{method_name:>18} {K_dim:>5} {N:>5} {t:>9.3f} {tflops:>7.3f}  {label}")
            results.append(dict(method=method_name, kernel="dequant+cublas", M=M,
                                K=K_dim, N=N, time_us=round(t, 3), tflops=round(tflops, 4), label=label))

        # cuBLAS fp16 baseline
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        out_c = torch.empty(M, N, dtype=torch.float16, device="cuda")
        t = bench(lambda: torch.mm(A, W.t(), out=out_c), inner, outer)
        tflops = flops / (t / 1e6) / 1e12
        print(f"{'cublas_fp16':>18} {K_dim:>5} {N:>5} {t:>9.3f} {tflops:>7.3f}  {label}")
        results.append(dict(method="cublas_fp16", kernel="dense", M=M, K=K_dim,
                            N=N, time_us=round(t, 3), tflops=round(tflops, 4), label=label))

        print()

    return results


def print_speedup_summary(results):
    """Print speedup tables: VQ vs kbit and vs cuBLAS."""
    print("=" * 80)
    print("SPEEDUP SUMMARY")
    print("=" * 80)

    # Group by (M, K, N) so quantized and cuBLAS are in the same group
    from collections import defaultdict
    groups = defaultdict(dict)
    for r in results:
        key = (r["M"], r["K"], r["N"])
        groups[key][r["method"]] = r

    print(f"{'M':>3} {'Shape':>10} {'Method':>18} {'us':>8} {'vs cuBLAS':>10} {'vs kbit':>10}")
    print("-" * 65)

    for key in sorted(groups.keys()):
        M, K, N = key
        methods = groups[key]
        t_cublas = methods.get("cublas_fp16", {}).get("time_us", 1)

        # Find kbit baseline
        kbit_key = None
        for mk in methods:
            if "kbit" in mk:
                kbit_key = mk
                break
        t_kbit = methods.get(kbit_key, {}).get("time_us", 1) if kbit_key else None

        for method_name in sorted(methods.keys()):
            r = methods[method_name]
            vs_cublas = t_cublas / r["time_us"] if r["time_us"] > 0 else 0
            vs_kbit_str = ""
            if t_kbit is not None and r["time_us"] > 0:
                vs_kbit = t_kbit / r["time_us"]
                vs_kbit_str = f"{vs_kbit:>9.2f}x"
            else:
                vs_kbit_str = f"{'---':>10}"

            shape_str = f"{K}x{N}"
            print(f"{M:>3} {shape_str:>10} {method_name:>18} {r['time_us']:>8.3f}"
                  f" {vs_cublas:>9.2f}x {vs_kbit_str}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="VQ production kernel benchmark")
    parser.add_argument("--inner", type=int, default=500,
                        help="Graph replays per measurement (default: 500)")
    parser.add_argument("--outer", type=int, default=15,
                        help="Measurements per benchmark (default: 15)")
    parser.add_argument("--output", type=str, default="results/vq_bench.json",
                        help="JSON output path (default: results/vq_bench.json)")
    parser.add_argument("--scalar-only", action="store_true",
                        help="Only run scalar GEMV benchmarks")
    parser.add_argument("--mma-only", action="store_true",
                        help="Only run MMA benchmarks")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Timing: {args.inner} replays/measurement, median of {args.outer}")
    print()

    all_results = []

    if not args.mma_only:
        all_results.extend(bench_scalar_gemv(args.inner, args.outer))

    if not args.scalar_only:
        all_results.extend(bench_mma(args.inner, args.outer))
        all_results.extend(bench_dequant_cublas(args.inner, args.outer))

    print_speedup_summary(all_results)

    # Save JSON
    output = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "inner": args.inner,
        "outer": args.outer,
        "results": all_results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
