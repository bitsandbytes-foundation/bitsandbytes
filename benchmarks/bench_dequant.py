"""Dequant + cuBLAS overhead analysis.

Measures dequantize_kbit GPU kernel time per shape×k (via ncu or --use-events),
fp16 matmul time per shape×M, and computes the overhead ratio.

Usage:
  # Recommended: ncu for dequant (accurate), CUDA events for matmul
  bash benchmarks/bench_dequant.sh

  # Quick (CUDA events only, includes ~35us dispatch overhead on dequant):
  python benchmarks/bench_dequant.py --use-events

Env: M_VALS (default "4,8,16,32,64,128,256,512,1024,2048,4096")
     DEQUANT_CSV: comma-separated dequant times injected by bench_dequant.sh
                  (order: k=2 × 5 shapes, k=3 × 5, k=4 × 5, k=5 × 5)
"""
import os, sys, argparse

for p in [".", ".."]:
    if os.path.isdir(os.path.join(p, "bitsandbytes")):
        sys.path.insert(0, os.path.abspath(p))
        break

import torch
import bitsandbytes  # noqa: E402
from bitsandbytes.functional import create_normal_float_codebook  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--use-events", action="store_true",
                    help="Use CUDA events for dequant timing (includes dispatch overhead)")
args = parser.parse_args()

shapes = [
    ("gateup", 2048, 5120),
    ("down",   5120, 2048),
    ("Q",      2048, 4096),
    ("O",      4096, 2048),
    ("KV",     2048,  512),
]
k_bits_list = [2, 3, 4, 5]
m_vals = [int(x) for x in os.environ.get(
    "M_VALS", "4,8,16,32,64,128,256,512,1024,2048,4096").split(",")]

dev = torch.device("cuda")
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)
WARMUP = 50
ITERS = 200

# --- Dequant times ---
dequant_us = {}
dequant_env = os.environ.get("DEQUANT_CSV", "")
if dequant_env:
    # Injected by bench_dequant.sh (ncu-measured)
    # Order: k=2 × 5 shapes, k=3 × 5, k=4 × 5, k=5 × 5
    vals = [float(x) for x in dequant_env.split(",")]
    i = 0
    for k in k_bits_list:
        for name, _, _ in shapes:
            dequant_us[(name, k)] = vals[i]
            i += 1
elif args.use_events:
    # Fallback: CUDA events (includes ~35us dispatch overhead)
    for k in k_bits_list:
        codebook = create_normal_float_codebook(k, device=dev)
        for name, K_dim, N in shapes:
            n_elements = K_dim * N
            W = torch.randn(n_elements, device=dev, dtype=torch.float32)
            packed, absmax = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
            for _ in range(WARMUP):
                torch.ops.bitsandbytes.dequantize_kbit(
                    packed, codebook, absmax, k, n_elements, torch.float16)
            torch.cuda.synchronize()
            start_ev.record()
            for _ in range(ITERS):
                torch.ops.bitsandbytes.dequantize_kbit(
                    packed, codebook, absmax, k, n_elements, torch.float16)
            end_ev.record()
            torch.cuda.synchronize()
            dequant_us[(name, k)] = start_ev.elapsed_time(end_ev) * 1000 / ITERS
else:
    print("ERROR: Run via bench_dequant.sh (ncu) or with --use-events", file=sys.stderr)
    sys.exit(1)

# --- Print dequant times ---
print("=== Dequant kernel time (us) ===")
print(f"{'shape':<8}", end="")
for k in k_bits_list:
    print(f" {'k='+str(k):>8}", end="")
print()
print("---")
for name, _, _ in shapes:
    print(f"{name:<8}", end="")
    for k in k_bits_list:
        print(f" {dequant_us[(name, k)]:>8.1f}", end="")
    print()
print()

# --- Measure fp16 matmul time per shape×M ---
matmul_us = {}
for name, K_dim, N in shapes:
    W = torch.randn(K_dim, N, dtype=torch.float16, device=dev)
    for M in m_vals:
        A = torch.randn(M, K_dim, dtype=torch.float16, device=dev)
        out = torch.empty(M, N, dtype=torch.float16, device=dev)
        for _ in range(WARMUP):
            torch.mm(A, W, out=out)
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(ITERS):
            torch.mm(A, W, out=out)
        end_ev.record()
        torch.cuda.synchronize()
        matmul_us[(name, M)] = start_ev.elapsed_time(end_ev) * 1000 / ITERS

# --- Print combined table per k ---
for k in k_bits_list:
    print(f"=== k={k}: dequant + fp16 matmul overhead ===")
    print(f"{'shape':<8} {'M':>6} {'fp16 (us)':>10} {'dequant (us)':>13} {'total (us)':>11} {'speed':>7}")
    print("-" * 60)
    for name, K_dim, N in shapes:
        d = dequant_us[(name, k)]
        for M in m_vals:
            mm = matmul_us[(name, M)]
            total = d + mm
            speed = mm / total
            print(f"{name:<8} {M:>6} {mm:>10.1f} {d:>13.1f} {total:>11.1f} {speed:>7.2f}")
        print()
    print()
