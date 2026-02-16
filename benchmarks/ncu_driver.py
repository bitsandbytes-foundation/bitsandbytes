"""ncu kernel driver — runs all shape×k×M configs in a single process.

Used by bench_ncu.sh. Env vars:
  KERNEL: "mma" or "scalar"
  M_VALS: comma-separated M values (default "1,2,3,4,8")

Each config runs WARMUP + PROFILED kernel launches. ncu captures all
matching launches; the sweep script skips warmup and averages profiled.
"""
import os, sys, torch

# Allow running from repo root or benchmarks/
for p in [".", ".."]:
    if os.path.isdir(os.path.join(p, "bitsandbytes")):
        sys.path.insert(0, os.path.abspath(p))
        break

import bitsandbytes  # noqa: E402
from bitsandbytes.functional import create_normal_float_codebook  # noqa: E402

KERNEL = os.environ.get("KERNEL", "mma")
m_vals = [int(x) for x in os.environ.get("M_VALS", "1,2,3,4,8").split(",")]

shapes = [
    ("gateup", 2048, 5120),
    ("down",   5120, 2048),
    ("Q",      2048, 4096),
    ("O",      4096, 2048),
    ("KV",     2048,  512),
]
k_bits_list = [2, 3, 4, 5]
WARMUP = 5
PROFILED = 5

dev = torch.device("cuda")

# Pre-quantize all shape×k combos on GPU (fast)
data = {}
for name, K_dim, N in shapes:
    for k in k_bits_list:
        codebook = create_normal_float_codebook(k, device=dev)
        W = torch.randn(K_dim * N, device=dev, dtype=torch.float32)
        packed_flat, absmax = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
        packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
            packed_flat, absmax, K_dim, N, k)
        data[(name, k)] = (K_dim, N, packed_tiled, absmax_tiled, codebook)

# Build config list
configs = []
for name, K_dim, N in shapes:
    for k in k_bits_list:
        for M in m_vals:
            configs.append((name, k, M))

# Run all configs: warmup then profiled
for name, k, M in configs:
    K_dim, N, packed_tiled, absmax_tiled, codebook = data[(name, k)]
    A = torch.randn(M, K_dim, dtype=torch.float16, device=dev)

    if KERNEL == "mma":
        fn = lambda: torch.ops.bitsandbytes.kbit_gemm_prod(
            A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, 1)
    else:
        fn = lambda: torch.ops.bitsandbytes.kbit_scalar_gemv(
            A, packed_tiled, absmax_tiled, codebook, K_dim, N, k)

    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    for _ in range(PROFILED):
        fn()
    torch.cuda.synchronize()
