"""ncu VQ kernel driver — runs VQ kernel configs for ncu profiling.

Used by bench_vq_ncu.sh. Env vars:
  KERNEL: "vq_scalar" or "vq_mma"
  P_VAL: VQ dimension (2 or 4)
  M_VALS: comma-separated M values (default "1" for scalar, "5,8,16" for mma)

Each config runs WARMUP + PROFILED kernel launches. ncu captures all
matching launches; the sweep script skips warmup and averages profiled.
"""

import os
import sys

import torch

# Allow running from repo root or benchmarks/
for p in [".", ".."]:
    if os.path.isdir(os.path.join(p, "bitsandbytes")):
        sys.path.insert(0, os.path.abspath(p))
        break

from bitsandbytes import _ops  # noqa: F401
from bitsandbytes.functional import create_vq_codebook, quantize_vq, repack_vq

KERNEL = os.environ.get("KERNEL", "vq_scalar")
P_VAL = int(os.environ.get("P_VAL", "2"))

default_m = "1" if KERNEL == "vq_scalar" else "5,8,16"
m_vals = [int(x) for x in os.environ.get("M_VALS", default_m).split(",")]

# Scalar kernel only supports M<=4
if KERNEL == "vq_scalar":
    m_vals = [m for m in m_vals if m <= 4]

print(f"ACTUAL_M_VALS={','.join(str(m) for m in m_vals)}", file=sys.stderr)

# Dense shapes (same as kbit ncu_driver)
dense_shapes = [
    ("gateup", 2048, 5120),
    ("down", 5120, 2048),
    ("Q", 2048, 4096),
    ("O", 4096, 2048),
    ("KV", 2048, 512),
]

# MMA uses a subset of shapes
mma_shapes = [
    ("gateup", 2048, 5120),
    ("down", 5120, 2048),
    ("Q", 2048, 4096),
]

WARMUP = 5
PROFILED = 5
dev = torch.device("cuda")

shapes = dense_shapes if KERNEL == "vq_scalar" else mma_shapes

# Pre-quantize all shapes
data = {}
codebook = create_vq_codebook(P_VAL, device=dev)
for name, K_dim, N in shapes:
    W = torch.randn(N, K_dim, dtype=torch.float16, device=dev)
    packed_flat, absmax_flat, _ = quantize_vq(W, p=P_VAL, codebook=codebook)
    packed_tiled, absmax_tiled = repack_vq(packed_flat, absmax_flat, K_dim, N, P_VAL)
    data[name] = (K_dim, N, packed_tiled, absmax_tiled)

# Run all configs
for name, K_dim, N in shapes:
    K_dim, N, packed_tiled, absmax_tiled = data[name]
    for M in m_vals:
        A = torch.randn(M, K_dim, dtype=torch.float16, device=dev)

        if KERNEL == "vq_scalar":
            out = torch.zeros(M, N, dtype=torch.float16, device=dev)
            fn = lambda: torch.ops.bitsandbytes.vq_scalar_gemv_tiled_(
                A, packed_tiled, absmax_tiled, codebook, K_dim, N, P_VAL, out)
        else:  # vq_mma
            fn = lambda: torch.ops.bitsandbytes.vq_gemm_prod(
                A, packed_tiled, absmax_tiled, codebook, K_dim, N, P_VAL, 1)

        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        for _ in range(PROFILED):
            fn()
        torch.cuda.synchronize()
