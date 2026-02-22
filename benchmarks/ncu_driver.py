"""ncu kernel driver — runs all shape x k x M configs in a single process.

Used by bench_ncu.sh. Env vars:
  KERNEL: "mma", "scalar", or "grouped_mma"
  M_VALS: comma-separated M values (default "1,2,3,4,5,6,7,8")
  NUM_EXPERTS: number of active experts for grouped_mma kernel (default 8)

Each config runs WARMUP + PROFILED kernel launches. ncu captures all
matching launches; the sweep script skips warmup and averages profiled.

For scalar kernel, M values > 4 are skipped (kernel only supports M<=4).
The script prints the actual M values used to stderr for the shell script.
"""

import os
import sys

import torch

# Allow running from repo root or benchmarks/
for p in [".", ".."]:
    if os.path.isdir(os.path.join(p, "bitsandbytes")):
        sys.path.insert(0, os.path.abspath(p))
        break

from bitsandbytes.functional import create_normal_float_codebook  # noqa: E402

KERNEL = os.environ.get("KERNEL", "mma")
m_vals = [int(x) for x in os.environ.get("M_VALS", "1,2,3,4,5,6,7,8").split(",")]
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", "8"))

# Scalar kernel only supports M<=4
if KERNEL == "scalar":
    m_vals = [m for m in m_vals if m <= 4]

# Print actual M values to stderr so shell script knows what to parse
print(f"ACTUAL_M_VALS={','.join(str(m) for m in m_vals)}", file=sys.stderr)

# Dense/attention shapes
dense_shapes = [
    ("gateup", 2048, 5120),
    ("down", 5120, 2048),
    ("Q", 2048, 4096),
    ("O", 4096, 2048),
    ("KV", 2048, 512),
]

# MoE expert shapes (Qwen3-Coder-Next 70B)
moe_shapes = [
    ("moe_gu", 2048, 512),
    ("moe_dn", 512, 2048),
]

k_bits_list = [2, 3, 4, 5]
WARMUP = 5
PROFILED = 5

dev = torch.device("cuda")

if KERNEL in ("mma", "scalar"):
    # Pre-quantize dense shapes
    data = {}
    for name, K_dim, N in dense_shapes:
        for k in k_bits_list:
            codebook = create_normal_float_codebook(k, device=dev)
            W = torch.randn(K_dim * N, device=dev, dtype=torch.float32)
            packed_flat, absmax_flat = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
            packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(packed_flat, absmax_flat, K_dim, N, k)
            data[(name, k)] = (K_dim, N, packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook)

    configs = []
    for name, K_dim, N in dense_shapes:
        for k in k_bits_list:
            for M in m_vals:
                configs.append((name, k, M))

    for name, k, M in configs:
        K_dim, N, packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook = data[(name, k)]
        A = torch.randn(M, K_dim, dtype=torch.float16, device=dev)

        if KERNEL == "mma":
            fn = lambda: torch.ops.bitsandbytes.kbit_gemm_prod(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, 1)
        else:
            # Scalar GEMV uses flat layout with uint8 E4M4 absmax
            fn = lambda: torch.ops.bitsandbytes.kbit_scalar_gemv(A, packed_flat, absmax_flat, codebook, K_dim, N, k)

        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        for _ in range(PROFILED):
            fn()
        torch.cuda.synchronize()

elif KERNEL == "grouped_mma":
    # Pre-quantize MoE expert weights (NUM_EXPERTS copies, tiled layout for MMA)
    moe_data = {}
    for name, K_dim, N in moe_shapes:
        for k in k_bits_list:
            codebook = create_normal_float_codebook(k, device=dev)
            packed_list = []
            absmax_list = []
            for _ in range(NUM_EXPERTS):
                W = torch.randn(K_dim * N, device=dev, dtype=torch.float32)
                pf, af = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
                pt, at = torch.ops.bitsandbytes.repack_kbit(pf, af, K_dim, N, k)
                packed_list.append(pt)
                absmax_list.append(at)
            B_packed_all = torch.cat(packed_list, dim=0)
            B_absmax_all = torch.cat(absmax_list, dim=0)
            moe_data[(name, k)] = (K_dim, N, B_packed_all, B_absmax_all, codebook)

    configs = []
    for name, K_dim, N in moe_shapes:
        for k in k_bits_list:
            for M in m_vals:
                configs.append((name, k, M))

    for name, k, M in configs:
        K_dim, N, B_packed_all, B_absmax_all, codebook = moe_data[(name, k)]
        total_tokens = M * NUM_EXPERTS
        A_concat = torch.randn(total_tokens, K_dim, dtype=torch.float16, device=dev)
        offsets = list(range(0, total_tokens + 1, M))
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device=dev)

        fn = lambda: torch.ops.bitsandbytes.kbit_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets, K_dim, N, k, NUM_EXPERTS, M
        )

        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        for _ in range(PROFILED):
            fn()
        torch.cuda.synchronize()
