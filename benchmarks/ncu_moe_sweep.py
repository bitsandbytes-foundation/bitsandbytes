"""NCU driver for MoE grouped MMA sweep across wide M range.

Only k=4, but all power-of-2 M values from 1 to 4096.
Usage: ncu --kernel-name "kbit_grouped_gemm_prod" --metrics gpu__time_duration.avg python benchmarks/ncu_moe_sweep.py
"""
import os, sys, torch

for p in [".", ".."]:
    if os.path.isdir(os.path.join(p, "bitsandbytes")):
        sys.path.insert(0, os.path.abspath(p))
        break

import bitsandbytes
from bitsandbytes.functional import create_normal_float_codebook

NUM_EXPERTS = 8
K_BITS = 4
WARMUP = 3
PROFILED = 5

dev = torch.device("cuda")
codebook = create_normal_float_codebook(K_BITS, device=dev)

shapes = [
    ("moe_gu", 2048, 512),
    ("moe_dn",  512, 2048),
]

m_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Pre-quantize
moe_data = {}
for name, K_dim, N in shapes:
    packed_list, absmax_list = [], []
    for _ in range(NUM_EXPERTS):
        W = torch.randn(K_dim * N, device=dev, dtype=torch.float32)
        pf, af = torch.ops.bitsandbytes.quantize_kbit(W, codebook, K_BITS)
        pt, at = torch.ops.bitsandbytes.repack_kbit(pf, af, K_dim, N, K_BITS)
        packed_list.append(pt)
        absmax_list.append(at)
    B_packed_all = torch.cat(packed_list, dim=0)
    B_absmax_all = torch.cat(absmax_list, dim=0)
    moe_data[name] = (K_dim, N, B_packed_all, B_absmax_all)

# Print config to stderr
print(f"shapes={[s[0] for s in shapes]} k={K_BITS} M={m_vals} W={WARMUP} P={PROFILED}", file=sys.stderr)

for name, K_dim, N in shapes:
    K_dim, N, B_packed_all, B_absmax_all = moe_data[name]
    for M in m_vals:
        total_tokens = M * NUM_EXPERTS
        A_concat = torch.randn(total_tokens, K_dim, dtype=torch.float16, device=dev)
        offsets = list(range(0, total_tokens + 1, M))
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device=dev)

        fn = lambda: torch.ops.bitsandbytes.kbit_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, K_BITS, NUM_EXPERTS, M)

        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        for _ in range(PROFILED):
            fn()
        torch.cuda.synchronize()
