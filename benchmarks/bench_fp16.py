"""cuBLAS fp16 baseline — CUDA event timing, pre-allocated I/O.

Benchmarks dense matmul (torch.mm) and batched MoE matmul (torch.bmm).

Env: M_VALS (default "1,2,3,4,8"), NUM_EXPERTS (default "8")
"""

import os

import torch

dense_shapes = [
    ("gateup", 2048, 5120),
    ("down", 5120, 2048),
    ("Q", 2048, 4096),
    ("O", 4096, 2048),
    ("KV", 2048, 512),
]
moe_shapes = [
    ("moe_gu", 2048, 512),
    ("moe_dn", 512, 2048),
]

m_vals = [int(x) for x in os.environ.get("M_VALS", "1,2,3,4,8").split(",")]
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", "8"))
dev = torch.device("cuda")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

WARMUP = 50
ITERS = 200

# --- Dense layers (torch.mm) ---
print(f"{'shape':<8} {'M':>2} {'avg_us':>10}")
print("---")

for name, K, N in dense_shapes:
    W = torch.randn(K, N, dtype=torch.float16, device=dev)
    for M in m_vals:
        A = torch.randn(M, K, dtype=torch.float16, device=dev)
        out = torch.empty(M, N, dtype=torch.float16, device=dev)
        for _ in range(WARMUP):
            torch.mm(A, W, out=out)
        torch.cuda.synchronize()
        start.record()
        for _ in range(ITERS):
            torch.mm(A, W, out=out)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000 / ITERS
        print(f"{name:<8} {M:>2} {us:>10.2f}")

# --- MoE layers (torch.bmm) ---
print()
print(f"{'shape':<8} {'M':>2} {'nexp':>4} {'avg_us':>10}")
print("---")

for name, K, N in moe_shapes:
    # Weight: [num_experts, K, N] — each expert has its own weight matrix
    W_batch = torch.randn(NUM_EXPERTS, K, N, dtype=torch.float16, device=dev)
    for M in m_vals:
        # A: [num_experts, M, K] — M tokens per expert
        A_batch = torch.randn(NUM_EXPERTS, M, K, dtype=torch.float16, device=dev)
        out = torch.empty(NUM_EXPERTS, M, N, dtype=torch.float16, device=dev)
        for _ in range(WARMUP):
            torch.bmm(A_batch, W_batch, out=out)
        torch.cuda.synchronize()
        start.record()
        for _ in range(ITERS):
            torch.bmm(A_batch, W_batch, out=out)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000 / ITERS
        print(f"{name:<8} {M:>2} {NUM_EXPERTS:>4} {us:>10.2f}")
