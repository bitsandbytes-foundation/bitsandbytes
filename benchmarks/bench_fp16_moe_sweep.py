"""fp16 BMM baseline for MoE shapes across wide M range.

Uses CUDA events (accurate for fp16 bmm which has no Python overhead).
"""

import torch

NUM_EXPERTS = 8
WARMUP = 50
ITERS = 200

dev = torch.device("cuda")

shapes = [
    ("moe_gu", 2048, 512),
    ("moe_dn", 512, 2048),
]

m_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

print(f"{'shape':<8} {'M':>5} {'fp16_us':>8}")
print("-" * 24)

for name, K_dim, N in shapes:
    for M in m_vals:
        A = torch.randn(NUM_EXPERTS, M, K_dim, dtype=torch.float16, device=dev)
        B = torch.randn(NUM_EXPERTS, K_dim, N, dtype=torch.float16, device=dev)

        fn = lambda: torch.bmm(A, B)
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            fn()
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end) / ITERS * 1000  # us
        print(f"{name:<8} {M:>5} {t:>8.1f}")
    print()
