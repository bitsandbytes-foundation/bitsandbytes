"""cuBLAS fp16 baseline — CUDA event timing, pre-allocated I/O.

Env: M_VALS (default "1,2,3,4,8")
"""
import os, torch

shapes = [
    ("gateup", 2048, 5120),
    ("down",   5120, 2048),
    ("Q",      2048, 4096),
    ("O",      4096, 2048),
    ("KV",     2048,  512),
]
m_vals = [int(x) for x in os.environ.get("M_VALS", "1,2,3,4,8").split(",")]
dev = torch.device("cuda")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print(f"{'shape':<8} {'M':>2} {'avg_us':>10}")
print("---")

for name, K, N in shapes:
    W = torch.randn(K, N, dtype=torch.float16, device=dev)
    for M in m_vals:
        A = torch.randn(M, K, dtype=torch.float16, device=dev)
        out = torch.empty(M, N, dtype=torch.float16, device=dev)
        for _ in range(50):
            torch.mm(A, W, out=out)
        torch.cuda.synchronize()
        start.record()
        for _ in range(200):
            torch.mm(A, W, out=out)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000 / 200
        print(f"{name:<8} {M:>2} {us:>10.2f}")
