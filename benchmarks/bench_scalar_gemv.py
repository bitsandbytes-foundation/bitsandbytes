"""Benchmark scalar GEMV kernel vs MMA kernel vs cuBLAS vs dequant+cuBLAS.

Measures latency (us) and effective bandwidth (GB/s) for M=1,2,3,4
across shapes matching real model projections.
"""

import sys
import torch

sys.path.insert(0, ".")
import bitsandbytes  # noqa: E402
from bitsandbytes import _ops  # noqa: E402, F401
from bitsandbytes.functional import dequantize_kbit, quantize_kbit  # noqa: E402
from scipy.stats import norm  # noqa: E402

BLOCKSIZE = 32
WARMUP = 200
ITERS = 1000


def create_normal_float_codebook(k: int) -> torch.Tensor:
    n_levels = 1 << k
    quantiles = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    values = torch.tensor(norm.ppf(quantiles.numpy()), dtype=torch.float32)
    values = values / values.abs().max()
    return values


def prepare_weights(K_dim, N, k):
    codebook = create_normal_float_codebook(k).cuda()
    W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
    packed_flat, absmax_flat = torch.ops.bitsandbytes.quantize_kbit(
        W.reshape(-1), codebook, k
    )
    # Repacked data for MMA reference
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
        packed_flat, absmax_flat.cuda(), K_dim, N, k
    )
    # Also prepare for dequant kernel
    packed_flat2, absmax_flat2, cb_flat2 = quantize_kbit(
        W.reshape(-1).float().half(), k=k, absmax_format="e4m4"
    )
    return packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, W, packed_flat2, absmax_flat2, cb_flat2


def bench_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def kbit_data_bytes(K_dim, N, k, M):
    n_blocks = (K_dim * N) // BLOCKSIZE
    b_packed_bytes = n_blocks * k * 4
    b_absmax_bytes = n_blocks * 4  # float32 absmax (no E4M4 encoding)
    a_bytes = M * K_dim * 2
    return a_bytes + b_packed_bytes + b_absmax_bytes


def main():
    k = 4
    # Qwen3-Coder-Next shapes (hidden=2048, intermediate=5120, head_dim=256,
    # 16 attn heads, 2 KV heads, 512 experts top-10, moe_intermediate=512)
    shapes = [
        ("dense gate/up 2048x5120", 2048, 5120),
        ("dense down   5120x2048", 5120, 2048),
        ("Q proj       2048x4096", 2048, 4096),
        ("O proj       4096x2048", 4096, 2048),
        ("KV proj      2048x512",  2048, 512),
        ("linear key   2048x2048", 2048, 2048),
        ("MoE gate/up  2048x512",  2048, 512),
        ("MoE down     512x2048",  512,  2048),
    ]

    M_values = [1, 2, 3, 4]

    print(f"{'Shape':<26} {'M':>2}  {'Scalar':>8} {'MMA':>8} {'cuBLAS':>8} {'Dq+cuB':>8}  "
          f"{'S BW':>6} {'vs MMA':>7} {'vs cuB':>7} {'vs Dq+C':>7}")
    print("-" * 115)

    for label, K_dim, N in shapes:
        packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, W, pf2, af2, cf2 = prepare_weights(K_dim, N, k)
        n = K_dim * N

        for M in M_values:
            A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
            W_fp16 = W.half()

            # Scalar GEMV (flat layout, float32 absmax)
            C_out = torch.empty(M, N, device="cuda", dtype=torch.float16)
            t_scalar = bench_fn(lambda: torch.ops.bitsandbytes.kbit_scalar_gemv(
                A, packed_flat, absmax_flat, codebook, K_dim, N, k, 0, out=C_out))

            # MMA kernel (uses repacked tiled data)
            t_mma = bench_fn(lambda: torch.ops.bitsandbytes.kbit_gemm_prod(
                A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, 1))

            # cuBLAS
            t_cublas = bench_fn(lambda: torch.mm(A, W_fp16.t()))

            # Dequant + cuBLAS
            def dequant_cublas():
                W_deq = dequantize_kbit(pf2, af2, cf2, k=k, n=n, dtype=torch.float16)
                W_deq = W_deq.reshape(N, K_dim)
                return torch.mm(A, W_deq.t())
            t_dq_cublas = bench_fn(dequant_cublas)

            # Bandwidth
            kbit_bytes = kbit_data_bytes(K_dim, N, k, M)
            bw_scalar = kbit_bytes / (t_scalar * 1e-6) / 1e9

            speedup_mma = t_mma / t_scalar
            speedup_cublas = t_cublas / t_scalar
            speedup_dq = t_dq_cublas / t_scalar

            print(f"{label:<26} {M:>2}  {t_scalar:>7.1f}u {t_mma:>7.1f}u {t_cublas:>7.1f}u {t_dq_cublas:>7.1f}u  "
                  f"{bw_scalar:>5.0f}G {speedup_mma:>6.2f}x {speedup_cublas:>6.2f}x {speedup_dq:>6.2f}x")

        print()


if __name__ == "__main__":
    main()
