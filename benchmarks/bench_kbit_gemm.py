"""Benchmark for kbit GEMM kernel.

Measures throughput (TFLOPS) and effective memory bandwidth (GB/s) for:
1. kbit_gemm_prod (production kernel, fp16 and bf16)
2. cuBLAS fp16 GEMM (baseline)
3. Standalone dequant + cuBLAS (simulated fused baseline)
"""

import argparse
import sys
import time

import torch

# Ensure bitsandbytes is importable from the worktree
sys.path.insert(0, ".")
import bitsandbytes  # noqa: E402
from bitsandbytes import _ops  # noqa: E402, F401
from scipy.stats import norm  # noqa: E402

BLOCKSIZE = 32


def create_normal_float_codebook(k: int) -> torch.Tensor:
    n_levels = 1 << k
    quantiles = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    values = torch.tensor(norm.ppf(quantiles.numpy()), dtype=torch.float32)
    values = values / values.abs().max()
    return values


def quantize_kbit_ref(A, codebook, blocksize=BLOCKSIZE):
    A_flat = A.float().reshape(-1)
    n = A_flat.numel()
    pad = (blocksize - n % blocksize) % blocksize
    if pad > 0:
        A_flat = torch.nn.functional.pad(A_flat, (0, pad))
    n_padded = A_flat.numel()
    num_blocks = n_padded // blocksize
    blocks = A_flat.reshape(num_blocks, blocksize)
    absmax = blocks.abs().max(dim=1).values
    absmax_safe = absmax.clamp(min=1e-8)
    normalized = blocks / absmax_safe.unsqueeze(1)
    cb = codebook.float().unsqueeze(0).unsqueeze(0)
    norm_exp = normalized.unsqueeze(2)
    distances = (norm_exp - cb).abs()
    indices = distances.argmin(dim=2).to(torch.uint8)
    indices = indices.reshape(-1)[:n]
    return indices, absmax


def pack_kbit_ref(indices, k, blocksize=BLOCKSIZE):
    n = indices.numel()
    pad = (blocksize - n % blocksize) % blocksize
    if pad > 0:
        indices = torch.nn.functional.pad(indices.int(), (0, pad))
    n_padded = indices.numel()
    num_blocks = n_padded // blocksize
    blocks = indices.int().reshape(num_blocks, blocksize)
    packed_words = []
    for b in range(num_blocks):
        for bit in range(k):
            word = 0
            for i in range(blocksize):
                word |= ((int(blocks[b, i]) >> bit) & 1) << i
            if word >= (1 << 31):
                word -= 1 << 32
            packed_words.append(word)
    return torch.tensor(packed_words, dtype=torch.int32)


def prepare_weights(K_dim, N, k):
    """Quantize and repack random weights using CUDA kernels. Returns (packed_tiled, absmax_tiled, codebook)."""
    codebook = create_normal_float_codebook(k)
    W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
    # Use CUDA quantize kernel (fast)
    packed_flat, absmax = torch.ops.bitsandbytes.quantize_kbit(W.reshape(-1), codebook.cuda(), k)
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
        packed_flat, absmax.cuda(), K_dim, N, k
    )
    return packed_tiled, absmax_tiled, codebook.cuda(), W


def bench_kbit_gemm(M, K_dim, N, k, k_chunks, dtype, packed_tiled, absmax_tiled, codebook,
                    warmup=10, iters=100):
    """Benchmark the production kbit GEMM kernel."""
    A = torch.randn(M, K_dim, dtype=dtype, device="cuda")

    # Warmup
    for _ in range(warmup):
        torch.ops.bitsandbytes.kbit_gemm_prod(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, k_chunks)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.ops.bitsandbytes.kbit_gemm_prod(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k, k_chunks)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters


def bench_cublas(M, K_dim, N, dtype, W_fp16, warmup=10, iters=100):
    """Benchmark cuBLAS fp16 GEMM as baseline."""
    A = torch.randn(M, K_dim, dtype=dtype, device="cuda")
    W = W_fp16.to(dtype).cuda()

    # Warmup
    for _ in range(warmup):
        torch.mm(A, W.T)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(A, W.T)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark kbit GEMM kernel")
    parser.add_argument("--k", type=int, default=4, help="Bit width (2-5)")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--k-chunks", type=int, default=1, help="Split-K chunks")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    k = args.k

    # LLM-typical shapes
    configs = [
        # (M, K_dim, N)
        (1, 4096, 4096),
        (1, 4096, 11008),
        (4, 4096, 4096),
        (4, 4096, 11008),
        (8, 4096, 4096),
        (16, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
    ]

    print(f"kbit GEMM Benchmark: K={k}, dtype={args.dtype}, k_chunks={args.k_chunks}")
    print(f"Warmup={args.warmup}, Iters={args.iters}")
    print()
    print(f"{'M':>5} {'K_dim':>6} {'N':>6} | {'kbit (us)':>10} {'kbit TFLOPS':>12} {'kbit GB/s':>10} | "
          f"{'cuBLAS (us)':>12} {'cuBLAS TFLOPS':>14} | {'Speedup':>8}")
    print("-" * 115)

    for M, K_dim, N in configs:
        # Pad N to multiple of 128 if needed
        N_padded = ((N + 127) // 128) * 128

        # Prepare weights
        packed_tiled, absmax_tiled, codebook, W = prepare_weights(K_dim, N_padded, k)

        # Benchmark kbit GEMM
        t_kbit = bench_kbit_gemm(M, K_dim, N_padded, k, args.k_chunks, dtype,
                                 packed_tiled, absmax_tiled, codebook,
                                 warmup=args.warmup, iters=args.iters)

        # Benchmark cuBLAS
        t_cublas = bench_cublas(M, K_dim, N_padded, dtype, W.half(),
                                warmup=args.warmup, iters=args.iters)

        # Compute metrics
        flops = 2 * M * K_dim * N_padded
        tflops_kbit = flops / t_kbit / 1e12
        tflops_cublas = flops / t_cublas / 1e12

        # Effective bandwidth for kbit: A (fp16) + B (compressed) + C (fp16)
        a_bytes = M * K_dim * 2
        b_bytes = N_padded * K_dim * k / 8 + N_padded * (K_dim // 32)  # packed + absmax
        c_bytes = M * N_padded * 2
        total_bytes = a_bytes + b_bytes + c_bytes
        gbps_kbit = total_bytes / t_kbit / 1e9

        speedup = t_cublas / t_kbit

        print(f"{M:5d} {K_dim:6d} {N_padded:6d} | {t_kbit*1e6:10.1f} {tflops_kbit:12.3f} {gbps_kbit:10.1f} | "
              f"{t_cublas*1e6:12.1f} {tflops_cublas:14.3f} | {speedup:8.2f}x")

    print()


if __name__ == "__main__":
    main()
