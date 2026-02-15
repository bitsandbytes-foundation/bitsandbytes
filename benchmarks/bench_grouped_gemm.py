"""Benchmark for kbit grouped expert GEMM kernel.

Compares:
1. Grouped GEMM (one kernel launch for all experts)
2. Individual kbit_gemm_prod calls (one per expert, sequential)
3. cuBLAS fp16 GEMM (one per expert, sequential)

Simulates MoE inference with varying batch sizes and expert counts.
"""

import argparse
import sys
import time

import torch

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


def prepare_expert_weights(K_dim, N, k, num_experts):
    codebook = create_normal_float_codebook(k).cuda()
    packed_list = []
    absmax_list = []
    W_list = []

    for _ in range(num_experts):
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        packed_flat, absmax = torch.ops.bitsandbytes.quantize_kbit(
            W.reshape(-1), codebook, k
        )
        packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
            packed_flat, absmax.cuda(), K_dim, N, k
        )
        packed_list.append(packed_tiled)
        absmax_list.append(absmax_tiled)
        W_list.append(W)

    B_packed_all = torch.cat(packed_list, dim=0)
    B_absmax_all = torch.cat(absmax_list, dim=0)
    return B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list


def bench_grouped_gemm(A_concat, B_packed_all, B_absmax_all, codebook,
                       expert_offsets, K_dim, N, k, num_experts,
                       warmup=20, iters=200):
    for _ in range(warmup):
        torch.ops.bitsandbytes.kbit_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, k, num_experts,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.ops.bitsandbytes.kbit_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, k, num_experts,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def bench_individual_kbit(A_list, packed_list, absmax_list, codebook,
                          K_dim, N, k, warmup=20, iters=200):
    for _ in range(warmup):
        for i in range(len(A_list)):
            torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i], packed_list[i], absmax_list[i], codebook,
                K_dim, N, k, 1,
            )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        for i in range(len(A_list)):
            torch.ops.bitsandbytes.kbit_gemm_prod(
                A_list[i], packed_list[i], absmax_list[i], codebook,
                K_dim, N, k, 1,
            )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def bench_individual_cublas(A_list, W_list, warmup=20, iters=200):
    for _ in range(warmup):
        for i in range(len(A_list)):
            torch.mm(A_list[i], W_list[i].T)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        for i in range(len(A_list)):
            torch.mm(A_list[i], W_list[i].T)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark grouped expert GEMM")
    parser.add_argument("--k", type=int, default=4, help="Bit width (2-5)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    k = args.k

    # MoE scenarios
    configs = [
        # (K_dim, N, num_experts, M_per_expert, description)
        # Qwen3-Coder-Next gate/up expert
        (2048, 512, 8, 1, "Qwen3 gate/up 8exp M=1"),
        (2048, 512, 8, 4, "Qwen3 gate/up 8exp M=4"),
        (2048, 512, 8, 8, "Qwen3 gate/up 8exp M=8"),
        (2048, 512, 32, 1, "Qwen3 gate/up 32exp M=1"),
        (2048, 512, 64, 1, "Qwen3 gate/up 64exp M=1"),
        (2048, 512, 128, 1, "Qwen3 gate/up 128exp M=1"),
        # Qwen3-Coder-Next down expert
        (512, 2048, 8, 1, "Qwen3 down 8exp M=1"),
        (512, 2048, 8, 4, "Qwen3 down 8exp M=4"),
        (512, 2048, 64, 1, "Qwen3 down 64exp M=1"),
        # GLM-4.7-Flash routed expert
        (2048, 1536, 8, 1, "GLM4.7 routed 8exp M=1"),
        (2048, 1536, 8, 4, "GLM4.7 routed 8exp M=4"),
        (2048, 1536, 64, 1, "GLM4.7 routed 64exp M=1"),
    ]

    print(f"Grouped Expert GEMM Benchmark: K={k}")
    print(f"Warmup={args.warmup}, Iters={args.iters}")
    print()
    print(f"{'Description':<30} | {'K_dim':>5} {'N':>5} {'#exp':>4} {'M/e':>3} | "
          f"{'Grouped(us)':>11} {'Indiv(us)':>10} {'cuBLAS(us)':>10} | "
          f"{'vs Indiv':>8} {'vs cuBLAS':>9}")
    print("-" * 120)

    for K_dim, N, num_experts, M_per_expert, desc in configs:
        N_padded = ((N + 127) // 128) * 128

        B_packed_all, B_absmax_all, codebook, W_list, packed_list, absmax_list = (
            prepare_expert_weights(K_dim, N_padded, k, num_experts)
        )

        # Build activations
        A_list = []
        offsets = [0]
        for i in range(num_experts):
            A_i = torch.randn(M_per_expert, K_dim, dtype=torch.float16, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + M_per_expert)

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        # Benchmark grouped
        t_grouped = bench_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N_padded, k, num_experts,
            warmup=args.warmup, iters=args.iters,
        )

        # Benchmark individual kbit
        t_individual = bench_individual_kbit(
            A_list, packed_list, absmax_list, codebook,
            K_dim, N_padded, k,
            warmup=args.warmup, iters=args.iters,
        )

        # Benchmark individual cuBLAS
        W_fp16_list = [W.half().cuda() for W in W_list]
        t_cublas = bench_individual_cublas(
            A_list, W_fp16_list,
            warmup=args.warmup, iters=args.iters,
        )

        speedup_vs_indiv = t_individual / t_grouped
        speedup_vs_cublas = t_cublas / t_grouped

        print(f"{desc:<30} | {K_dim:5d} {N_padded:5d} {num_experts:4d} {M_per_expert:3d} | "
              f"{t_grouped*1e6:11.1f} {t_individual*1e6:10.1f} {t_cublas*1e6:10.1f} | "
              f"{speedup_vs_indiv:7.2f}x {speedup_vs_cublas:8.2f}x")

    print()


if __name__ == "__main__":
    main()
