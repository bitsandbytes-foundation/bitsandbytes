"""Benchmark for VQ grouped expert GEMM kernel.

Compares:
1. VQ grouped GEMM (one kernel launch for all experts)
2. cuBLAS batched GEMM via torch.bmm (one launch, fp16 weights)
3. Individual vq_linear calls (per-expert, correct routing)
4. Individual cuBLAS calls via torch.mm (one per expert, sequential)
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

from bitsandbytes import _ops  # noqa: F401
from bitsandbytes.functional import create_vq_codebook, quantize_vq, repack_vq, dequantize_vq, vq_linear


def prepare_vq_expert_weights(K_dim, N, p, num_experts):
    codebook = create_vq_codebook(p, device="cuda")
    packed_list = []
    absmax_list = []
    W_deq_list = []

    for _ in range(num_experts):
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        packed_flat, absmax_flat, _ = quantize_vq(W, p=p, codebook=codebook)
        W_deq = dequantize_vq(packed_flat, absmax_flat, codebook, p=p, n=N * K_dim).view(N, K_dim)
        packed_tiled, absmax_tiled = repack_vq(packed_flat, absmax_flat, K_dim, N, p=p)
        packed_list.append(packed_tiled)
        absmax_list.append(absmax_tiled)
        W_deq_list.append(W_deq)

    B_packed_all = torch.cat(packed_list, dim=0)
    B_absmax_all = torch.cat(absmax_list, dim=0)
    return B_packed_all, B_absmax_all, codebook, W_deq_list, packed_list, absmax_list


def bench_vq_grouped(
    A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets, K_dim, N, p, num_experts, max_M,
    warmup=20, iters=200
):
    for _ in range(warmup):
        torch.ops.bitsandbytes.vq_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, p, num_experts, max_M,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.ops.bitsandbytes.vq_grouped_gemm(
            A_concat, B_packed_all, B_absmax_all, codebook,
            expert_offsets, K_dim, N, p, num_experts, max_M,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def bench_batched_cublas(A_batched, W_batched_T, warmup=20, iters=200):
    for _ in range(warmup):
        torch.bmm(A_batched, W_batched_T)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.bmm(A_batched, W_batched_T)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def bench_individual_vq(A_list, packed_list, absmax_list, codebook, p, K_dim, N, warmup=20, iters=200):
    for _ in range(warmup):
        for i in range(len(A_list)):
            vq_linear(A_list[i], packed_list[i], absmax_list[i], codebook, p, K_dim, N)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        for i in range(len(A_list)):
            vq_linear(A_list[i], packed_list[i], absmax_list[i], codebook, p, K_dim, N)
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
    parser = argparse.ArgumentParser(description="Benchmark VQ grouped expert GEMM")
    parser.add_argument("--p", type=int, default=2, help="VQ dimension (2 or 4)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    p = args.p

    # MoE scenarios: (K_dim, N, num_experts, M_per_expert, description)
    configs = [
        # Qwen3 gate/up expert
        (2048, 512, 8, 1, "Qwen3 gate/up 8e M=1"),
        (2048, 512, 8, 4, "Qwen3 gate/up 8e M=4"),
        (2048, 512, 8, 8, "Qwen3 gate/up 8e M=8"),
        (2048, 512, 32, 1, "Qwen3 gate/up 32e M=1"),
        (2048, 512, 64, 1, "Qwen3 gate/up 64e M=1"),
        (2048, 512, 128, 1, "Qwen3 gate/up 128e M=1"),
        # Qwen3 down expert
        (512, 2048, 8, 1, "Qwen3 down 8e M=1"),
        (512, 2048, 8, 4, "Qwen3 down 8e M=4"),
        (512, 2048, 64, 1, "Qwen3 down 64e M=1"),
        # GLM-4.7-Flash routed expert
        (2048, 1536, 8, 1, "GLM4.7 routed 8e M=1"),
        (2048, 1536, 8, 4, "GLM4.7 routed 8e M=4"),
        (2048, 1536, 64, 1, "GLM4.7 routed 64e M=1"),
    ]

    print(f"VQ Grouped Expert GEMM Benchmark: p={p}")
    print(f"Warmup={args.warmup}, Iters={args.iters}")
    print()
    hdr = (
        f"{'Description':<28} | {'K':>4} {'N':>5} {'#e':>3} {'M':>2} | "
        f"{'vq grp':>8} {'bmm fp16':>8} {'vq seq':>8} {'mm seq':>8} | "
        f"{'vs bmm':>7} {'vs vq seq':>9} {'vs mm seq':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for K_dim, N, num_experts, M_per_expert, desc in configs:
        N_padded = ((N + 127) // 128) * 128

        B_packed_all, B_absmax_all, codebook, W_deq_list, packed_list, absmax_list = prepare_vq_expert_weights(
            K_dim, N_padded, p, num_experts
        )

        A_list = []
        offsets = [0]
        for i in range(num_experts):
            A_i = torch.randn(M_per_expert, K_dim, dtype=torch.float16, device="cuda")
            A_list.append(A_i)
            offsets.append(offsets[-1] + M_per_expert)

        A_concat = torch.cat(A_list, dim=0)
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

        # Build batched tensors for torch.bmm
        A_batched = torch.stack(A_list, dim=0)
        W_batched_T = torch.stack([W.T for W in W_deq_list], dim=0)

        # 1. VQ grouped GEMM
        t_grouped = bench_vq_grouped(
            A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets,
            K_dim, N_padded, p, num_experts, M_per_expert,
            warmup=args.warmup, iters=args.iters,
        )

        # 2. Batched cuBLAS
        t_bmm = bench_batched_cublas(
            A_batched, W_batched_T,
            warmup=args.warmup, iters=args.iters,
        )

        # 3. Individual vq_linear calls
        t_indiv_vq = bench_individual_vq(
            A_list, packed_list, absmax_list, codebook, p, K_dim, N_padded,
            warmup=args.warmup, iters=args.iters,
        )

        # 4. Individual cuBLAS calls
        t_indiv_mm = bench_individual_cublas(
            A_list, W_deq_list,
            warmup=args.warmup, iters=args.iters,
        )

        speedup_vs_bmm = t_bmm / t_grouped
        speedup_vs_vq_seq = t_indiv_vq / t_grouped
        speedup_vs_mm_seq = t_indiv_mm / t_grouped

        print(
            f"{desc:<28} | {K_dim:4d} {N_padded:5d} {num_experts:3d} {M_per_expert:2d} | "
            f"{t_grouped * 1e6:7.0f}us {t_bmm * 1e6:7.0f}us {t_indiv_vq * 1e6:7.0f}us {t_indiv_mm * 1e6:7.0f}us | "
            f"{speedup_vs_bmm:6.2f}x {speedup_vs_vq_seq:8.2f}x {speedup_vs_mm_seq:8.2f}x"
        )

    print()


if __name__ == "__main__":
    main()
