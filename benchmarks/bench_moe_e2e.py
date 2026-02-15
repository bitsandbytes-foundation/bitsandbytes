"""End-to-end MoE layer benchmark: kbit grouped GEMM vs cuBLAS bmm.

Simulates realistic token-by-token generation for Qwen3-Coder-Next and
GLM-4.7-Flash. Computes total time for gate/up + down projections per
MoE layer at various batch sizes.

Expert routing: uniform random (worst case for expert reuse).
"""

import argparse
import math
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
    """Prepare kbit-quantized weights for num_experts experts."""
    codebook = create_normal_float_codebook(k).cuda()
    packed_list = []
    absmax_list = []
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

    B_packed_all = torch.cat(packed_list, dim=0)
    B_absmax_all = torch.cat(absmax_list, dim=0)
    return B_packed_all, B_absmax_all, codebook


def simulate_routing(batch_size, total_experts, top_k):
    """Simulate MoE routing: each token picks top_k experts uniformly.

    Returns:
        expert_ids: list of active expert IDs (sorted)
        M_per_expert: dict {expert_id: num_tokens}
        total_tokens: batch_size * top_k
    """
    # Each token independently picks top_k experts
    counts = {}
    for _ in range(batch_size):
        chosen = torch.randperm(total_experts)[:top_k].tolist()
        for e in chosen:
            counts[e] = counts.get(e, 0) + 1

    expert_ids = sorted(counts.keys())
    M_per_expert = {e: counts[e] for e in expert_ids}
    return expert_ids, M_per_expert


def expected_unique_experts(batch_size, total_experts, top_k):
    """Expected number of unique active experts under uniform routing."""
    p_miss = (1 - top_k / total_experts) ** batch_size
    return total_experts * (1 - p_miss)


def bench_one(fn, warmup=20, iters=200):
    """Time a callable (already capturing all args)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def run_model_benchmark(model_name, shapes, total_experts, top_k,
                        batch_sizes, k, warmup, iters):
    """Benchmark one model's MoE layer across batch sizes.

    shapes: list of (K_dim, N, layer_name) for the MoE projections.
    """
    codebook = create_normal_float_codebook(k).cuda()

    print(f"\n{'='*80}")
    print(f"  {model_name}: {total_experts} experts, top-{top_k}, K={k}")
    print(f"  MoE projections: {', '.join(f'{name} ({K}x{N})' for K, N, name in shapes)}")
    print(f"{'='*80}")
    print()

    hdr = (f"{'Batch':>5} | {'#active':>7} {'avg M':>5} {'max M':>5} | "
           + "  ".join(f"{'kbit(us)':>8} {'bmm(us)':>8}" for _ in shapes)
           + f" | {'Total kbit':>10} {'Total bmm':>10} {'Speedup':>8}")
    print(hdr)
    print("-" * len(hdr))

    for batch_size in batch_sizes:
        # Simulate routing
        expert_ids, M_per_expert = simulate_routing(batch_size, total_experts, top_k)
        num_active = len(expert_ids)
        M_values = list(M_per_expert.values())
        avg_M = sum(M_values) / len(M_values)
        max_M = max(M_values)

        total_kbit_us = 0.0
        total_bmm_us = 0.0
        per_shape_results = []

        for K_dim, N, layer_name in shapes:
            N_padded = ((N + 127) // 128) * 128

            # Prepare kbit weights for active experts
            B_packed_all, B_absmax_all, cb = prepare_expert_weights(
                K_dim, N_padded, k, num_active
            )

            # Build A_concat and expert_offsets from routing
            A_list = []
            offsets = [0]
            for eid in expert_ids:
                M_i = M_per_expert[eid]
                A_i = torch.randn(M_i, K_dim, dtype=torch.float16, device="cuda")
                A_list.append(A_i)
                offsets.append(offsets[-1] + M_i)

            A_concat = torch.cat(A_list, dim=0)
            expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

            # Benchmark kbit grouped GEMM
            t_kbit = bench_one(
                lambda: torch.ops.bitsandbytes.kbit_grouped_gemm(
                    A_concat, B_packed_all, B_absmax_all, cb,
                    expert_offsets, K_dim, N_padded, k, num_active,
                ),
                warmup=warmup, iters=iters,
            )

            # Benchmark cuBLAS bmm (pad all experts to max_M)
            A_padded = torch.zeros(num_active, max_M, K_dim,
                                   dtype=torch.float16, device="cuda")
            for i, eid in enumerate(expert_ids):
                M_i = M_per_expert[eid]
                A_padded[i, :M_i, :] = A_list[i]

            W_batched_T = torch.randn(num_active, K_dim, N_padded,
                                       dtype=torch.float16, device="cuda")

            t_bmm = bench_one(
                lambda: torch.bmm(A_padded, W_batched_T),
                warmup=warmup, iters=iters,
            )

            per_shape_results.append((t_kbit, t_bmm))
            total_kbit_us += t_kbit * 1e6
            total_bmm_us += t_bmm * 1e6

        # Print row
        shape_cols = "  ".join(
            f"{t_k*1e6:7.0f}us {t_b*1e6:7.0f}us"
            for t_k, t_b in per_shape_results
        )
        speedup = total_bmm_us / total_kbit_us if total_kbit_us > 0 else 0
        print(f"{batch_size:5d} | {num_active:7d} {avg_M:5.2f} {max_M:5d} | "
              f"{shape_cols} | {total_kbit_us:9.0f}us {total_bmm_us:9.0f}us {speedup:7.2f}x")


def main():
    parser = argparse.ArgumentParser(description="End-to-end MoE layer benchmark")
    parser.add_argument("--k", type=int, default=4, help="Bit width")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    # Qwen3-Coder-Next: 512 experts, top-8
    run_model_benchmark(
        "Qwen3-Coder-Next",
        shapes=[
            (2048, 512, "gate/up"),
            (512, 2048, "down"),
        ],
        total_experts=512,
        top_k=8,
        batch_sizes=batch_sizes,
        k=args.k, warmup=args.warmup, iters=args.iters,
    )

    # GLM-4.7-Flash: 64 routed experts, top-4 (typical config)
    run_model_benchmark(
        "GLM-4.7-Flash (routed only)",
        shapes=[
            (2048, 1536, "gate/up"),
            (1536, 2048, "down"),
        ],
        total_experts=64,
        top_k=4,
        batch_sizes=batch_sizes,
        k=args.k, warmup=args.warmup, iters=args.iters,
    )

    # Print theoretical analysis
    print(f"\n{'='*80}")
    print("  Theoretical: expected unique experts under uniform routing")
    print(f"{'='*80}")
    print()
    for model, te, tk in [("Qwen3 (512e, top-8)", 512, 8),
                           ("GLM4.7 (64e, top-4)", 64, 4)]:
        print(f"  {model}:")
        for bs in batch_sizes:
            eu = expected_unique_experts(bs, te, tk)
            total_inv = bs * tk
            avg_m = total_inv / eu
            print(f"    batch={bs:3d}: {eu:6.1f} unique experts, "
                  f"avg M={avg_m:.2f}, total invocations={total_inv}")
        print()


if __name__ == "__main__":
    main()
