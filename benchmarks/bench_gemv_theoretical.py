"""Theoretical analysis: scalar kbit GEMV/small-M kernel vs cuBLAS bmm.

Computes expected performance for a specialized scalar kernel that avoids
tensor cores entirely. For M=1-4, the MMA overhead in the current GEMM
kernel wastes 75-93% of tensor core work. A scalar approach amortizes the
dequant cost across M rows, with only 1 extra FMA per row.

Also benchmarks cuBLAS bmm at each config for ground-truth comparison.
"""

import sys
import time

import torch

sys.path.insert(0, ".")
from scipy.stats import norm

from bitsandbytes import _ops  # noqa: F401


def create_normal_float_codebook(k: int) -> torch.Tensor:
    n_levels = 1 << k
    quantiles = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    values = torch.tensor(norm.ppf(quantiles.numpy()), dtype=torch.float32)
    values = values / values.abs().max()
    return values


def bench(fn, warmup=30, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def prepare_and_bench_bmm(K_dim, N, num_experts, M_per_expert):
    """Benchmark cuBLAS bmm for given config."""
    A = torch.randn(num_experts, M_per_expert, K_dim, dtype=torch.float16, device="cuda")
    W_T = torch.randn(num_experts, K_dim, N, dtype=torch.float16, device="cuda")
    return bench(lambda: torch.bmm(A, W_T))


def prepare_and_bench_grouped(K_dim, N, num_experts, M_per_expert, k):
    """Benchmark kbit grouped GEMM for given config."""
    codebook = create_normal_float_codebook(k).cuda()
    packed_list = []
    absmax_list = []
    for _ in range(num_experts):
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        pf, af = torch.ops.bitsandbytes.quantize_kbit(W.reshape(-1), codebook, k)
        pt, at = torch.ops.bitsandbytes.repack_kbit(pf, af.cuda(), K_dim, N, k)
        packed_list.append(pt)
        absmax_list.append(at)

    B_packed_all = torch.cat(packed_list)
    B_absmax_all = torch.cat(absmax_list)

    A_list = [torch.randn(M_per_expert, K_dim, dtype=torch.float16, device="cuda") for _ in range(num_experts)]
    offsets = [0]
    for i in range(num_experts):
        offsets.append(offsets[-1] + M_per_expert)
    A_concat = torch.cat(A_list)
    expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    return bench(
        lambda: torch.ops.bitsandbytes.kbit_grouped_gemm(
            A_concat,
            B_packed_all,
            B_absmax_all,
            codebook,
            expert_offsets,
            K_dim,
            N,
            k,
            num_experts,
        )
    )


def expected_unique_experts(batch_size, total_experts, top_k):
    p_miss = (1 - top_k / total_experts) ** batch_size
    return total_experts * (1 - p_miss)


def main():
    k = 4

    # RTX 4090 specs
    L2_BW_GBs = 2000
    DRAM_BW_GBs = 900
    L2_SIZE_MB = 72
    NUM_SMS = 128
    CORES_PER_SM = 128
    CLOCK_GHZ = 2.52

    # INT32 throughput (for dequant ops)
    INT_TOPS = NUM_SMS * CORES_PER_SM * CLOCK_GHZ  # ~41.2 TOPS

    # Qwen3 shapes
    shapes = [
        (2048, 512, "gate/up"),
        (512, 2048, "down"),
    ]

    total_experts_qwen = 512
    top_k_qwen = 8
    total_experts_glm = 64
    top_k_glm = 4

    print(f"Scalar kbit GEMV Analysis: K={k}, RTX 4090")
    print(f"INT32 throughput: {INT_TOPS:.1f} TOPS, L2 BW: {L2_BW_GBs} GB/s")
    print()

    for model_name, total_exp, top_k, shapes_list in [
        ("Qwen3-Coder-Next (512 experts, top-8)", total_experts_qwen, top_k_qwen, shapes),
        (
            "GLM-4.7-Flash (64 experts, top-4)",
            total_experts_glm,
            top_k_glm,
            [(2048, 1536, "gate/up"), (1536, 2048, "down")],
        ),
    ]:
        print(f"{'=' * 100}")
        print(f"  {model_name}")
        print(f"{'=' * 100}")
        print()

        hdr = (
            f"{'Batch':>5} | {'#exp':>4} {'M/e':>4} | "
            f"{'Scalar est':>10} {'bmm meas':>10} {'grp meas':>10} | "
            f"{'Scalar/bmm':>10} {'Scalar/grp':>10}"
        )
        print(hdr)
        print("-" * len(hdr))

        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            # Compute expected routing
            num_active = expected_unique_experts(batch_size, total_exp, top_k)
            num_active_int = max(1, round(num_active))
            total_invocations = batch_size * top_k
            avg_M = total_invocations / num_active
            # For bmm, we use M = round(avg_M) (uniform distribution)
            M_per_expert = max(1, round(avg_M))

            # Cap at actual total experts
            num_active_int = min(num_active_int, total_exp)

            # --- Theoretical scalar kernel estimate ---
            total_scalar_us = 0.0
            for K_dim, N, _ in shapes_list:
                N_padded = ((N + 127) // 128) * 128

                # Data sizes
                kbit_per_expert = N_padded * K_dim * k / 8 + N_padded * (K_dim // 32)
                total_kbit = num_active_int * kbit_per_expert
                a_data = num_active_int * M_per_expert * K_dim * 2
                total_data = total_kbit + a_data

                # Bandwidth (L2 if fits, DRAM otherwise)
                if total_data < L2_SIZE_MB * 1e6:
                    bw = L2_BW_GBs
                else:
                    bw = DRAM_BW_GBs
                t_bw_us = total_data / (bw * 1e3)  # GB/s → MB/us

                # Compute: (13 + M) ops per B element
                total_elements = num_active_int * N_padded * K_dim
                ops_per_element = 13 + M_per_expert
                total_ops = total_elements * ops_per_element
                t_compute_us = total_ops / (INT_TOPS * 1e6)  # TOPS → Mops/us

                # Estimated: max(bw, compute) × 1.8 overhead
                t_est = max(t_bw_us, t_compute_us) * 1.8
                total_scalar_us += t_est

            # --- Measured bmm ---
            total_bmm_us = 0.0
            for K_dim, N, _ in shapes_list:
                N_padded = ((N + 127) // 128) * 128
                t = prepare_and_bench_bmm(K_dim, N_padded, num_active_int, M_per_expert)
                total_bmm_us += t * 1e6

            # --- Measured grouped GEMM ---
            total_grp_us = 0.0
            for K_dim, N, _ in shapes_list:
                N_padded = ((N + 127) // 128) * 128
                t = prepare_and_bench_grouped(K_dim, N_padded, num_active_int, M_per_expert, k)
                total_grp_us += t * 1e6

            scalar_vs_bmm = total_bmm_us / total_scalar_us
            scalar_vs_grp = total_grp_us / total_scalar_us

            print(
                f"{batch_size:5d} | {num_active_int:4d} {M_per_expert:4d} | "
                f"{total_scalar_us:9.0f}us {total_bmm_us:9.0f}us {total_grp_us:9.0f}us | "
                f"{scalar_vs_bmm:9.2f}x {scalar_vs_grp:9.2f}x"
            )

        print()

    # Detailed breakdown for batch=1
    print(f"\n{'=' * 100}")
    print("  Detailed breakdown: Qwen3 batch=1 (8 experts, M=1)")
    print(f"{'=' * 100}")
    print()
    for K_dim, N, name in shapes:
        N_padded = ((N + 127) // 128) * 128
        ne = 8
        for M in [1, 2, 4]:
            kbit_data = ne * (N_padded * K_dim * k / 8 + N_padded * (K_dim // 32))
            total_elements = ne * N_padded * K_dim
            ops = 13 + M
            total_ops = total_elements * ops

            t_bw = kbit_data / (L2_BW_GBs * 1e3)
            t_compute = total_ops / (INT_TOPS * 1e6)
            t_est = max(t_bw, t_compute) * 1.8

            print(f"  {name} ({K_dim}x{N_padded}), 8 experts, M={M}:")
            print(f"    kbit data: {kbit_data / 1e6:.2f} MB, L2 BW time: {t_bw:.1f} us")
            print(
                f"    {total_elements / 1e6:.1f}M elements × {ops} ops = "
                f"{total_ops / 1e6:.0f}M ops → compute: {t_compute:.1f} us"
            )
            print(f"    Estimated (×1.8): {t_est:.1f} us")
            print()


if __name__ == "__main__":
    main()
