"""Analysis: small-batch strategies for kbit MoE GEMM.

Benchmarks three approaches for the batch=1 to batch=8 regime:
1. kbit grouped GEMM (current kernel)
2. cuBLAS bmm (fp16 baseline)
3. Dequant-to-fp16 + cuBLAS bmm (hybrid approach)

Also estimates theoretical performance of a specialized kbit GEMV kernel.
"""

import sys
import time

import torch

sys.path.insert(0, ".")
import bitsandbytes  # noqa: E402
from bitsandbytes import _ops  # noqa: E402, F401
from scipy.stats import norm  # noqa: E402


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


def main():
    k = 4
    codebook = create_normal_float_codebook(k).cuda()

    # Qwen3-Coder-Next MoE shapes
    shapes = [
        (2048, 512, "gate/up"),
        (512, 2048, "down"),
    ]

    print(f"Small-Batch MoE Strategy Analysis (K={k}, RTX 4090)")
    print(f"Model: Qwen3-Coder-Next (512 experts, top-8)")
    print()

    for K_dim, N, layer_name in shapes:
        N_padded = ((N + 127) // 128) * 128
        print(f"{'='*90}")
        print(f"  Layer: {layer_name} ({K_dim} x {N_padded})")
        print(f"{'='*90}")
        print()

        hdr = (f"{'#exp':>4} {'M':>2} | {'kbit grp':>8} {'bmm fp16':>8} "
               f"{'dq+bmm':>8} | {'grp/bmm':>8} {'dq+bmm/bmm':>11}")
        print(hdr)
        print("-" * len(hdr))

        for num_experts in [1, 4, 8, 16, 32, 64]:
            M_per_expert = 1

            # --- Prepare kbit weights ---
            packed_list = []
            absmax_list = []
            # Keep flat packed + absmax for dequant path
            flat_packed_list = []
            flat_absmax_list = []
            W_list = []

            for _ in range(num_experts):
                W = torch.randn(N_padded, K_dim, dtype=torch.float16, device="cuda")
                packed_flat, absmax_flat = torch.ops.bitsandbytes.quantize_kbit(
                    W.reshape(-1), codebook, k
                )
                packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
                    packed_flat, absmax_flat.cuda(), K_dim, N_padded, k
                )
                packed_list.append(packed_tiled)
                absmax_list.append(absmax_tiled)
                flat_packed_list.append(packed_flat)
                flat_absmax_list.append(absmax_flat.cuda())
                W_list.append(W)

            B_packed_all = torch.cat(packed_list, dim=0)
            B_absmax_all = torch.cat(absmax_list, dim=0)

            # --- Build activations ---
            A_list = []
            offsets = [0]
            for i in range(num_experts):
                A_i = torch.randn(M_per_expert, K_dim, dtype=torch.float16, device="cuda")
                A_list.append(A_i)
                offsets.append(offsets[-1] + M_per_expert)

            A_concat = torch.cat(A_list, dim=0)
            expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

            # --- 1. kbit grouped GEMM ---
            t_grouped = bench(lambda: torch.ops.bitsandbytes.kbit_grouped_gemm(
                A_concat, B_packed_all, B_absmax_all, codebook,
                expert_offsets, K_dim, N_padded, k, num_experts,
            ))

            # --- 2. cuBLAS bmm (fp16 baseline) ---
            A_batched = torch.stack(A_list, dim=0)
            W_batched_T = torch.stack([W.T for W in W_list], dim=0)

            t_bmm = bench(lambda: torch.bmm(A_batched, W_batched_T))

            # --- 3. Dequant + bmm ---
            # Pre-allocate output buffer for dequantized weights
            n_elements = N_padded * K_dim
            W_deq_flat = [torch.empty(n_elements, dtype=torch.float16, device="cuda")
                          for _ in range(num_experts)]

            n_elements = N_padded * K_dim

            def dequant_then_bmm():
                # Dequant each expert's weights to fp16
                deq_list = []
                for i in range(num_experts):
                    deq = torch.ops.bitsandbytes.dequantize_kbit(
                        flat_packed_list[i], codebook, flat_absmax_list[i],
                        k, n_elements, torch.float16,
                    )
                    deq_list.append(deq.view(N_padded, K_dim).T)
                # Stack into batched tensor and run bmm
                W_batch = torch.stack(deq_list, dim=0)
                return torch.bmm(A_batched, W_batch)

            t_dq_bmm = bench(dequant_then_bmm)

            # Also time just the dequant part
            def just_dequant():
                for i in range(num_experts):
                    torch.ops.bitsandbytes.dequantize_kbit(
                        flat_packed_list[i], codebook, flat_absmax_list[i],
                        k, n_elements, torch.float16,
                    )

            t_dq_only = bench(just_dequant)

            ratio_grp = t_grouped / t_bmm
            ratio_dq = t_dq_bmm / t_bmm

            print(f"{num_experts:4d} {M_per_expert:2d} | {t_grouped*1e6:7.0f}us "
                  f"{t_bmm*1e6:7.0f}us {t_dq_bmm*1e6:7.0f}us | "
                  f"{ratio_grp:7.2f}x {ratio_dq:10.2f}x"
                  f"   (dq alone: {t_dq_only*1e6:.0f}us)")

        print()

    # Theoretical GEMV analysis
    print(f"\n{'='*90}")
    print("  Theoretical: specialized kbit GEMV for batch=1")
    print(f"{'='*90}")
    print()
    print("  For M=1 (one token per expert), the GEMM kernel wastes 93.75% of tensor")
    print("  core work (TILE_M=16 but only 1 row has data). A scalar GEMV avoids this.")
    print()

    for K_dim, N, name in shapes:
        N_padded = ((N + 127) // 128) * 128
        kbit_bytes = num_experts * (N_padded * K_dim * k // 8 + N_padded * (K_dim // 32))
        fp16_bytes = num_experts * N_padded * K_dim * 2

        # RTX 4090 specs
        l2_bw = 2000  # GB/s effective L2 bandwidth
        dram_bw = 900  # GB/s
        sms = 128
        cores_per_sm = 128
        clock_ghz = 2.52

        # For 8 experts:
        ne = 8
        kbit_data = ne * (N_padded * K_dim * k / 8 + N_padded * (K_dim // 32))
        fp16_data = ne * N_padded * K_dim * 2

        # Bandwidth time (L2-resident for 8 experts)
        t_bw_kbit = kbit_data / (l2_bw * 1e9) * 1e6  # us
        t_bw_fp16 = fp16_data / (l2_bw * 1e9) * 1e6

        # Instruction time for kbit GEMV
        # Per element: ~14 integer/fp ops for dequant + FMA
        # Total elements: ne * N * K_dim
        total_elements = ne * N_padded * K_dim
        ops_per_element = 14
        total_ops = total_elements * ops_per_element
        # INT32 throughput: sms * cores * clock = ~41 TOPS
        int_throughput = sms * cores_per_sm * clock_ghz * 1e9
        t_compute = total_ops / int_throughput * 1e6  # us

        # Estimated total (max of bandwidth and compute, with some overhead)
        t_estimated = max(t_bw_kbit, t_compute) * 1.5  # 1.5x for overhead

        print(f"  {name} ({K_dim}x{N_padded}), 8 experts, M=1:")
        print(f"    kbit data:  {kbit_data/1e6:.2f} MB → L2 read: {t_bw_kbit:.1f} us")
        print(f"    fp16 data:  {fp16_data/1e6:.1f} MB → L2 read: {t_bw_fp16:.1f} us")
        print(f"    Compute (dequant+FMA): {total_elements/1e6:.1f}M elements × {ops_per_element} ops = {t_compute:.1f} us")
        print(f"    Estimated GEMV time: {t_estimated:.0f} us")
        print(f"    vs cuBLAS bmm ~17 us → {17/t_estimated:.1f}x")
        print()


if __name__ == "__main__":
    main()
