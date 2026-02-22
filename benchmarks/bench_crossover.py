"""Comprehensive crossover analysis and per-model speedup estimation.

Benchmarks:
1. Dequant + cuBLAS vs fused kbit GEMM at varying M (dense layers)
2. Grouped kbit GEMM vs cuBLAS bmm (MoE expert layers)
3. Full model speedup table per batch size (all layers combined)

Target models: Qwen3-Coder-Next, GLM-4.7-Flash
"""

import sys
import time

import torch

sys.path.insert(0, ".")
import bitsandbytes  # noqa: E402
from bitsandbytes import _ops  # noqa: E402, F401
from bitsandbytes.functional import encode_absmax_e4m4  # noqa: E402
from scipy.stats import norm  # noqa: E402


def create_normal_float_codebook(k: int) -> torch.Tensor:
    n_levels = 1 << k
    quantiles = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    values = torch.tensor(norm.ppf(quantiles.numpy()), dtype=torch.float32)
    values = values / values.abs().max()
    return values


def bench(fn, warmup=30, iters=300):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


# ─── Dense layer benchmarks (varying M) ────────────────────────────────────

def bench_dense_crossover(K_dim, N, k, codebook, M_values):
    """Benchmark fused kbit GEMM vs dequant+cuBLAS vs cuBLAS-only at varying M."""
    N_padded = ((N + 127) // 128) * 128

    # Quantize weight
    W = torch.randn(N_padded, K_dim, dtype=torch.float16, device="cuda")
    packed_flat, absmax_flat = torch.ops.bitsandbytes.quantize_kbit(
        W.reshape(-1), codebook, k
    )
    # repack_kbit expects fp32 absmax (does its own E4M4 encoding)
    packed_tiled, absmax_tiled = torch.ops.bitsandbytes.repack_kbit(
        packed_flat, absmax_flat.cuda(), K_dim, N_padded, k
    )
    # Pre-encode absmax to E4M4 for dequant path (avoid re-encoding per call)
    absmax_e4m4 = encode_absmax_e4m4(absmax_flat).cuda()
    n_elements = N_padded * K_dim

    # Pre-dequantize once for cuBLAS baseline
    W_fp16 = W.T.contiguous()  # (K_dim, N_padded) for matmul

    results = []
    for M in M_values:
        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        # 1. Fused kbit GEMM (production kernel)
        t_fused = bench(lambda: torch.ops.bitsandbytes.kbit_gemm_prod(
            A, packed_tiled, absmax_tiled, codebook, K_dim, N_padded, k, 1,
        ))

        # 2. cuBLAS fp16 (baseline — assumes weights already in fp16)
        t_cublas = bench(lambda: torch.mm(A, W_fp16))

        # 3. Dequant + cuBLAS (absmax already E4M4, no re-encoding)
        def dequant_then_mm():
            deq = torch.ops.bitsandbytes.dequantize_kbit(
                packed_flat, codebook, absmax_e4m4,
                k, n_elements, torch.float16,
            )
            return torch.mm(A, deq.view(N_padded, K_dim).T)
        t_dq_mm = bench(dequant_then_mm)

        # 4. Just the dequant (to see its cost)
        t_dq_only = bench(lambda: torch.ops.bitsandbytes.dequantize_kbit(
            packed_flat, codebook, absmax_e4m4,
            k, n_elements, torch.float16,
        ))

        results.append({
            "M": M,
            "fused_us": t_fused * 1e6,
            "cublas_us": t_cublas * 1e6,
            "dq_mm_us": t_dq_mm * 1e6,
            "dq_only_us": t_dq_only * 1e6,
        })

    return results


# ─── MoE layer benchmarks (varying batch → varying experts) ────────────────

def expected_unique_experts(batch_size, total_experts, top_k):
    p_miss = (1 - top_k / total_experts) ** batch_size
    return total_experts * (1 - p_miss)


def bench_moe_layer(K_dim, N, k, codebook, num_experts, M_per_expert):
    """Benchmark grouped kbit GEMM vs cuBLAS bmm for one MoE layer shape."""
    N_padded = ((N + 127) // 128) * 128

    # Quantize expert weights
    packed_list, absmax_list, W_list = [], [], []
    for _ in range(num_experts):
        W = torch.randn(N_padded, K_dim, dtype=torch.float16, device="cuda")
        pf, af = torch.ops.bitsandbytes.quantize_kbit(W.reshape(-1), codebook, k)
        pt, at = torch.ops.bitsandbytes.repack_kbit(pf, af.cuda(), K_dim, N_padded, k)
        packed_list.append(pt)
        absmax_list.append(at)
        W_list.append(W)

    B_packed_all = torch.cat(packed_list)
    B_absmax_all = torch.cat(absmax_list)

    # Build activations
    A_list = [torch.randn(M_per_expert, K_dim, dtype=torch.float16, device="cuda")
              for _ in range(num_experts)]
    offsets = [0]
    for i in range(num_experts):
        offsets.append(offsets[-1] + M_per_expert)
    A_concat = torch.cat(A_list)
    expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    # 1. Grouped kbit GEMM
    t_grouped = bench(lambda: torch.ops.bitsandbytes.kbit_grouped_gemm(
        A_concat, B_packed_all, B_absmax_all, codebook,
        expert_offsets, K_dim, N_padded, k, num_experts,
    ))

    # 2. cuBLAS bmm
    A_batched = torch.stack(A_list, dim=0)
    W_batched_T = torch.stack([W.T.contiguous() for W in W_list], dim=0)
    # Ensure shapes match: A_batched (ne, M, K), W_batched_T (ne, K, N)
    t_bmm = bench(lambda: torch.bmm(A_batched, W_batched_T))

    return t_grouped * 1e6, t_bmm * 1e6


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    k = 4
    codebook = create_normal_float_codebook(k).cuda()

    # ════════════════════════════════════════════════════════════════════════
    # Part 1: Dense layer crossover (dequant+cuBLAS vs fused kbit GEMM)
    # ════════════════════════════════════════════════════════════════════════

    dense_shapes = {
        "Qwen3": [
            (2048, 5120, "dense gate/up"),
            (5120, 2048, "dense down"),
            (2048, 4096, "Q proj"),
            (2048, 512,  "KV proj"),
            (4096, 2048, "O proj"),
        ],
        "GLM4.7": [
            (2048, 10240, "shared gate/up"),
            (10240, 2048, "shared down"),
        ],
    }

    M_values = [1, 2, 4, 8, 16, 32, 64, 128]

    print(f"{'='*100}")
    print(f"  Part 1: Dense Layer Crossover (K={k}, fused kbit vs dequant+cuBLAS vs cuBLAS)")
    print(f"{'='*100}")
    print()

    # Store results for Part 3
    dense_crossover_data = {}

    for model_name, shapes in dense_shapes.items():
        print(f"--- {model_name} ---")
        print()
        for K_dim, N, layer_name in shapes:
            N_padded = ((N + 127) // 128) * 128
            print(f"  {layer_name} ({K_dim} x {N_padded}):")

            hdr = (f"    {'M':>4} | {'fused':>8} {'cuBLAS':>8} {'dq+mm':>8} "
                   f"{'dq only':>8} | {'fused/cub':>9} {'dq+mm/cub':>9} {'best':>12}")
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))

            results = bench_dense_crossover(K_dim, N, k, codebook, M_values)
            key = (model_name, layer_name)
            dense_crossover_data[key] = results

            for r in results:
                fused_ratio = r["cublas_us"] / r["fused_us"]
                dq_ratio = r["cublas_us"] / r["dq_mm_us"]
                best_kbit = min(r["fused_us"], r["dq_mm_us"])
                best_ratio = r["cublas_us"] / best_kbit
                best_label = "fused" if r["fused_us"] <= r["dq_mm_us"] else "dq+mm"
                print(f"    {r['M']:4d} | {r['fused_us']:7.0f}us {r['cublas_us']:7.0f}us "
                      f"{r['dq_mm_us']:7.0f}us {r['dq_only_us']:7.0f}us | "
                      f"{fused_ratio:8.2f}x {dq_ratio:8.2f}x "
                      f"{best_ratio:5.2f}x ({best_label})")
            print()
        print()

    # ════════════════════════════════════════════════════════════════════════
    # Part 2: MoE layer performance at realistic batch sizes
    # ════════════════════════════════════════════════════════════════════════

    print(f"{'='*100}")
    print(f"  Part 2: MoE Expert Layers (grouped kbit GEMM vs cuBLAS bmm)")
    print(f"{'='*100}")
    print()

    moe_configs = {
        "Qwen3": {
            "total_experts": 512,
            "top_k": 8,
            "shapes": [(2048, 512, "MoE gate/up"), (512, 2048, "MoE down")],
        },
        "GLM4.7": {
            "total_experts": 64,
            "top_k": 4,
            "shapes": [(2048, 1536, "routed gate/up"), (1536, 2048, "routed down")],
        },
    }

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    # Store results for Part 3
    moe_data = {}

    for model_name, cfg in moe_configs.items():
        total_exp = cfg["total_experts"]
        top_k = cfg["top_k"]
        shapes = cfg["shapes"]

        print(f"--- {model_name} ({total_exp} experts, top-{top_k}) ---")

        hdr = (f"  {'batch':>5} | {'#exp':>4} {'M/e':>4} | ",)
        parts = []
        for _, _, name in shapes:
            parts.append(f"{'grp':>7} {'bmm':>7} {'ratio':>6}")
        hdr = f"  {'batch':>5} | {'#exp':>4} {'M/e':>4} | " + " | ".join(parts) + " | total grp/bmm"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for bs in batch_sizes:
            num_active = expected_unique_experts(bs, total_exp, top_k)
            num_active_int = max(1, min(round(num_active), total_exp))
            total_invocations = bs * top_k
            M_per_expert = max(1, round(total_invocations / num_active))

            total_grp = 0
            total_bmm = 0
            parts_str = []

            for K_dim, N, name in shapes:
                t_grp, t_bmm = bench_moe_layer(
                    K_dim, N, k, codebook, num_active_int, M_per_expert
                )
                total_grp += t_grp
                total_bmm += t_bmm
                ratio = t_bmm / t_grp
                parts_str.append(f"{t_grp:6.0f}us {t_bmm:6.0f}us {ratio:5.2f}x")

                # Store for Part 3
                key = (model_name, name, bs)
                moe_data[key] = (t_grp, t_bmm)

            total_ratio = total_bmm / total_grp
            line = f"  {bs:5d} | {num_active_int:4d} {M_per_expert:4d} | " + " | ".join(parts_str)
            line += f" | {total_ratio:5.2f}x"
            print(line)

        print()

    # ════════════════════════════════════════════════════════════════════════
    # Part 3: Full model speedup per batch size
    # ════════════════════════════════════════════════════════════════════════

    print(f"{'='*100}")
    print(f"  Part 3: Full Model Speedup (all layers, per batch size)")
    print(f"{'='*100}")
    print()
    print("  Strategy: for each layer, pick the fastest kbit approach (fused or dq+cuBLAS)")
    print("  and compare total time against cuBLAS fp16 (no quantization).")
    print()

    # Model layer definitions (per transformer layer)
    # Each entry: (K_dim, N, layer_name, type, count_per_layer)
    # type: "dense" or "moe"
    # For MoE: the benchmark handles expert routing internally
    model_layers = {
        "Qwen3": {
            "dense": [
                (2048, 4096, "Q proj", 1),
                (2048, 512,  "KV proj", 1),
                (4096, 2048, "O proj", 1),
                (2048, 5120, "dense gate/up", 1),
                (5120, 2048, "dense down", 1),
            ],
            "moe_shapes": ["MoE gate/up", "MoE down"],
            "total_experts": 512,
            "top_k": 8,
        },
        "GLM4.7": {
            "dense": [
                (2048, 10240, "shared gate/up", 1),
                (10240, 2048, "shared down", 1),
                # Attention projections (estimated, hidden=2048)
                (2048, 2048, "Q proj", 1),
                (2048, 512,  "KV proj", 1),
                (2048, 2048, "O proj", 1),
            ],
            "moe_shapes": ["routed gate/up", "routed down"],
            "total_experts": 64,
            "top_k": 4,
        },
    }

    # For GLM attention projections, we need to benchmark those too
    # (they weren't in Part 1). Do it now.
    glm_attn_shapes = [
        (2048, 2048, "Q proj"),
        (2048, 512,  "KV proj"),
        (2048, 2048, "O proj"),
    ]
    for K_dim, N, layer_name in glm_attn_shapes:
        key = ("GLM4.7", layer_name)
        if key not in dense_crossover_data:
            results = bench_dense_crossover(K_dim, N, k, codebook, M_values)
            dense_crossover_data[key] = results

    for model_name, cfg in model_layers.items():
        print(f"{'─'*80}")
        print(f"  {model_name}")
        print(f"{'─'*80}")
        print()

        hdr = (f"  {'batch':>5} | {'dense kbit':>10} {'dense cub':>10} "
               f"{'MoE kbit':>10} {'MoE cub':>10} | "
               f"{'total kbit':>10} {'total cub':>10} {'speedup':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for bs in batch_sizes:
            # --- Dense layers ---
            total_dense_kbit_us = 0
            total_dense_cublas_us = 0

            for K_dim, N, layer_name, count in cfg["dense"]:
                key = (model_name, layer_name)
                if key not in dense_crossover_data:
                    # Benchmark missing shape
                    results = bench_dense_crossover(K_dim, N, k, codebook, M_values)
                    dense_crossover_data[key] = results

                # Find the result for M=bs (or closest)
                results = dense_crossover_data[key]
                # Find closest M
                best_r = min(results, key=lambda r: abs(r["M"] - bs))
                if best_r["M"] != bs:
                    # Need to benchmark this exact M
                    best_r = None
                    for r in results:
                        if r["M"] == bs:
                            best_r = r
                            break
                    if best_r is None:
                        # Use closest available
                        best_r = min(results, key=lambda r: abs(r["M"] - bs))

                best_kbit = min(best_r["fused_us"], best_r["dq_mm_us"])
                total_dense_kbit_us += best_kbit * count
                total_dense_cublas_us += best_r["cublas_us"] * count

            # --- MoE layers ---
            total_moe_kbit_us = 0
            total_moe_cublas_us = 0

            for moe_name in cfg["moe_shapes"]:
                key = (model_name, moe_name, bs)
                if key in moe_data:
                    t_grp, t_bmm = moe_data[key]
                    total_moe_kbit_us += t_grp
                    total_moe_cublas_us += t_bmm
                else:
                    # Fallback: wasn't benchmarked at this batch size
                    total_moe_kbit_us += 0
                    total_moe_cublas_us += 0

            total_kbit = total_dense_kbit_us + total_moe_kbit_us
            total_cublas = total_dense_cublas_us + total_moe_cublas_us
            speedup = total_cublas / total_kbit if total_kbit > 0 else 0

            print(f"  {bs:5d} | {total_dense_kbit_us:9.0f}us {total_dense_cublas_us:9.0f}us "
                  f"{total_moe_kbit_us:9.0f}us {total_moe_cublas_us:9.0f}us | "
                  f"{total_kbit:9.0f}us {total_cublas:9.0f}us {speedup:7.2f}x")

        print()

    # ════════════════════════════════════════════════════════════════════════
    # Part 4: Projected speedup with scalar kernel (theoretical)
    # ════════════════════════════════════════════════════════════════════════

    print(f"{'='*100}")
    print(f"  Part 4: Projected Model Speedup WITH Scalar Kernel (theoretical)")
    print(f"{'='*100}")
    print()
    print("  Uses 1.8x overhead factor for scalar kernel estimate at M<=4.")
    print("  Dense layers at M<=4: scalar estimate instead of fused GEMM.")
    print("  MoE layers at M<=4: scalar estimate instead of grouped GEMM.")
    print()

    L2_BW_GBs = 2000
    DRAM_BW_GBs = 900
    L2_SIZE_MB = 72

    def scalar_estimate_us(K_dim, N, k, num_experts, M_per_expert):
        """Estimate scalar kernel time for a batched shape."""
        N_padded = ((N + 127) // 128) * 128
        kbit_per_expert = N_padded * K_dim * k / 8 + N_padded * (K_dim // 32)
        total_kbit = num_experts * kbit_per_expert
        a_data = num_experts * M_per_expert * K_dim * 2
        total_data = total_kbit + a_data

        bw = L2_BW_GBs if total_data < L2_SIZE_MB * 1e6 else DRAM_BW_GBs
        t_bw_us = total_data / (bw * 1e3)

        total_elements = num_experts * N_padded * K_dim
        ops_per_element = 13 + M_per_expert
        INT_TOPS = 128 * 128 * 2.52  # ~41.3 TOPS
        t_compute_us = total_elements * ops_per_element / (INT_TOPS * 1e6)

        return max(t_bw_us, t_compute_us) * 1.8

    for model_name, cfg in model_layers.items():
        moe_cfg = moe_configs[model_name]
        total_exp = moe_cfg["total_experts"]
        top_k_val = moe_cfg["top_k"]

        print(f"{'─'*80}")
        print(f"  {model_name}")
        print(f"{'─'*80}")
        print()

        hdr = (f"  {'batch':>5} | {'dense kbit':>10} {'dense cub':>10} "
               f"{'MoE kbit':>10} {'MoE cub':>10} | "
               f"{'total kbit':>10} {'total cub':>10} {'speedup':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for bs in batch_sizes:
            # Expert routing
            num_active = expected_unique_experts(bs, total_exp, top_k_val)
            num_active_int = max(1, min(round(num_active), total_exp))
            total_invocations = bs * top_k_val
            M_per_expert = max(1, round(total_invocations / num_active))

            use_scalar = (bs <= 4)

            # --- Dense layers ---
            total_dense_kbit_us = 0
            total_dense_cublas_us = 0

            for K_dim, N, layer_name, count in cfg["dense"]:
                key = (model_name, layer_name)
                results = dense_crossover_data.get(key, [])
                best_r = min(results, key=lambda r: abs(r["M"] - bs)) if results else None

                if use_scalar and best_r:
                    # Use scalar estimate for M<=4
                    t_scalar = scalar_estimate_us(K_dim, N, k, 1, bs)
                    t_kbit = min(t_scalar, best_r["fused_us"], best_r["dq_mm_us"])
                elif best_r:
                    t_kbit = min(best_r["fused_us"], best_r["dq_mm_us"])
                else:
                    t_kbit = 0

                total_dense_kbit_us += t_kbit * count
                total_dense_cublas_us += (best_r["cublas_us"] if best_r else 0) * count

            # --- MoE layers ---
            total_moe_kbit_us = 0
            total_moe_cublas_us = 0

            for moe_name in cfg["moe_shapes"]:
                K_dim_moe = [s[0] for s in moe_cfg["shapes"] if s[2] == moe_name][0]
                N_moe = [s[1] for s in moe_cfg["shapes"] if s[2] == moe_name][0]

                if use_scalar:
                    t_scalar = scalar_estimate_us(
                        K_dim_moe, N_moe, k, num_active_int, M_per_expert
                    )
                    t_kbit = t_scalar
                else:
                    key = (model_name, moe_name, bs)
                    if key in moe_data:
                        t_grp, _ = moe_data[key]
                        t_kbit = t_grp
                    else:
                        t_kbit = 0

                total_moe_kbit_us += t_kbit

                # cuBLAS bmm baseline
                key = (model_name, moe_name, bs)
                if key in moe_data:
                    _, t_bmm = moe_data[key]
                    total_moe_cublas_us += t_bmm
                else:
                    total_moe_cublas_us += 0

            total_kbit = total_dense_kbit_us + total_moe_kbit_us
            total_cublas = total_dense_cublas_us + total_moe_cublas_us
            speedup = total_cublas / total_kbit if total_kbit > 0 else 0

            marker = " ← scalar" if use_scalar else ""
            print(f"  {bs:5d} | {total_dense_kbit_us:9.0f}us {total_dense_cublas_us:9.0f}us "
                  f"{total_moe_kbit_us:9.0f}us {total_moe_cublas_us:9.0f}us | "
                  f"{total_kbit:9.0f}us {total_cublas:9.0f}us {speedup:7.2f}x{marker}")

        print()


if __name__ == "__main__":
    main()
