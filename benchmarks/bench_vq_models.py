"""Benchmark: VQ vs kbit vs FP16 across real model shapes.

Compares three methods:
1. kbit k=4 (bit-plane, old method) — uses scalar GEMV M<=4, MMA M=5-16
2. VQ p=2 (codebook, new method) — uses scalar GEMV M<=4, MMA M=5-16
3. cuBLAS FP16 (dense baseline)

Models:
- Qwen3-Coder-Next 70B (MoE): hidden=2048, dense MLP=5120, attn Q=4096
- GLM-4.7-Flash (MLA+MoE): hidden=2048, dense MLP=10240, MoE expert=1536

Usage:
  cd /path/to/bnb-kbit-gemm
  python benchmarks/bench_vq_models.py
"""

import json
import os
import sys

import torch

sys.path.insert(0, ".")
from bitsandbytes import _ops  # noqa: F401
from bitsandbytes.functional import (
    create_normal_float_codebook,
    create_vq_codebook,
    quantize_kbit,
    quantize_vq,
    repack_vq,
)


def bench(fn, inner: int, outer: int) -> float:
    """CUDA graph replay timing. Returns median us per iteration."""
    for _ in range(30):
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()

    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(outer):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / inner)
    times.sort()
    return times[len(times) // 2]


def prepare_vq(K_dim, N, p=2):
    dev = torch.device("cuda")
    cb = create_vq_codebook(p, device=dev)
    W = torch.randn(N, K_dim, dtype=torch.float16, device=dev)
    packed, absmax, cb = quantize_vq(W, p=p, codebook=cb)
    pt, at = repack_vq(packed, absmax, K_dim, N, p)
    return pt, at, cb


def prepare_kbit(K_dim, N, k=4):
    dev = torch.device("cuda")
    cb = create_normal_float_codebook(k, device=dev)
    W = torch.randn(N, K_dim, dtype=torch.float16, device=dev)
    packed, absmax, cb = quantize_kbit(W, k=k, codebook=cb)
    pt, at = torch.ops.bitsandbytes.repack_kbit(packed, absmax, K_dim, N, k)
    return pt, at, cb


# ---- Model shape definitions ----

MODELS = {
    "Qwen3 70B": [
        (2048, 5120, "dense gate/up"),
        (5120, 2048, "dense down"),
        (2048, 4096, "Q proj"),
        (4096, 2048, "O proj"),
        (2048, 512, "KV proj"),
    ],
    "GLM-4.7-Flash": [
        (2048, 10240, "shared gate/up"),
        (10240, 2048, "shared down"),
        (2048, 5120, "O proj"),       # 20 heads × 256 v_head_dim = 5120
        (2048, 768, "q_a_proj"),       # q_lora_rank compression
        (2048, 640, "kv_a_proj"),      # kv_lora_rank + rope_dim = 512+64=576 → pad to 640
        (768, 5120, "q_b_proj"),       # expand to 20*(192+64)=5120
        (512, 8960, "kv_b_proj"),      # expand to 20*(192+256)=8960
        (2048, 1536, "MoE gate/up"),   # per expert
        (1536, 2048, "MoE down"),      # per expert
    ],
}

M_VALUES = [1, 8, 16]
INNER = 300
OUTER = 10


def run_benchmarks():
    results = []
    dev = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Timing: {INNER} replays, median of {OUTER}")
    print()

    for model_name, shapes in MODELS.items():
        print(f"{'=' * 90}")
        print(f"  {model_name}")
        print(f"{'=' * 90}")
        print()

        for K_dim, N_raw, layer_name in shapes:
            # Pad N to multiple of 128 for MMA kernel compatibility
            N = ((N_raw + 127) // 128) * 128
            # Pad K to multiple of 64
            K_dim_pad = ((K_dim + 63) // 64) * 64

            print(f"  {layer_name} ({K_dim}x{N_raw}" +
                  (f" → padded {K_dim_pad}x{N})" if (K_dim_pad != K_dim or N != N_raw) else ")"))

            # Prepare quantized weights
            vq_pt, vq_at, vq_cb = prepare_vq(K_dim_pad, N, p=2)
            kb_pt, kb_at, kb_cb = prepare_kbit(K_dim_pad, N, k=4)
            W_fp16 = torch.randn(N, K_dim_pad, dtype=torch.float16, device=dev)

            print(f"  {'M':>4}  {'kbit k=4':>10}  {'VQ p=2':>10}  {'FP16':>10}  "
                  f"{'VQ/kbit':>8}  {'VQ/FP16':>8}  {'kbit/FP16':>9}")
            print(f"  {'':>4}  {'(us)':>10}  {'(us)':>10}  {'(us)':>10}  "
                  f"{'speedup':>8}  {'speedup':>8}  {'speedup':>9}")
            print(f"  {'-' * 76}")

            for M in M_VALUES:
                A = torch.randn(M, K_dim_pad, dtype=torch.float16, device=dev)

                # ---- kbit k=4 ----
                if M <= 4:
                    out_kb = torch.zeros(M, N, dtype=torch.float16, device=dev)
                    t_kbit = bench(
                        lambda: torch.ops.bitsandbytes.kbit_scalar_gemv_tiled_(
                            A, kb_pt, kb_at, kb_cb, K_dim_pad, N, 4, out_kb),
                        INNER, OUTER)
                else:
                    t_kbit = bench(
                        lambda: torch.ops.bitsandbytes.kbit_gemm_prod(
                            A, kb_pt, kb_at, kb_cb, K_dim_pad, N, 4, 1),
                        INNER, OUTER)

                # ---- VQ p=2 ----
                if M <= 4:
                    out_vq = torch.zeros(M, N, dtype=torch.float16, device=dev)
                    t_vq = bench(
                        lambda: torch.ops.bitsandbytes.vq_scalar_gemv_tiled_(
                            A, vq_pt, vq_at, vq_cb, K_dim_pad, N, 2, out_vq),
                        INNER, OUTER)
                else:
                    t_vq = bench(
                        lambda: torch.ops.bitsandbytes.vq_gemm_prod(
                            A, vq_pt, vq_at, vq_cb, K_dim_pad, N, 2, 1),
                        INNER, OUTER)

                # ---- cuBLAS FP16 ----
                out_fp = torch.empty(M, N, dtype=torch.float16, device=dev)
                t_fp16 = bench(
                    lambda: torch.mm(A, W_fp16.t(), out=out_fp),
                    INNER, OUTER)

                # Speedups
                vq_vs_kbit = t_kbit / t_vq
                vq_vs_fp16 = t_fp16 / t_vq
                kbit_vs_fp16 = t_fp16 / t_kbit

                print(f"  {M:>4}  {t_kbit:>10.2f}  {t_vq:>10.2f}  {t_fp16:>10.2f}  "
                      f"{vq_vs_kbit:>7.2f}x  {vq_vs_fp16:>7.2f}x  {kbit_vs_fp16:>8.2f}x")

                results.append({
                    "model": model_name, "layer": layer_name,
                    "K": K_dim_pad, "N": N, "M": M,
                    "kbit_us": round(t_kbit, 2),
                    "vq_us": round(t_vq, 2),
                    "fp16_us": round(t_fp16, 2),
                    "vq_vs_kbit": round(vq_vs_kbit, 3),
                    "vq_vs_fp16": round(vq_vs_fp16, 3),
                })

            print()

    return results


def print_summary(results):
    """Print compact summary grouped by M."""
    print(f"{'=' * 90}")
    print(f"  SUMMARY: Geometric mean speedups")
    print(f"{'=' * 90}")

    import math
    for M in M_VALUES:
        m_results = [r for r in results if r["M"] == M]
        if not m_results:
            continue

        for model_name in MODELS:
            model_results = [r for r in m_results if r["model"] == model_name]
            if not model_results:
                continue

            vq_kbit_ratios = [r["vq_vs_kbit"] for r in model_results]
            vq_fp16_ratios = [r["vq_vs_fp16"] for r in model_results]

            geo_vk = math.exp(sum(math.log(x) for x in vq_kbit_ratios) / len(vq_kbit_ratios))
            geo_vf = math.exp(sum(math.log(x) for x in vq_fp16_ratios) / len(vq_fp16_ratios))

            kernel = "scalar" if M <= 4 else "MMA"
            print(f"  M={M:>2} ({kernel:>6})  {model_name:<20}  "
                  f"VQ vs kbit: {geo_vk:.2f}x   VQ vs FP16: {geo_vf:.2f}x")

    print()


def main():
    results = run_benchmarks()
    print_summary(results)

    # Save JSON
    output = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "results": results,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/vq_model_bench.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to results/vq_model_bench.json")


if __name__ == "__main__":
    main()
