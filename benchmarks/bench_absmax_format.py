#!/usr/bin/env python3
"""Benchmark: float32 absmax vs uint8 E4M4 absmax for scalar GEMV.

Compares the V8 scalar GEMV kernel using:
  - float32 absmax (current default via kbit_scalar_gemv)
  - uint8 E4M4 absmax (experiment via kbit_scalar_gemv_u8)

Uses representative shapes from Qwen3-Coder-Next 70B.
"""

import torch
import time
import math
import bitsandbytes  # noqa: F401 — registers torch ops

from bitsandbytes.functional import create_normal_float_codebook


# ---- E4M4 encode (Python, matching CUDA encode_e4m4_absmax) ----
E4M4_BIAS = 11

def encode_e4m4_absmax(vals: torch.Tensor) -> torch.Tensor:
    """Encode float32 absmax values to uint8 E4M4 format."""
    out = torch.zeros(vals.shape, dtype=torch.uint8, device=vals.device)
    mask = vals > 0
    v = vals[mask].float()

    e_unbiased = torch.floor(torch.log2(v)).int()
    e_biased = (e_unbiased + E4M4_BIAS).clamp(0, 15)

    # Normal path: m = round((v / 2^e_unbiased - 1) * 16)
    m = torch.round((v / torch.exp2(e_unbiased.float()) - 1.0) * 16.0).int().clamp(0, 15)

    # Subnormal path (e_biased == 0): m = round(v / 2^(1-BIAS) * 16)
    subnormal = e_biased == 0
    if subnormal.any():
        subnormal_scale = 2.0 ** (1 - E4M4_BIAS)
        m[subnormal] = torch.round(v[subnormal] / subnormal_scale * 16.0).int().clamp(0, 15)

    raw = (e_biased << 4 | m).to(torch.uint8)
    out[mask] = raw
    return out


# ---- Benchmark config ----
SHAPES = [
    ("gateup",  7168, 18944),
    ("down",   18944,  7168),
    ("Q",       7168,  7168),
    ("O",       7168,  7168),
    ("KV",      7168,  1024),
]
K_BITS_LIST = [2, 3, 4, 5]
M_VALS = [1, 2, 3, 4]
WARMUP = 200
ITERS = 1000

dev = "cuda"


def bench():
    print(f"{'shape':>8s}  {'k':>2s}  {'M':>2s}  {'fp32_abs(us)':>12s}  {'u8_abs(us)':>11s}  {'ratio':>6s}")
    print("-" * 58)

    for name, K_dim, N in SHAPES:
        for k in K_BITS_LIST:
            codebook = create_normal_float_codebook(k, device=dev)
            W = torch.randn(K_dim * N, device=dev, dtype=torch.float32)
            packed_flat, absmax_flat = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
            absmax_u8 = encode_e4m4_absmax(absmax_flat)

            for M in M_VALS:
                A = torch.randn(M, K_dim, dtype=torch.float16, device=dev)

                # float32 absmax
                fn_f32 = lambda: torch.ops.bitsandbytes.kbit_scalar_gemv(
                    A, packed_flat, absmax_flat, codebook, K_dim, N, k)
                # uint8 E4M4 absmax
                fn_u8 = lambda: torch.ops.bitsandbytes.kbit_scalar_gemv_u8(
                    A, packed_flat, absmax_u8, codebook, K_dim, N, k)

                # Warmup
                for _ in range(WARMUP):
                    fn_f32()
                    fn_u8()
                torch.cuda.synchronize()

                # Time float32
                start = time.perf_counter()
                for _ in range(ITERS):
                    fn_f32()
                torch.cuda.synchronize()
                t_f32 = (time.perf_counter() - start) / ITERS * 1e6

                # Time uint8
                start = time.perf_counter()
                for _ in range(ITERS):
                    fn_u8()
                torch.cuda.synchronize()
                t_u8 = (time.perf_counter() - start) / ITERS * 1e6

                ratio = t_f32 / t_u8 if t_u8 > 0 else float('inf')
                print(f"{name:>8s}  {k:>2d}  {M:>2d}  {t_f32:>12.1f}  {t_u8:>11.1f}  {ratio:>5.2f}x")


if __name__ == "__main__":
    # Verify correctness first
    print("=== Correctness check ===")
    k = 3
    K_dim, N = 7168, 7168
    codebook = create_normal_float_codebook(k, device=dev)
    W = torch.randn(K_dim * N, device=dev, dtype=torch.float32)
    packed, absmax_f32 = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
    absmax_u8 = encode_e4m4_absmax(absmax_f32)
    A = torch.randn(1, K_dim, dtype=torch.float16, device=dev)

    out_f32 = torch.ops.bitsandbytes.kbit_scalar_gemv(A, packed, absmax_f32, codebook, K_dim, N, k)
    out_u8 = torch.ops.bitsandbytes.kbit_scalar_gemv_u8(A, packed, absmax_u8, codebook, K_dim, N, k)

    # E4M4 is lossy, so outputs won't match exactly. Check relative error.
    rel_err = (out_f32 - out_u8).abs() / (out_f32.abs() + 1e-8)
    print(f"  Max relative error: {rel_err.max().item():.4f}")
    print(f"  Mean relative error: {rel_err.mean().item():.6f}")
    print()

    print("=== Performance comparison ===")
    print("ratio > 1.00 means uint8 is faster\n")
    bench()
