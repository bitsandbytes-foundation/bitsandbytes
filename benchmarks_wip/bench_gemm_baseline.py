"""Phase M1 baseline + Phase M3 comparison for gemm_4bit.

M1 question: as M grows, does the GEMM cost overtake dequant? (Answer: crossover near
M~512; below it gemm is dequant-bound.)

M3 question: what does the native path (chunked dequant -> scratch -> MPSMatrixMultiplication
-> bias, ONE command buffer / ONE sync) buy over the dequant + F.linear fallback (native
dequant wait + torch GEMM + second sync)? `gemm_4bit` routes native automatically when built,
so `native` here is just the op; `fallback` reproduces the old tail verbatim.

bf16 question: bf16 used to have no native path at all (MPSMatrixMultiplication hard-asserts
on it), so its "native" column WAS the fallback and the ratio was 1.00x by construction. It
now runs the GEMM through MPSGraph, so the ratio is finally a real measurement. The shapes
under `--cogkit` are the ones CogView4-6B QLoRA actually hits at 512x512 batch 1: hidden 4096,
MLP 16384, M = 1024 image tokens (+ text).

A `clone()` control is printed first. GPU timings on this machine are worthless under
contention; if the control moves between runs, nothing else in the table is comparable.
"""

import time

import torch

import bitsandbytes.backends.mps.ops as mps_ops
import bitsandbytes.functional as F

DEV = "mps"
ITERS = 30
WARMUP = 8


def sync():
    torch.mps.synchronize()


def timed(fn):
    for _ in range(WARMUP):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    sync()
    return (time.perf_counter() - t0) / ITERS * 1e3


def bench(M, N, K, dtype, quant_type="nf4", blocksize=64):
    A = torch.randn(1, M, K, dtype=dtype, device=DEV)
    B = torch.randn(N, K, dtype=dtype, device=DEV)
    B_q, qs = F.quantize_4bit(B, blocksize=blocksize, quant_type=quant_type)

    def native():
        # Routes through bnb_mps_gemm_4bit when the native library is built (fp32/fp16).
        return torch.ops.bitsandbytes.gemm_4bit(A, B_q, list(B.shape), qs.absmax, blocksize, quant_type)

    def dequant_only():
        return torch.ops.bitsandbytes.dequantize_4bit(B_q, qs.absmax, blocksize, quant_type, list(B.shape), dtype)

    def fallback():
        # The pre-M3 tail: native dequant (its own sync) + torch F.linear (torch's queue).
        B_dq = mps_ops._dequantize_4bit_impl(B_q, qs.absmax, blocksize, quant_type, list(B.shape), dtype)
        return torch.nn.functional.linear(A, B_dq)

    B_dq = dequant_only()

    def linear_only():
        return torch.nn.functional.linear(A, B_dq)

    t_nat, t_fb, t_deq, t_lin = timed(native), timed(fallback), timed(dequant_only), timed(linear_only)
    print(
        f"  M={M:>4} N={N:>5} K={K:>5} {str(dtype).replace('torch.', ''):>8}  "
        f"native={t_nat:7.3f}ms  fallback={t_fb:7.3f}ms  ({t_fb / t_nat:4.2f}x)  "
        f"[fallback = dequant {t_deq:6.3f} + linear {t_lin:6.3f}]"
    )


def bench_bwd(M, N, K, dtype, quant_type="nf4", blocksize=64):
    """gemm_4bit_backward: grad_A[M,K] = grad_output[M,N] . B_dq[N,K].

    `fallback` is the composition MatMul4Bit.backward ran inline before this op existed --
    native dequant (its own sync) then a torch matmul (torch's queue).
    """
    G = torch.randn(1, M, N, dtype=dtype, device=DEV)
    B = torch.randn(N, K, dtype=dtype, device=DEV)
    B_q, qs = F.quantize_4bit(B, blocksize=blocksize, quant_type=quant_type)

    def native():
        return torch.ops.bitsandbytes.gemm_4bit_backward(G, B_q, list(B.shape), qs.absmax, blocksize, quant_type)

    def fallback():
        B_dq = mps_ops._dequantize_4bit_impl(B_q, qs.absmax, blocksize, quant_type, list(B.shape), dtype)
        return torch.matmul(G, B_dq)

    t_nat, t_fb = timed(native), timed(fallback)
    print(
        f"  M={M:>4} N={N:>5} K={K:>5} {str(dtype).replace('torch.', ''):>8}  "
        f"native={t_nat:7.3f}ms  fallback={t_fb:7.3f}ms  ({t_fb / t_nat:4.2f}x)"
    )


def control():
    """Contention canary. A fixed 64 MB device clone, unrelated to any code under test: if
    this number differs across runs, the machine was busy and the table is not comparable."""
    x = torch.empty(16 * 1024 * 1024, dtype=torch.float32, device=DEV)
    return timed(lambda: x.clone())


# The shapes CogView4-6B QLoRA hits: (N, K) for qkv-proj, out-proj, mlp-in, mlp-out.
COGKIT_SHAPES = ((12288, 4096), (4096, 4096), (16384, 4096), (4096, 16384))


if __name__ == "__main__":
    import sys

    native = "native" if mps_ops._native_available() else "FALLBACK-ONLY (no native build)"
    print(f"iters={ITERS} warmup={WARMUP} device={DEV} lib={native}")
    print(f"control (64MB clone): {control():.3f}ms\n")

    if "--cogkit" in sys.argv:
        for dtype in (torch.bfloat16, torch.float16):
            print(f"=== gemm_4bit, CogView4-6B shapes, {str(dtype).replace('torch.', '')} ===")
            for M in (1024, 1280):
                for N, K in COGKIT_SHAPES:
                    bench(M, N, K, dtype)
            print()
        for dtype in (torch.bfloat16, torch.float16):
            print(f"=== gemm_4bit_BACKWARD, CogView4-6B shapes, {str(dtype).replace('torch.', '')} ===")
            for M in (1024, 1280):
                for N, K in COGKIT_SHAPES:
                    bench_bwd(M, N, K, dtype)
            print()
        # Second control read. If it has drifted from the first, the machine changed
        # underneath the table and the table is not internally comparable.
        print(f"control (64MB clone), after: {control():.3f}ms")
    else:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            print(f"=== gemm_4bit, N=K=4096, {str(dtype).replace('torch.', '')} ===")
            for M in (8, 64, 512, 2048):
                bench(M, 4096, 4096, dtype)
            print()
