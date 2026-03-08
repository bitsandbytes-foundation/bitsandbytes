"""Benchmark hand-written NVFP4 GEMM kernel with GLM-4.7 shapes.

Measures kernel-only time (no Python overhead) using CUDA events.
Run on SM_120+ hardware (Blackwell).

Usage:
    python benchmarks/bench_hw_gemm.py
"""

import ctypes
import os
import time

import torch

# GLM-4.7 (352B MoE) layer shapes: (K, N)
GLM47_SHAPES = {
    "dense_qkv": (4096, 4096),
    "dense_o_proj": (4096, 4096),
    "moe_gate_up": (4096, 13696),
    "moe_down": (13696, 4096),
}

BATCH_SIZES = [1, 2, 4, 8, 16, 32]

WARMUP = 20
ITERS = 100


def get_lib():
    """Load the bitsandbytes CUDA library."""
    lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bitsandbytes")
    for suffix in ["cuda131", "cuda130", "cuda128"]:
        lib_path = os.path.join(lib_dir, f"libbitsandbytes_{suffix}.so")
        if os.path.exists(lib_path):
            return ctypes.cdll.LoadLibrary(lib_path)
    raise RuntimeError(f"Could not find bitsandbytes CUDA library in {lib_dir}")


def swizzled_scale_size(rows, scale_K):
    """Compute the size of the CUTLASS block-scaled (swizzled) scale buffer."""
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (scale_K + 3) // 4
    return n_row_blocks * n_col_blocks * 512


def bench_gemm_bf16(lib, M, N, K):
    """Benchmark cgemm_nvfp4_bf16 (hand-written kernel)."""
    scale_K = K // 16
    # Allocate packed FP4 data and swizzled-layout scales
    A_packed = torch.randint(0, 255, (M * K // 2,), dtype=torch.uint8, device="cuda")
    B_packed = torch.randint(0, 255, (N * K // 2,), dtype=torch.uint8, device="cuda")
    A_scales = torch.randint(0, 255, (swizzled_scale_size(M, scale_K),), dtype=torch.uint8, device="cuda")
    B_scales = torch.randint(0, 255, (swizzled_scale_size(N, scale_K),), dtype=torch.uint8, device="cuda")
    D_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    workspace = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    stream = torch.cuda.current_stream()

    def run():
        lib.cgemm_nvfp4_bf16(
            ctypes.c_void_p(A_packed.data_ptr()),
            ctypes.c_void_p(B_packed.data_ptr()),
            ctypes.c_void_p(A_scales.data_ptr()),
            ctypes.c_void_p(B_scales.data_ptr()),
            ctypes.c_void_p(D_out.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(K),
            ctypes.c_void_p(stream.cuda_stream),
        )

    # Warmup
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    # Timed iterations with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        run()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / ITERS
    return elapsed_ms


def bench_cublas_bf16(M, N, K):
    """Benchmark cuBLAS BF16 GEMM via torch.matmul."""
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    def run():
        torch.matmul(A, B.T)

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        run()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / ITERS
    return elapsed_ms


def main():
    lib = get_lib()

    # Verify the kernel exists
    if not hasattr(lib, "cgemm_nvfp4_bf16"):
        raise RuntimeError("cgemm_nvfp4_bf16 not found — need SM_120 build")

    print("=" * 90)
    print("Hand-written NVFP4 GEMM benchmark (GLM-4.7 shapes)")
    print("=" * 90)
    print()

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print()

    header = f"{'Layer':<16} {'M':>4} {'N':>6} {'K':>6} {'NVFP4 (ms)':>11} {'BF16 (ms)':>10} {'Speedup':>8} {'NVFP4 TFLOPS':>13}"
    print(header)
    print("-" * len(header))

    for layer_name, (K, N) in GLM47_SHAPES.items():
        for M in BATCH_SIZES:
            nvfp4_ms = bench_gemm_bf16(lib, M, N, K)
            cublas_ms = bench_cublas_bf16(M, N, K)
            speedup = cublas_ms / nvfp4_ms if nvfp4_ms > 0 else 0
            flops = 2 * M * N * K
            tflops = flops / (nvfp4_ms * 1e-3) / 1e12

            print(
                f"{layer_name:<16} {M:>4} {N:>6} {K:>6} "
                f"{nvfp4_ms:>10.3f} {cublas_ms:>10.3f} "
                f"{speedup:>7.2f}x {tflops:>12.1f}T"
            )
        print()


if __name__ == "__main__":
    main()
