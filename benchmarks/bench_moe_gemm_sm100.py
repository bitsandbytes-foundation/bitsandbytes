"""Benchmark NVFP4 MoE GEMM kernels on SM_100 (B200) with CUDA graph timing.

Compares:
  - Batched NVFP4 GEMM (fixed max_M, CUDA-graph friendly)
  - cuBLAS BF16 batched GEMM baseline
  - Dense NVFP4 CUTLASS GEMM (single expert, for reference)

Uses CUDA graphs for accurate kernel timing (no Python dispatch overhead).
The grouped NVFP4 GEMM uses CUDA events (cannot use graphs due to host-side
metadata computation and cudaMemcpyAsync per call).

Usage (on B200):
    python benchmarks/bench_moe_gemm_sm100.py
"""

import ctypes as ct
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# GLM-4.7 (352B MoE) shapes: (K_hidden, N_output)
MOE_SHAPES = {
    "gate_up":  (4096, 13696),
    "down":     (13696, 4096),
}

# Tokens per expert for different scenarios
# max_M is the max across experts; others are padded to max_M for batched
EXPERT_CONFIGS = [
    # (label, num_experts, tokens_per_expert_list)
    ("8e_uniform_8",   8, [8]*8),
    ("8e_uniform_32",  8, [32]*8),
    ("8e_uniform_64",  8, [64]*8),
    ("8e_uniform_128", 8, [128]*8),
    ("8e_skewed",      8, [128, 64, 32, 16, 8, 4, 2, 1]),
    ("8e_sparse",      8, [128, 0, 64, 0, 32, 0, 16, 0]),
]

WARMUP = 20
ITERS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lib():
    """Load the bitsandbytes CUDA library."""
    lib_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bitsandbytes",
    )
    for suffix in ["cuda128", "cuda130", "cuda131"]:
        lib_path = os.path.join(lib_dir, f"libbitsandbytes_{suffix}.so")
        if os.path.exists(lib_path):
            return ct.cdll.LoadLibrary(lib_path)
    raise RuntimeError(f"Could not find bitsandbytes CUDA library in {lib_dir}")


def get_ptr(t):
    return ct.c_void_p(t.data_ptr())


def sfa_size_batched(lib, N, max_M, K, num_experts):
    lib.cgemm_nvfp4_moe_sm100_sfa_size.restype = ct.c_size_t
    return lib.cgemm_nvfp4_moe_sm100_sfa_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts))


def sfb_size_batched(lib, N, max_M, K, num_experts):
    lib.cgemm_nvfp4_moe_sm100_sfb_size.restype = ct.c_size_t
    return lib.cgemm_nvfp4_moe_sm100_sfb_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts))


# ---------------------------------------------------------------------------
# Benchmark: Batched NVFP4 GEMM with CUDA graph
# ---------------------------------------------------------------------------

def bench_batched_nvfp4(lib, max_M, N, K, num_experts):
    """Benchmark the batched NVFP4 MoE GEMM using CUDA graph capture."""
    device = torch.device("cuda")
    half_K = K // 2

    # Allocate packed FP4 data
    A_batched = torch.randint(0, 255, (num_experts * max_M * half_K,),
                              dtype=torch.uint8, device=device)
    B_batched = torch.randint(0, 255, (num_experts * N * half_K,),
                              dtype=torch.uint8, device=device)

    # Scale factor buffers
    sfa_bytes = sfa_size_batched(lib, N, max_M, K, num_experts)
    sfb_bytes = sfb_size_batched(lib, N, max_M, K, num_experts)
    SFA = torch.randint(0, 255, (max(sfa_bytes, 1),), dtype=torch.uint8, device=device)
    SFB = torch.randint(0, 255, (max(sfb_bytes, 1),), dtype=torch.uint8, device=device)

    D_out = torch.empty(num_experts * max_M * N, dtype=torch.bfloat16, device=device)

    # Workspace
    lib.cgemm_nvfp4_moe_sm100_workspace_size.restype = ct.c_size_t
    ws_size = lib.cgemm_nvfp4_moe_sm100_workspace_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts))
    workspace = torch.empty(max(ws_size, 1), dtype=torch.uint8, device=device)

    # Initialize (one-time)
    lib.cgemm_nvfp4_moe_sm100_init.restype = ct.c_int
    ret = lib.cgemm_nvfp4_moe_sm100_init(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts),
        get_ptr(workspace), ct.c_size_t(ws_size))
    if ret != 0:
        return None  # Init failed

    stream = torch.cuda.current_stream()
    stream_ptr = ct.c_void_p(stream.cuda_stream)

    lib.cgemm_nvfp4_moe_sm100_run.restype = ct.c_int

    def run_kernel():
        lib.cgemm_nvfp4_moe_sm100_run(
            get_ptr(A_batched), get_ptr(B_batched),
            get_ptr(SFA), get_ptr(SFB),
            get_ptr(D_out),
            ct.c_float(1.0), stream_ptr)

    # Warmup
    for _ in range(WARMUP):
        run_kernel()
    torch.cuda.synchronize()

    # Timed iterations with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        run_kernel()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / ITERS
    return elapsed_ms


# ---------------------------------------------------------------------------
# Benchmark: Grouped NVFP4 GEMM (no CUDA graph — host-side metadata per call)
# ---------------------------------------------------------------------------

def has_grouped_nvfp4(lib):
    """Check if grouped NVFP4 kernel helpers are available."""
    return (hasattr(lib, "cgemm_nvfp4_grouped_sm100_fused")
            and hasattr(lib, "cgemm_nvfp4_grouped_sm100_meta_size")
            and hasattr(lib, "cgemm_nvfp4_grouped_sm100_workspace_size"))


def bench_grouped_nvfp4(lib, tokens_per_expert, N, K, num_experts):
    """Benchmark the grouped NVFP4 GEMM using the raw cutlass function.

    Uses the raw cgemm_nvfp4_grouped_cutlass_sm100 with pre-allocated per-expert
    buffers. Cannot use CUDA graphs due to host-side metadata per call.
    Returns None if the kernel is not available.
    """
    if not hasattr(lib, "cgemm_nvfp4_grouped_cutlass_sm100"):
        return None

    device = torch.device("cuda")
    half_K = K // 2
    scale_W = K // 16

    total_tokens = sum(tokens_per_expert)
    if total_tokens == 0:
        return None

    # Per-expert SFA/SFB sizes (swizzled layout)
    n_col_blocks = (scale_W + 3) // 4
    n_sfb_row_blocks = (N + 127) // 128

    # Allocate per-expert buffers
    A_list, B_list, SFA_list, SFB_list, D_list = [], [], [], [], []
    for e in range(num_experts):
        M_e = tokens_per_expert[e]
        if M_e == 0:
            M_e = 1  # CUTLASS needs at least 1 row
        n_sfa_row_blocks = (M_e + 127) // 128
        sfa_bytes = n_sfa_row_blocks * n_col_blocks * 512
        sfb_bytes = n_sfb_row_blocks * n_col_blocks * 512

        A_list.append(torch.randint(0, 255, (M_e * half_K,), dtype=torch.uint8, device=device))
        B_list.append(torch.randint(0, 255, (N * half_K,), dtype=torch.uint8, device=device))
        SFA_list.append(torch.randint(0, 255, (max(sfa_bytes, 1),), dtype=torch.uint8, device=device))
        SFB_list.append(torch.randint(0, 255, (max(sfb_bytes, 1),), dtype=torch.uint8, device=device))
        D_list.append(torch.empty(M_e, N, dtype=torch.bfloat16, device=device))

    # Build host pointer arrays (int64 raw pointer values)
    host_ptr_A = (ct.c_int64 * num_experts)(*[t.data_ptr() for t in A_list])
    host_ptr_B = (ct.c_int64 * num_experts)(*[t.data_ptr() for t in B_list])
    host_ptr_SFA = (ct.c_int64 * num_experts)(*[t.data_ptr() for t in SFA_list])
    host_ptr_SFB = (ct.c_int64 * num_experts)(*[t.data_ptr() for t in SFB_list])
    host_ptr_D = (ct.c_int64 * num_experts)(*[t.data_ptr() for t in D_list])

    M_arr = (ct.c_int * num_experts)(*[max(t, 1) for t in tokens_per_expert])

    # Metadata and workspace — use generous sizes
    # Meta: ~2KB per expert (stride arrays, pointer arrays, problem shapes)
    meta_size = num_experts * 2048
    metadata_dev = torch.empty(meta_size, dtype=torch.uint8, device=device)
    # Workspace: ~16MB should be enough for any configuration
    ws_size = 16 * 1024 * 1024
    workspace_dev = torch.empty(ws_size, dtype=torch.uint8, device=device)

    stream = torch.cuda.current_stream()
    stream_ptr = ct.c_void_p(stream.cuda_stream)

    def run_kernel():
        lib.cgemm_nvfp4_grouped_cutlass_sm100(
            host_ptr_A, host_ptr_B,
            host_ptr_SFA, host_ptr_SFB, host_ptr_D,
            M_arr,
            ct.c_int(N), ct.c_int(K), ct.c_int(num_experts),
            ct.c_float(1.0),
            get_ptr(metadata_dev), get_ptr(workspace_dev),
            ct.c_size_t(ws_size),
            stream_ptr)

    # Warmup (catch any init errors)
    try:
        for _ in range(WARMUP):
            run_kernel()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  [grouped warmup failed: {e}]")
        return None

    # Timed iterations (CUDA events, no graph)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        run_kernel()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / ITERS
    return elapsed_ms


# ---------------------------------------------------------------------------
# Benchmark: cuBLAS BF16 batched GEMM with CUDA graph
# ---------------------------------------------------------------------------

def bench_dense_nvfp4(lib, M, N, K):
    """Benchmark dense NVFP4 GEMM (single GEMM, no MoE batching).

    Uses cgemm_nvfp4_cutlass_sm100 for a single M×N×K GEMM.
    Returns None if kernel is not available.
    """
    if not hasattr(lib, "cgemm_nvfp4_cutlass_sm100"):
        return None

    device = torch.device("cuda")
    half_K = K // 2

    # Use per-expert SFA/SFB size (L=1)
    lib.cgemm_nvfp4_moe_sm100_sfa_size_per_expert.restype = ct.c_size_t
    lib.cgemm_nvfp4_moe_sm100_sfb_size_per_expert.restype = ct.c_size_t
    sfa_bytes = lib.cgemm_nvfp4_moe_sm100_sfa_size_per_expert(
        ct.c_int(N), ct.c_int(M), ct.c_int(K))
    sfb_bytes = lib.cgemm_nvfp4_moe_sm100_sfb_size_per_expert(
        ct.c_int(N), ct.c_int(M), ct.c_int(K))

    A = torch.randint(0, 255, (M * half_K,), dtype=torch.uint8, device=device)
    B = torch.randint(0, 255, (N * half_K,), dtype=torch.uint8, device=device)
    SFA = torch.randint(0, 255, (max(sfa_bytes, 1),), dtype=torch.uint8, device=device)
    SFB = torch.randint(0, 255, (max(sfb_bytes, 1),), dtype=torch.uint8, device=device)
    D = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    alpha_dev = torch.tensor([1.0], dtype=torch.float32, device=device)

    stream = torch.cuda.current_stream()

    def run_kernel():
        lib.cgemm_nvfp4_cutlass_sm100(
            get_ptr(A), get_ptr(B),
            get_ptr(SFA), get_ptr(SFB),
            get_ptr(D),
            ct.c_int(M), ct.c_int(N), ct.c_int(K),
            get_ptr(alpha_dev),
            ct.c_void_p(stream.cuda_stream))

    # Warmup
    for _ in range(WARMUP):
        run_kernel()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        run_kernel()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / ITERS
    return elapsed_ms


def bench_cublas_bf16(max_M, N, K, num_experts):
    """Benchmark cuBLAS BF16 batched GEMM using CUDA graph capture.

    Simulates MoE: num_experts independent GEMMs of shape (max_M, K) @ (K, N).
    Uses torch.bmm for a single batched launch.
    """
    device = torch.device("cuda")

    # Batched matmul: (num_experts, max_M, K) @ (num_experts, K, N)
    A = torch.randn(num_experts, max_M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device=device)
    C = torch.empty(num_experts, max_M, N, dtype=torch.bfloat16, device=device)

    def run_kernel():
        torch.bmm(A, B, out=C)

    # Warmup
    for _ in range(WARMUP):
        run_kernel()
    torch.cuda.synchronize()

    # Timed iterations with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        run_kernel()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / ITERS
    return elapsed_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    lib = get_lib()

    # Check SM_100 kernels are available
    if not hasattr(lib, "cgemm_nvfp4_moe_sm100_init"):
        print("ERROR: cgemm_nvfp4_moe_sm100_init not found — need SM_100 build")
        sys.exit(1)

    # Grouped needs fused dispatch + helper functions; raw function alone segfaults
    # due to unknown metadata/workspace sizes. Skip unless fused API is available.
    has_grouped = has_grouped_nvfp4(lib)
    if not has_grouped:
        print("NOTE: grouped NVFP4 fused API not available — skipping grouped benchmark")

    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("=" * 110)
    print(f"NVFP4 MoE GEMM Benchmark — SM_100 (B200)")
    print(f"GPU: {gpu_name}  (SM {cap[0]}.{cap[1]})")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print(f"Timing: CUDA events (kernel-only, no Python dispatch overhead)")
    print(f"Grouped NVFP4: {'available' if has_grouped else 'NOT available (skipped)'}")
    print("=" * 110)

    for shape_name, (K, N) in MOE_SHAPES.items():
        print(f"\n{'─' * 100}")
        print(f"  Shape: {shape_name}  (K={K}, N={N})")
        print(f"{'─' * 100}")

        header = (
            f"{'Config':<20} {'E':>3} {'max_M':>6} {'TotalTok':>9} "
            f"{'NVFP4(ms)':>10} {'BF16(ms)':>10} "
            f"{'Speedup':>8} "
            f"{'NVFP4 T':>10} {'BF16 T':>10}"
        )
        print(header)
        print("-" * len(header))

        for label, num_experts, tpe in EXPERT_CONFIGS:
            max_M = max(tpe) if max(tpe) > 0 else 1
            total_tokens = sum(tpe)

            # Compute effective FLOPs (padded dimensions for both)
            total_flops = 2 * num_experts * max_M * N * K

            # Run benchmarks
            batched_ms = bench_batched_nvfp4(lib, max_M, N, K, num_experts)
            bf16_ms = bench_cublas_bf16(max_M, N, K, num_experts)

            # Compute speedups and TFLOPS
            speedup = bf16_ms / batched_ms if batched_ms and batched_ms > 0 else 0

            nvfp4_tflops = total_flops / (batched_ms * 1e-3) / 1e12 if batched_ms and batched_ms > 0 else 0
            bf16_tflops = total_flops / (bf16_ms * 1e-3) / 1e12 if bf16_ms and bf16_ms > 0 else 0

            b_str = f"{batched_ms:.3f}" if batched_ms else "FAIL"
            bf_str = f"{bf16_ms:.3f}" if bf16_ms else "FAIL"

            print(
                f"{label:<20} {num_experts:>3} {max_M:>6} {total_tokens:>9} "
                f"{b_str:>10} {bf_str:>10} "
                f"{speedup:>7.2f}x "
                f"{nvfp4_tflops:>9.1f}T {bf16_tflops:>9.1f}T"
            )

    # Dense NVFP4 comparison (single GEMM, no MoE batching)
    has_dense = hasattr(lib, "cgemm_nvfp4_cutlass_sm100")
    if has_dense:
        print(f"\n{'═' * 100}")
        print(f"  Dense NVFP4 vs BF16 (single GEMM, no MoE batching)")
        print(f"{'═' * 100}")

        DENSE_M_SIZES = [1, 8, 32, 128, 512, 1024]
        header = (
            f"{'Shape':<20} {'M':>6} "
            f"{'NVFP4(ms)':>10} {'BF16(ms)':>10} "
            f"{'Speedup':>8} "
            f"{'NVFP4 T':>10} {'BF16 T':>10}"
        )
        print(header)
        print("-" * len(header))

        for shape_name, (K, N) in MOE_SHAPES.items():
            for M in DENSE_M_SIZES:
                flops = 2 * M * N * K
                nvfp4_ms = bench_dense_nvfp4(lib, M, N, K)
                # Dense BF16: single matmul, not batched
                A_bf = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
                B_bf = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
                for _ in range(WARMUP):
                    torch.matmul(A_bf, B_bf)
                torch.cuda.synchronize()
                se = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                se.record()
                for _ in range(ITERS):
                    torch.matmul(A_bf, B_bf)
                ee.record()
                torch.cuda.synchronize()
                bf16_ms = se.elapsed_time(ee) / ITERS

                speedup = bf16_ms / nvfp4_ms if nvfp4_ms and nvfp4_ms > 0 else 0
                nvfp4_t = flops / (nvfp4_ms * 1e-3) / 1e12 if nvfp4_ms and nvfp4_ms > 0 else 0
                bf16_t = flops / (bf16_ms * 1e-3) / 1e12 if bf16_ms and bf16_ms > 0 else 0

                n_str = f"{nvfp4_ms:.3f}" if nvfp4_ms else "FAIL"
                b_str = f"{bf16_ms:.3f}" if bf16_ms else "FAIL"

                print(
                    f"{shape_name:<20} {M:>6} "
                    f"{n_str:>10} {b_str:>10} "
                    f"{speedup:>7.2f}x "
                    f"{nvfp4_t:>9.1f}T {bf16_t:>9.1f}T"
                )
            print()

    print()
    print("Notes:")
    print("  - NVFP4: Batched CUTLASS GEMM (fixed max_M per expert, TMA + block-scaled FP4)")
    print("  - BF16: cuBLAS torch.bmm (batched) or torch.matmul (dense)")
    print("  - Speedup: NVFP4 vs BF16 (>1x = NVFP4 faster)")
    print("  - TFLOPS: effective throughput based on padded dimensions")
    print("  - Memory: NVFP4 weights are 3.6x smaller than BF16")


if __name__ == "__main__":
    main()
