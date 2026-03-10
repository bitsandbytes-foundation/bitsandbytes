"""Benchmark NVFP4 MoE pipeline vs BF16 on B200 — with CUDA graph capture.

Compares four modes:
  1. BF16 bmm (cuBLAS):       batched BF16 matmul, CUDA events timing
  2. BF16 bmm (CUDA graph):   same, but replayed from a captured graph
  3. NVFP4 pipeline (eager):  full _forward_batched (quantize→scatter→GEMM→gather)
  4. NVFP4 GEMM-only (graph): just run() replayed from CUDA graph (no quant/scatter)
  5. NVFP4 pipeline (graph):  scatter→scale→GEMM→gather captured in a single graph

Mode 4 shows the GEMM kernel throughput ceiling.
Mode 5 shows what CUDA graph capture gives for the full pipeline.

Usage (on B200):
    python bench_moe_pipeline.py
"""

import ctypes as ct
import time

import torch


WARMUP = 20
ITERS = 100


def get_ptr(t):
    return ct.c_void_p(t.data_ptr())


# ---------------------------------------------------------------------------
# 1. BF16 bmm — CUDA events
# ---------------------------------------------------------------------------

def bench_bf16_bmm(num_experts, max_M, N, K):
    """cuBLAS BF16 batched GEMM with CUDA event timing."""
    A = torch.randn(num_experts, max_M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")
    C = torch.empty(num_experts, max_M, N, dtype=torch.bfloat16, device="cuda")

    for _ in range(WARMUP):
        torch.bmm(A, B, out=C)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        torch.bmm(A, B, out=C)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


# ---------------------------------------------------------------------------
# 2. BF16 bmm — CUDA graph
# ---------------------------------------------------------------------------

def bench_bf16_bmm_graph(num_experts, max_M, N, K):
    """cuBLAS BF16 batched GEMM replayed from a CUDA graph."""
    A = torch.randn(num_experts, max_M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")
    C = torch.empty(num_experts, max_M, N, dtype=torch.bfloat16, device="cuda")

    # Warmup on default stream
    for _ in range(WARMUP):
        torch.bmm(A, B, out=C)
    torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.bmm(A, B, out=C)

    # Warm graph replay
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


# ---------------------------------------------------------------------------
# 3. NVFP4 full pipeline — eager (layer.forward)
# ---------------------------------------------------------------------------

def bench_nvfp4_eager(layer, x, expert_offsets):
    """Full NVFP4 MoE pipeline via layer.forward() — includes quant+scatter+GEMM+gather."""
    for _ in range(WARMUP):
        _ = layer(x, expert_offsets)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        _ = layer(x, expert_offsets)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


# ---------------------------------------------------------------------------
# 4. NVFP4 GEMM-only — CUDA graph (just run())
# ---------------------------------------------------------------------------

def bench_nvfp4_gemm_graph(max_M, N, K, num_experts):
    """NVFP4 batched GEMM kernel only — init once, graph-capture run().

    This measures the pure GEMM throughput ceiling with zero Python overhead.
    """
    from bitsandbytes.cextension import lib
    from bitsandbytes.functional import get_ptr

    device = torch.device("cuda")
    half_K = K // 2

    # Allocate persistent buffers
    A_bat = torch.randint(0, 255, (num_experts * max_M * half_K,),
                          dtype=torch.uint8, device=device)
    B_all = torch.randint(0, 255, (num_experts * N * half_K,),
                          dtype=torch.uint8, device=device)

    lib.cgemm_nvfp4_moe_sm100_sfa_size.restype = ct.c_size_t
    lib.cgemm_nvfp4_moe_sm100_sfb_size.restype = ct.c_size_t
    sfa_bytes = lib.cgemm_nvfp4_moe_sm100_sfa_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts))
    sfb_bytes = lib.cgemm_nvfp4_moe_sm100_sfb_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts))
    SFA = torch.randint(0, 255, (max(sfa_bytes, 1),), dtype=torch.uint8, device=device)
    SFB = torch.randint(0, 255, (max(sfb_bytes, 1),), dtype=torch.uint8, device=device)

    D_out = torch.empty(num_experts * max_M, N, dtype=torch.bfloat16, device=device)
    alpha = torch.tensor([1.0], dtype=torch.float32, device=device)

    lib.cgemm_nvfp4_moe_sm100_workspace_size.restype = ct.c_size_t
    ws_size = lib.cgemm_nvfp4_moe_sm100_workspace_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts))
    workspace = torch.empty(max(ws_size, 1), dtype=torch.uint8, device=device)

    stream = torch.cuda.current_stream()
    stream_ptr = ct.c_void_p(stream.cuda_stream)

    # Init (one-time, bakes pointers)
    lib.cgemm_nvfp4_moe_sm100_init.restype = ct.c_int
    ret = lib.cgemm_nvfp4_moe_sm100_init(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts),
        get_ptr(A_bat), get_ptr(B_all),
        get_ptr(SFA), get_ptr(SFB),
        get_ptr(D_out), get_ptr(alpha),
        get_ptr(workspace), ct.c_size_t(ws_size), stream_ptr,
    )
    if ret != 0:
        return None

    lib.cgemm_nvfp4_moe_sm100_run.restype = ct.c_int

    # Warmup run()
    for _ in range(WARMUP):
        lib.cgemm_nvfp4_moe_sm100_run(stream_ptr)
    torch.cuda.synchronize()

    # Capture graph of run() — must use the capture stream, not the default stream
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capture_stream_ptr = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        lib.cgemm_nvfp4_moe_sm100_run(capture_stream_ptr)

    # Warm graph replay
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


# ---------------------------------------------------------------------------
# 5. NVFP4 full pipeline — CUDA graph (scatter + scale + GEMM + gather)
# ---------------------------------------------------------------------------

def bench_nvfp4_pipeline_graph(layer, x, expert_offsets):
    """Capture the post-quantization pipeline in a CUDA graph.

    Captures: scatter → scale_to_blocked_batched → GEMM run → gather.
    Quantization is done eagerly before graph capture since dimensions
    must be known upfront.
    """
    from bitsandbytes.backends.cuda.ops import _batched_moe_sm100_init_if_needed
    from bitsandbytes.cextension import lib
    from bitsandbytes.functional import (
        _get_tensor_stream, get_ptr,
        moe_gather_bf16, quantize_nvfp4_raw,
    )

    N, K = layer.output_features, layer.input_features
    num_experts = layer.num_experts

    if not layer._quantized:
        layer._quantize_weights()

    expert_offsets_i32 = expert_offsets.to(torch.int32)
    tokens_per_expert = expert_offsets_i32[1:] - expert_offsets_i32[:-1]
    raw_max_M = tokens_per_expert.max().item()
    max_M = ((raw_max_M + 127) // 128) * 128
    total_tokens = expert_offsets_i32[-1].item()

    x_2d = x.reshape(-1, K).to(torch.bfloat16).contiguous()

    # Pre-quantize (this part can't be graphed due to dynamic scale)
    act_scale = x_2d.abs().max()
    global_scale = (1.0 / act_scale).to(torch.float32)
    packed_all, scales_all = quantize_nvfp4_raw(x_2d, global_scale)

    # Ensure persistent cache exists
    cache_key = (max_M, N, K, num_experts)
    if not hasattr(layer, "_batched_cache") or layer._batched_cache.get("key") != cache_key:
        dev = x.device
        W = K // 16
        n_col_blocks = (W + 3) // 4
        n_row_blocks = (max_M + 127) // 128
        sfa_per_expert = n_row_blocks * n_col_blocks * 512
        sfa_total = num_experts * sfa_per_expert

        layer._batched_cache = {
            "key": cache_key,
            "A_batched": torch.empty(num_experts * max_M * (K // 2), dtype=torch.uint8, device=dev),
            "SFA_batched": torch.zeros(sfa_total, dtype=torch.uint8, device=dev),
            "D_out": torch.empty(num_experts * max_M, N, dtype=torch.bfloat16, device=dev),
            "alpha_dev": torch.empty(1, dtype=torch.float32, device=dev),
        }
    cache = layer._batched_cache

    # Pre-allocate gather output (persistent for graph capture)
    gather_out = torch.empty(total_tokens * N, dtype=torch.bfloat16, device=x.device)

    W = K // 16
    n_col_blocks = (W + 3) // 4
    n_row_blocks = (max_M + 127) // 128
    sfa_per_expert = n_row_blocks * n_col_blocks * 512

    expert_row_offsets = expert_offsets_i32[:-1]
    expert_M_dev = tokens_per_expert.to(torch.int32)
    expert_out_offsets = torch.arange(
        num_experts, dtype=torch.int32, device=x.device,
    ) * sfa_per_expert

    stream = _get_tensor_stream(x_2d)

    # Set alpha
    cache["alpha_dev"].copy_(
        (act_scale * layer.weight_tensor_scale).to(torch.float32).reshape(1)
    )

    # Init GEMM (one-time, bakes pointers)
    _batched_moe_sm100_init_if_needed(
        cache["A_batched"], layer.weight_packed,
        cache["SFA_batched"], layer.weight_scales_batched,
        cache["D_out"], cache["alpha_dev"],
        max_M, N, K, num_experts, stream,
    )

    # Define the pipeline kernel sequence — uses current stream (changes during graph capture)
    def pipeline():
        s = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        # Scatter FP4 data
        lib.cmoe_scatter_nvfp4(
            get_ptr(packed_all), get_ptr(cache["A_batched"]),
            get_ptr(expert_offsets_i32),
            ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts), s,
        )
        # Swizzle scales
        cache["SFA_batched"].zero_()
        lib.cscale_to_blocked_batched(
            get_ptr(scales_all), get_ptr(cache["SFA_batched"]),
            get_ptr(expert_row_offsets), get_ptr(expert_M_dev),
            get_ptr(expert_out_offsets),
            ct.c_int(W), ct.c_int(num_experts), ct.c_int(n_row_blocks), s,
        )
        # GEMM run
        lib.cgemm_nvfp4_moe_sm100_run(s)
        # Gather
        lib.cmoe_gather_bf16(
            get_ptr(cache["D_out"].view(-1)), get_ptr(gather_out),
            get_ptr(expert_offsets_i32),
            ct.c_int(max_M), ct.c_int(N), ct.c_int(num_experts),
            ct.c_int(total_tokens), s,
        )

    # Warmup
    for _ in range(WARMUP):
        pipeline()
    torch.cuda.synchronize()

    # Capture graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline()

    # Warm graph replay
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_config(name, num_experts, K, N, tokens_per_expert):
    """Run all benchmark variants for a given MoE configuration."""
    from bitsandbytes.nn.modules import LinearNVFP4MoE

    total_tokens = sum(tokens_per_expert)
    max_M_raw = max(tokens_per_expert)
    max_M = ((max_M_raw + 127) // 128) * 128

    offsets = [0]
    for n in tokens_per_expert:
        offsets.append(offsets[-1] + n)
    expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    x = torch.randn(total_tokens, K, dtype=torch.bfloat16, device="cuda")

    # NVFP4 layer
    layer = LinearNVFP4MoE(num_experts, K, N, bias=False)
    torch.nn.init.normal_(layer.weight.data, std=0.02)
    layer = layer.cuda()

    # Effective FLOPs (padded — what the hardware actually computes)
    padded_flops = 2 * num_experts * max_M * N * K

    results = {}

    # 1. BF16 bmm (events)
    results["bf16_events"] = bench_bf16_bmm(num_experts, max_M, N, K)

    # 2. BF16 bmm (graph)
    try:
        results["bf16_graph"] = bench_bf16_bmm_graph(num_experts, max_M, N, K)
    except Exception as e:
        results["bf16_graph"] = None

    # 3. NVFP4 eager pipeline
    results["nvfp4_eager"] = bench_nvfp4_eager(layer, x, expert_offsets)

    # 4. NVFP4 GEMM-only (graph)
    try:
        results["nvfp4_gemm_graph"] = bench_nvfp4_gemm_graph(max_M, N, K, num_experts)
    except Exception as e:
        results["nvfp4_gemm_graph"] = None

    # 5. NVFP4 pipeline (graph)
    try:
        results["nvfp4_pipe_graph"] = bench_nvfp4_pipeline_graph(layer, x, expert_offsets)
    except Exception as e:
        results["nvfp4_pipe_graph"] = None

    results["padded_flops"] = padded_flops
    return results


def fmt(ms):
    return f"{ms:.3f}" if ms is not None else "  FAIL"


def tflops(flops, ms):
    if ms is None or ms <= 0:
        return "    -"
    return f"{flops / (ms * 1e-3) / 1e12:.1f}"


def main():
    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("=" * 130)
    print(f"NVFP4 MoE Pipeline Benchmark — CUDA Graph vs BF16")
    print(f"GPU: {gpu}  (SM {cap[0]}.{cap[1]})")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print("=" * 130)

    configs = [
        # --- GLM-4.7 gate_up (K=4096, N=13696) ---
        ("gate_up 8e×8tok",   8, 4096, 13696, [8]*8),
        ("gate_up 8e×32tok",  8, 4096, 13696, [32]*8),
        ("gate_up 8e×64tok",  8, 4096, 13696, [64]*8),
        ("gate_up 8e×128tok", 8, 4096, 13696, [128]*8),
        ("gate_up 8e×512tok", 8, 4096, 13696, [512]*8),
        ("gate_up 8e skewed", 8, 4096, 13696, [128, 64, 32, 16, 8, 4, 2, 1]),
        # --- GLM-4.7 down (K=13696, N=4096) ---
        ("down 8e×8tok",   8, 13696, 4096, [8]*8),
        ("down 8e×32tok",  8, 13696, 4096, [32]*8),
        ("down 8e×64tok",  8, 13696, 4096, [64]*8),
        ("down 8e×128tok", 8, 13696, 4096, [128]*8),
        ("down 8e×512tok", 8, 13696, 4096, [512]*8),
        ("down 8e skewed", 8, 13696, 4096, [128, 64, 32, 16, 8, 4, 2, 1]),
    ]

    header = (
        f"{'Config':<23} "
        f"{'BF16 evt':>9} {'BF16 grp':>9} "
        f"{'FP4 eagr':>9} {'FP4 Ggrp':>9} {'FP4 Pgrp':>9} "
        f"{'FP4G/BF':>8} {'FP4P/BF':>8} "
        f"{'BF T':>7} {'G T':>7} {'P T':>7}"
    )
    print()
    print(header)
    print("-" * len(header))

    for name, ne, K, N, tpe in configs:
        r = run_config(name, ne, K, N, tpe)
        f = r["padded_flops"]

        # Speedup: NVFP4 graph vs BF16 graph (or events if graph failed)
        bf_ref = r["bf16_graph"] or r["bf16_events"]
        gemm_spd = bf_ref / r["nvfp4_gemm_graph"] if r["nvfp4_gemm_graph"] else 0
        pipe_spd = bf_ref / r["nvfp4_pipe_graph"] if r["nvfp4_pipe_graph"] else 0

        print(
            f"{name:<23} "
            f"{fmt(r['bf16_events']):>9} {fmt(r['bf16_graph']):>9} "
            f"{fmt(r['nvfp4_eager']):>9} {fmt(r['nvfp4_gemm_graph']):>9} {fmt(r['nvfp4_pipe_graph']):>9} "
            f"{gemm_spd:>7.2f}x {pipe_spd:>7.2f}x "
            f"{tflops(f, bf_ref):>7} {tflops(f, r['nvfp4_gemm_graph']):>7} {tflops(f, r['nvfp4_pipe_graph']):>7}"
        )

    print()
    print("Legend:")
    print("  BF16 evt  = cuBLAS BF16 torch.bmm, CUDA event timing")
    print("  BF16 grp  = cuBLAS BF16 torch.bmm, CUDA graph replay")
    print("  FP4 eagr  = NVFP4 full pipeline (quantize+scatter+GEMM+gather), eager Python")
    print("  FP4 Ggrp  = NVFP4 GEMM kernel only, CUDA graph replay (throughput ceiling)")
    print("  FP4 Pgrp  = NVFP4 pipeline (scatter+scale+GEMM+gather), CUDA graph replay")
    print("  FP4G/BF   = speedup of NVFP4 GEMM graph vs BF16 graph")
    print("  FP4P/BF   = speedup of NVFP4 pipeline graph vs BF16 graph")
    print("  T columns = effective TFLOPS (padded dimensions)")
    print("  Memory:     NVFP4 weights are 3.6x smaller than BF16")


if __name__ == "__main__":
    main()
