"""Benchmark NVFP4 MoE pipeline vs BF16 on B200 — CUDA graph capture.

Compares three modes:
  1. BF16 bmm (CUDA graph):    cuBLAS BF16 batched matmul, graph replay
  2. NVFP4 pipeline (graph):   scatter→scale→GEMM→gather in a single graph
  3. BF16 bmm (eager events):  cuBLAS BF16 batched matmul, event timing

Mode 2 is the key result: the full NVFP4 pipeline with zero Python overhead.

Usage (on B200):
    python bench_moe_pipeline.py
"""

import ctypes as ct

import torch


WARMUP = 20
ITERS = 100


def get_ptr(t):
    return ct.c_void_p(t.data_ptr())


# ---------------------------------------------------------------------------
# BF16 bmm — CUDA graph
# ---------------------------------------------------------------------------

def bench_bf16_graph(num_experts, max_M, N, K):
    """cuBLAS BF16 batched GEMM replayed from a CUDA graph."""
    A = torch.randn(num_experts, max_M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")
    C = torch.empty(num_experts, max_M, N, dtype=torch.bfloat16, device="cuda")

    for _ in range(WARMUP):
        torch.bmm(A, B, out=C)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.bmm(A, B, out=C)

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
# BF16 bmm — eager (CUDA events)
# ---------------------------------------------------------------------------

def bench_bf16_eager(num_experts, max_M, N, K):
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
# NVFP4 full pipeline — CUDA graph (scatter + scale + GEMM + gather)
# ---------------------------------------------------------------------------

def bench_nvfp4_pipeline_graph(layer, x, expert_offsets):
    """Capture scatter → scale_swizzle → GEMM run → gather in one CUDA graph.

    Quantization is done eagerly before capture (dynamic scale needs host sync).
    Everything else is graph-captured with zero Python overhead on replay.
    """
    from bitsandbytes.cextension import lib
    from bitsandbytes.functional import get_ptr, quantize_nvfp4_raw

    N, K = layer.output_features, layer.input_features
    num_experts = layer.num_experts
    dev = x.device

    if not layer._quantized:
        layer._quantize_weights()

    expert_offsets_i32 = expert_offsets.to(torch.int32)
    tokens_per_expert = expert_offsets_i32[1:] - expert_offsets_i32[:-1]
    raw_max_M = tokens_per_expert.max().item()
    max_M = ((raw_max_M + 127) // 128) * 128
    total_tokens = expert_offsets_i32[-1].item()

    x_2d = x.reshape(-1, K).to(torch.bfloat16).contiguous()

    # Pre-quantize activations (host sync for scale — can't be graphed)
    act_scale = x_2d.abs().max()
    global_scale = (1.0 / act_scale).to(torch.float32)
    packed_all, scales_all = quantize_nvfp4_raw(x_2d, global_scale)

    # Persistent buffers for graph capture
    W = K // 16
    n_col_blocks = (W + 3) // 4
    n_row_blocks = (max_M + 127) // 128
    sfa_per_expert = n_row_blocks * n_col_blocks * 512
    sfa_total = num_experts * sfa_per_expert

    A_batched = torch.empty(num_experts * max_M * (K // 2), dtype=torch.uint8, device=dev)
    SFA_batched = torch.zeros(sfa_total, dtype=torch.uint8, device=dev)
    D_out = torch.empty(num_experts * max_M, N, dtype=torch.bfloat16, device=dev)
    alpha_dev = (act_scale * layer.weight_tensor_scale).to(torch.float32).reshape(1).to(dev)
    gather_out = torch.empty(total_tokens * N, dtype=torch.bfloat16, device=dev)

    expert_row_offsets = expert_offsets_i32[:-1]
    expert_M_dev = tokens_per_expert.to(torch.int32)
    expert_out_offsets = torch.arange(num_experts, dtype=torch.int32, device=dev) * sfa_per_expert

    # GEMM init — call once outside graph capture, bakes pointers into s_state
    stream = torch.cuda.current_stream()
    stream_ptr = ct.c_void_p(stream.cuda_stream)

    lib.cgemm_nvfp4_moe_sm100_sfa_size.restype = ct.c_size_t
    lib.cgemm_nvfp4_moe_sm100_sfb_size.restype = ct.c_size_t
    lib.cgemm_nvfp4_moe_sm100_workspace_size.restype = ct.c_size_t
    lib.cgemm_nvfp4_moe_sm100_init.restype = ct.c_int
    lib.cgemm_nvfp4_moe_sm100_run.restype = ct.c_int

    ws_size = lib.cgemm_nvfp4_moe_sm100_workspace_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts))
    workspace = torch.empty(max(ws_size, 1), dtype=torch.uint8, device=dev)

    ret = lib.cgemm_nvfp4_moe_sm100_init(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts),
        get_ptr(A_batched), get_ptr(layer.weight_packed),
        get_ptr(SFA_batched), get_ptr(layer.weight_scales_batched),
        get_ptr(D_out), get_ptr(alpha_dev),
        get_ptr(workspace), ct.c_size_t(ws_size), stream_ptr,
    )
    if ret != 0:
        return None

    # The pipeline: scatter + scale_swizzle + GEMM run + gather
    def pipeline():
        s = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        lib.cmoe_scatter_nvfp4(
            get_ptr(packed_all), get_ptr(A_batched),
            get_ptr(expert_offsets_i32),
            ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts), s,
        )
        SFA_batched.zero_()
        lib.cscale_to_blocked_batched(
            get_ptr(scales_all), get_ptr(SFA_batched),
            get_ptr(expert_row_offsets), get_ptr(expert_M_dev),
            get_ptr(expert_out_offsets),
            ct.c_int(W), ct.c_int(num_experts), ct.c_int(n_row_blocks), s,
        )
        lib.cgemm_nvfp4_moe_sm100_run(s)
        lib.cmoe_gather_bf16(
            get_ptr(D_out.view(-1)), get_ptr(gather_out),
            get_ptr(expert_offsets_i32),
            ct.c_int(max_M), ct.c_int(N), ct.c_int(num_experts), s,
        )

    # Warmup on default stream
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

    # Timed measurement
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

    layer = LinearNVFP4MoE(num_experts, K, N, bias=False)
    torch.nn.init.normal_(layer.weight.data, std=0.02)
    layer = layer.cuda()

    padded_flops = 2 * num_experts * max_M * N * K

    results = {}
    results["bf16_graph"] = bench_bf16_graph(num_experts, max_M, N, K)
    results["bf16_eager"] = bench_bf16_eager(num_experts, max_M, N, K)

    try:
        results["nvfp4_graph"] = bench_nvfp4_pipeline_graph(layer, x, expert_offsets)
    except Exception as e:
        print(f"  [{name}] NVFP4 pipeline graph failed: {e}")
        results["nvfp4_graph"] = None

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
    print("=" * 100)
    print(f"NVFP4 MoE Pipeline Benchmark — CUDA Graph vs BF16")
    print(f"GPU: {gpu}  (SM {cap[0]}.{cap[1]})")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print("=" * 100)

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
        f"{'BF16 grp':>9} {'BF16 egr':>9} "
        f"{'FP4 pipe':>9} "
        f"{'FP4/BF16':>9} "
        f"{'BF16 T':>8} {'FP4 T':>8}"
    )
    print()
    print(header)
    print("-" * len(header))

    for name, ne, K, N, tpe in configs:
        r = run_config(name, ne, K, N, tpe)
        f = r["padded_flops"]

        bf_ref = r["bf16_graph"]
        speedup = bf_ref / r["nvfp4_graph"] if r["nvfp4_graph"] else 0

        print(
            f"{name:<23} "
            f"{fmt(r['bf16_graph']):>9} {fmt(r['bf16_eager']):>9} "
            f"{fmt(r['nvfp4_graph']):>9} "
            f"{speedup:>8.2f}x "
            f"{tflops(f, bf_ref):>8} {tflops(f, r['nvfp4_graph']):>8}"
        )

    print()
    print("Legend:")
    print("  BF16 grp  = cuBLAS BF16 torch.bmm, CUDA graph replay")
    print("  BF16 egr  = cuBLAS BF16 torch.bmm, CUDA event timing")
    print("  FP4 pipe  = NVFP4 pipeline (scatter+scale+GEMM+gather), CUDA graph replay")
    print("  FP4/BF16  = speedup of NVFP4 pipeline graph vs BF16 graph")
    print("  T columns = effective TFLOPS (padded dimensions)")
    print("  Note: NVFP4 weights are 3.6x smaller than BF16")


if __name__ == "__main__":
    main()
