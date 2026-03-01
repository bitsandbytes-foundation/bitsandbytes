"""
CPU→GPU Weight Streaming Benchmark
Tests whether layer-by-layer weight streaming from CPU (or NVMe) to GPU
can overlap with compute to hide transfer latency.

Tests:
  1. Raw PCIe H2D bandwidth (pinned vs pageable memory)
  2. Raw NVMe sequential read bandwidth
  3. Raw matmul throughput at various batch sizes
  4. Overlap test: simultaneous transfer + compute on separate streams
  5. Full pipeline: double-buffered layer streaming with real matmul
  6. NVMe→CPU→GPU pipeline: end-to-end from mmap'd safetensors file

Usage:
  python stream_bench.py                    # auto-detect layer size
  python stream_bench.py --layer-mb 470     # Llama-70B layer size
  python stream_bench.py --layer-mb 2250    # GLM-4.7 layer size
  python stream_bench.py --nvme /path/to/nvme/mount  # test NVMe reads
"""

import argparse
import os
import time

import torch
import torch.cuda

# ─── Helpers ───


def fmt_bw(gb_per_s):
    if gb_per_s >= 1:
        return f"{gb_per_s:.2f} GB/s"
    return f"{gb_per_s * 1000:.1f} MB/s"


def fmt_time(ms):
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.1f}ms"


def sync():
    torch.cuda.synchronize()


# ─── Test 1: Raw PCIe bandwidth ───


def test_pcie_bandwidth(size_mb=512, n_iter=10):
    """Measure actual H2D transfer bandwidth with pinned and pageable memory."""
    print(f"\n{'=' * 70}")
    print(f"  TEST 1: PCIe Host→Device Bandwidth ({size_mb} MB)")
    print(f"{'=' * 70}")

    nbytes = size_mb * 1024 * 1024
    n_elem = nbytes // 2  # float16

    # Pinned memory
    cpu_pinned = torch.empty(n_elem, dtype=torch.float16, pin_memory=True)
    cpu_pinned.fill_(1.0)
    gpu_buf = torch.empty(n_elem, dtype=torch.float16, device="cuda")

    # Warmup
    for _ in range(3):
        gpu_buf.copy_(cpu_pinned, non_blocking=False)
    sync()

    # Timed
    start = time.perf_counter()
    for _ in range(n_iter):
        gpu_buf.copy_(cpu_pinned, non_blocking=False)
    sync()
    elapsed = time.perf_counter() - start
    pinned_bw = (size_mb * n_iter / 1024) / elapsed

    # Pageable memory
    cpu_page = torch.empty(n_elem, dtype=torch.float16)
    cpu_page.fill_(1.0)

    for _ in range(3):
        gpu_buf.copy_(cpu_page, non_blocking=False)
    sync()

    start = time.perf_counter()
    for _ in range(n_iter):
        gpu_buf.copy_(cpu_page, non_blocking=False)
    sync()
    elapsed = time.perf_counter() - start
    page_bw = (size_mb * n_iter / 1024) / elapsed

    # Async pinned (non-blocking on a separate stream)
    stream = torch.cuda.Stream()
    for _ in range(3):
        with torch.cuda.stream(stream):
            gpu_buf.copy_(cpu_pinned, non_blocking=True)
        stream.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record(stream)
    for _ in range(n_iter):
        with torch.cuda.stream(stream):
            gpu_buf.copy_(cpu_pinned, non_blocking=True)
    end_event.record(stream)
    stream.synchronize()
    async_ms = start_event.elapsed_time(end_event)
    async_bw = (size_mb * n_iter / 1024) / (async_ms / 1000)

    print(f"  Pinned sync:   {fmt_bw(pinned_bw)}")
    print(f"  Pageable sync: {fmt_bw(page_bw)}")
    print(f"  Pinned async:  {fmt_bw(async_bw)}")

    del cpu_pinned, cpu_page, gpu_buf
    torch.cuda.empty_cache()

    return pinned_bw, async_bw


# ─── Test 2: NVMe read bandwidth ───


def test_nvme_bandwidth(nvme_path, size_mb=1024, n_iter=3):
    """Measure sequential read from NVMe into pinned CPU memory."""
    print(f"\n{'=' * 70}")
    print(f"  TEST 2: NVMe Sequential Read ({size_mb} MB)")
    print(f"{'=' * 70}")

    if nvme_path is None:
        print("  Skipped (use --nvme /path/to/mount to test)")
        return None

    # Write a temp file
    fpath = os.path.join(nvme_path, f"_stream_bench_{os.getpid()}.tmp")
    nbytes = size_mb * 1024 * 1024

    print(f"  Writing {size_mb} MB test file to {fpath}...")
    data = os.urandom(nbytes)
    with open(fpath, "wb") as f:
        f.write(data)

    # Drop page cache
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
    except (PermissionError, FileNotFoundError):
        print("  Warning: cannot drop page cache (need root). Results may be cached.")

    # Read into pinned memory buffer

    bandwidths = []
    for i in range(n_iter):
        # Drop caches between iterations if possible
        try:
            os.system("sync")
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3")
        except Exception:
            pass

        fd = os.open(fpath, os.O_RDONLY)
        start = time.perf_counter()
        total_read = 0
        block_size = 4 * 1024 * 1024  # 4 MB blocks
        while total_read < nbytes:
            chunk = os.read(fd, min(block_size, nbytes - total_read))
            if not chunk:
                break
            total_read += len(chunk)
        os.close(fd)
        elapsed = time.perf_counter() - start
        bw = (total_read / (1024**3)) / elapsed
        bandwidths.append(bw)
        print(f"  Run {i + 1}: {fmt_bw(bw)}")

    os.unlink(fpath)
    avg_bw = sum(bandwidths) / len(bandwidths)
    print(f"  Average: {fmt_bw(avg_bw)}")
    return avg_bw


# ─── Test 3: Matmul throughput ───


def test_matmul_throughput(hidden=8192, intermediate=28672, batch_tokens_list=None):
    """Measure actual matmul time for typical transformer layer shapes."""
    print(f"\n{'=' * 70}")
    print(f"  TEST 3: Matmul Throughput (hidden={hidden}, inter={intermediate})")
    print(f"{'=' * 70}")

    if batch_tokens_list is None:
        batch_tokens_list = [256, 512, 1024, 2048, 4096, 8192, 16384]

    results = {}
    print(f"  {'Tokens':>7s}  {'Time':>8s}  {'TFLOPS':>8s}  {'note'}")
    print(f"  {'─' * 7}  {'─' * 8}  {'─' * 8}  {'─' * 20}")

    for M in batch_tokens_list:
        # Simulate a transformer layer: QKV + O + gate + up + down
        # QKV: [M, h] x [h, 3h] (fused), O: [M, h] x [h, h]
        # Gate+Up: [M, h] x [h, 2*inter] (fused), Down: [M, inter] x [inter, h]
        K, N2 = hidden, intermediate

        A1 = torch.randn(M, K, dtype=torch.float16, device="cuda")
        W_qkvo = torch.randn(K, 4 * K, dtype=torch.float16, device="cuda")  # QKV+O fused
        W_gate_up = torch.randn(K, 2 * N2, dtype=torch.float16, device="cuda")
        W_down = torch.randn(N2, K, dtype=torch.float16, device="cuda")

        # Warmup
        for _ in range(3):
            o1 = torch.mm(A1, W_qkvo)
            o2 = torch.mm(A1, W_gate_up)
            mid = o2[:, :N2]  # take gate output
            o3 = torch.mm(mid, W_down)
        sync()

        n_iter = max(3, min(20, 5000 // M))

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(n_iter):
            o1 = torch.mm(A1, W_qkvo)
            o2 = torch.mm(A1, W_gate_up)
            mid = o2[:, :N2]
            o3 = torch.mm(mid, W_down)
        end_event.record()
        sync()

        ms = start_event.elapsed_time(end_event) / n_iter

        # FLOPs: 2*M*K*N per matmul
        flops = 2 * M * K * 4 * K + 2 * M * K * 2 * N2 + 2 * M * N2 * K
        tflops = flops / (ms / 1000) / 1e12

        results[M] = ms
        note = "← forward only, 1 layer"
        print(f"  {M:>7d}  {fmt_time(ms):>8s}  {tflops:>7.1f}T  {note}")

        del A1, W_qkvo, W_gate_up, W_down, o1, o2, o3, mid
        torch.cuda.empty_cache()

    return results


# ─── Test 4: Overlap test ───


def test_overlap(layer_mb=470, batch_tokens=4096, hidden=8192, intermediate=28672):
    """Test if compute on the default stream overlaps with H2D on a copy stream."""
    print(f"\n{'=' * 70}")
    print("  TEST 4: Compute + Transfer Overlap")
    print(f"  (layer={layer_mb} MB, tokens={batch_tokens})")
    print(f"{'=' * 70}")

    n_elem = (layer_mb * 1024 * 1024) // 2  # float16
    K, N2 = hidden, intermediate

    # Allocate buffers
    cpu_pinned = torch.empty(n_elem, dtype=torch.float16, pin_memory=True)
    cpu_pinned.fill_(1.0)
    gpu_recv = torch.empty(n_elem, dtype=torch.float16, device="cuda")

    # Compute buffers
    A = torch.randn(batch_tokens, K, dtype=torch.float16, device="cuda")
    W1 = torch.randn(K, 4 * K, dtype=torch.float16, device="cuda")
    W2 = torch.randn(K, 2 * N2, dtype=torch.float16, device="cuda")
    W3 = torch.randn(N2, K, dtype=torch.float16, device="cuda")

    copy_stream = torch.cuda.Stream()

    # ─ Measure transfer alone ─
    sync()
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)

    t_start.record(copy_stream)
    with torch.cuda.stream(copy_stream):
        gpu_recv.copy_(cpu_pinned, non_blocking=True)
    t_end.record(copy_stream)
    copy_stream.synchronize()
    transfer_ms = t_start.elapsed_time(t_end)

    # ─ Measure compute alone ─
    sync()
    c_start = torch.cuda.Event(enable_timing=True)
    c_end = torch.cuda.Event(enable_timing=True)

    c_start.record()
    o1 = torch.mm(A, W1)
    o2 = torch.mm(A, W2)
    o3 = torch.mm(o2[:, :N2], W3)
    c_end.record()
    sync()
    compute_ms = c_start.elapsed_time(c_end)

    # ─ Measure overlapped ─
    sync()
    both_start = torch.cuda.Event(enable_timing=True)
    both_end_copy = torch.cuda.Event(enable_timing=True)
    both_end_compute = torch.cuda.Event(enable_timing=True)
    both_end = torch.cuda.Event(enable_timing=True)

    both_start.record()

    # Launch copy on copy_stream
    with torch.cuda.stream(copy_stream):
        gpu_recv.copy_(cpu_pinned, non_blocking=True)
    both_end_copy.record(copy_stream)

    # Launch compute on default stream (no dependency on copy)
    o1 = torch.mm(A, W1)
    o2 = torch.mm(A, W2)
    o3 = torch.mm(o2[:, :N2], W3)
    both_end_compute.record()

    # Wait for both
    both_end.record()
    sync()
    copy_stream.synchronize()

    overlap_total = both_start.elapsed_time(both_end)
    overlap_copy = both_start.elapsed_time(both_end_copy)
    overlap_compute = both_start.elapsed_time(both_end_compute)

    sequential = transfer_ms + compute_ms
    speedup = sequential / overlap_total if overlap_total > 0 else 0
    hidden_pct = max(0, (sequential - overlap_total) / sequential * 100)

    print(f"  Transfer alone:   {fmt_time(transfer_ms)}")
    print(f"  Compute alone:    {fmt_time(compute_ms)}")
    print(f"  Sequential:       {fmt_time(sequential)}")
    print(f"  Overlapped total: {fmt_time(overlap_total)}")
    print(f"    copy finished:  {fmt_time(overlap_copy)}")
    print(f"    compute done:   {fmt_time(overlap_compute)}")
    print(f"  Overlap speedup:  {speedup:.2f}x")
    print(f"  Transfer hidden:  {hidden_pct:.0f}%")

    if speedup > 1.5:
        print("  → Good overlap! Transfer mostly hidden behind compute.")
    elif speedup > 1.1:
        print("  → Partial overlap. Some transfer hidden.")
    else:
        print("  → Little/no overlap. Transfer and compute may share PCIe/memory.")

    del cpu_pinned, gpu_recv, A, W1, W2, W3, o1, o2, o3
    torch.cuda.empty_cache()

    return transfer_ms, compute_ms, overlap_total


# ─── Test 5: Full pipeline simulation ───


def test_pipeline(n_layers=20, layer_mb=470, batch_tokens=4096, hidden=8192, intermediate=28672):
    """
    Simulate double-buffered layer streaming:
      - 2 GPU weight slots (A, B)
      - While computing on slot A, transfer next layer into slot B
      - Swap and repeat
    Compare to: all layers resident in GPU (no streaming).
    """
    print(f"\n{'=' * 70}")
    print(f"  TEST 5: Full Pipeline — {n_layers} layers, {layer_mb} MB each")
    print(f"  tokens={batch_tokens}, double-buffered streaming")
    print(f"{'=' * 70}")

    K, N2 = hidden, intermediate
    n_elem_layer = (layer_mb * 1024 * 1024) // 2

    # Allocate: 2 GPU weight slots, N CPU pinned layers
    print(f"  Allocating {n_layers} pinned CPU layers ({n_layers * layer_mb / 1024:.1f} GB)...")
    cpu_layers = []
    for i in range(n_layers):
        buf = torch.empty(n_elem_layer, dtype=torch.float16, pin_memory=True)
        buf.fill_(float(i % 10))
        cpu_layers.append(buf)

    gpu_slot = [
        torch.empty(n_elem_layer, dtype=torch.float16, device="cuda"),
        torch.empty(n_elem_layer, dtype=torch.float16, device="cuda"),
    ]

    # Activation buffer
    A = torch.randn(batch_tokens, K, dtype=torch.float16, device="cuda")

    # We'll reshape the flat weight buffer into matmul-friendly shapes for compute.
    # For simplicity, just do matmuls with the right dimensions using separate weight
    # tensors (the transfer uses the flat buffer, compute uses reshaped views/copies).
    W1 = torch.randn(K, 4 * K, dtype=torch.float16, device="cuda")
    W2 = torch.randn(K, 2 * N2, dtype=torch.float16, device="cuda")
    W3 = torch.randn(N2, K, dtype=torch.float16, device="cuda")

    copy_stream = torch.cuda.Stream()

    # Pre-allocate ALL output buffers to avoid alloc during pipeline
    O1 = torch.empty(batch_tokens, 4 * K, dtype=torch.float16, device="cuda")
    O2 = torch.empty(batch_tokens, 2 * N2, dtype=torch.float16, device="cuda")
    O3 = torch.empty(batch_tokens, K, dtype=torch.float16, device="cuda")

    def do_compute(_A=A, _W1=W1, _W2=W2, _W3=W3, _O1=O1, _O2=O2, _O3=O3, _N2=N2):
        """Simulate one layer's forward pass compute (zero-alloc)."""
        torch.mm(_A, _W1, out=_O1)
        torch.mm(_A, _W2, out=_O2)
        torch.mm(_O2[:, :_N2], _W3, out=_O3)

    # ─ Baseline: compute only (no transfer) ─
    sync()
    for _ in range(5):
        do_compute()
    sync()

    base_start = torch.cuda.Event(enable_timing=True)
    base_end = torch.cuda.Event(enable_timing=True)
    base_start.record()
    for _ in range(n_layers):
        do_compute()
    base_end.record()
    sync()
    baseline_ms = base_start.elapsed_time(base_end)

    # ─ Transfer only: sequential H2D of all layers ─
    sync()
    xfer_start = torch.cuda.Event(enable_timing=True)
    xfer_end = torch.cuda.Event(enable_timing=True)
    xfer_start.record(copy_stream)
    for i in range(n_layers):
        with torch.cuda.stream(copy_stream):
            gpu_slot[0].copy_(cpu_layers[i], non_blocking=True)
    xfer_end.record(copy_stream)
    copy_stream.synchronize()
    xfer_only_ms = xfer_start.elapsed_time(xfer_end)

    # ─ Double-buffered pipeline ─
    # Pre-load layer 0 into slot 0
    gpu_slot[0].copy_(cpu_layers[0], non_blocking=False)
    sync()

    # Pre-create events to avoid alloc in loop
    pipe_start = torch.cuda.Event(enable_timing=True)
    pipe_end = torch.cuda.Event(enable_timing=True)

    pipe_start.record()

    for i in range(n_layers):
        cur_slot = i % 2
        next_slot = 1 - cur_slot

        # Start async transfer of next layer (if any) on copy stream
        if i + 1 < n_layers:
            with torch.cuda.stream(copy_stream):
                gpu_slot[next_slot].copy_(cpu_layers[i + 1], non_blocking=True)

        # Compute on current layer (default stream, zero-alloc)
        do_compute()

        # Default stream waits for copy to finish before next iteration
        if i + 1 < n_layers:
            torch.cuda.current_stream().wait_stream(copy_stream)

    pipe_end.record()
    sync()

    pipeline_ms = pipe_start.elapsed_time(pipe_end)
    avg_layer = pipeline_ms / n_layers

    sequential_ms = baseline_ms + xfer_only_ms
    speedup = sequential_ms / pipeline_ms if pipeline_ms > 0 else 0
    overhead_pct = (pipeline_ms / baseline_ms - 1) * 100

    print("\n  Results:")
    print(
        f"  {'Compute only (no transfer):':40s} {fmt_time(baseline_ms):>10s}  ({fmt_time(baseline_ms / n_layers)}/layer)"
    )
    print(
        f"  {'Transfer only (no compute):':40s} {fmt_time(xfer_only_ms):>10s}  ({fmt_time(xfer_only_ms / n_layers)}/layer)"
    )
    print(f"  {'Sequential (compute + transfer):':40s} {fmt_time(sequential_ms):>10s}")
    print(f"  {'Double-buffered pipeline:':40s} {fmt_time(pipeline_ms):>10s}  ({fmt_time(avg_layer)}/layer)")
    print(f"  {'':40s}")
    print(f"  {'Pipeline vs sequential:':40s} {speedup:.2f}x faster")
    print(f"  {'Pipeline overhead vs compute-only:':40s} {overhead_pct:+.1f}%")

    if overhead_pct < 5:
        print("\n  → EXCELLENT: Transfer fully hidden. Streaming adds <5% overhead.")
    elif overhead_pct < 20:
        print(f"\n  → GOOD: Most transfer hidden. Streaming adds {overhead_pct:.0f}% overhead.")
    elif overhead_pct < 50:
        print(f"\n  → MODERATE: Partial overlap. {overhead_pct:.0f}% overhead from streaming.")
    else:
        print(f"\n  → POOR: Transfer dominates. {overhead_pct:.0f}% overhead — compute too fast.")

    # Memory summary
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
    total_weights = n_layers * layer_mb / 1024
    print("\n  Memory:")
    print(f"  {'Total weight data:':40s} {total_weights:.1f} GB")
    print(f"  {'GPU weight slots (2 layers):':40s} {2 * layer_mb / 1024:.2f} GB")
    print(f"  {'GPU peak memory:':40s} {gpu_mem:.2f} GB")
    print(f"  {'VRAM savings:':40s} {(1 - 2 * layer_mb / 1024 / total_weights) * 100:.0f}%")

    del cpu_layers, gpu_slot, A, W1, W2, W3
    torch.cuda.empty_cache()

    return baseline_ms, xfer_only_ms, pipeline_ms


# ─── Test 6: NVMe → CPU → GPU pipeline ───


def test_nvme_pipeline(
    nvme_path,
    n_layers=20,
    layer_mb=1237,
    batch_tokens=4096,
    hidden=5120,
    intermediate=12288,
    expert_intermediate=1536,
    n_active_experts=8,
):
    """
    End-to-end NVMe→CPU→GPU pipeline benchmark using safetensors.

    Creates a synthetic safetensors file on NVMe with layer-ordered tensors,
    then benchmarks the full three-stage pipeline:
      Stage 1: mmap'd read → CPU pinned staging buffer (triggers NVMe page faults)
      Stage 2: Async copy from pinned → GPU double-buffer slot (copy stream)
      Stage 3: Matmul compute on default stream (simulating layer forward)

    This is the GATING STEP for the NVMe weight streaming project.
    If overhead is unacceptable at viable batch sizes, the project fails.
    """
    print(f"\n{'=' * 70}")
    print(f"  TEST 6: NVMe→CPU→GPU Pipeline — {n_layers} layers, {layer_mb} MB each")
    print(f"  tokens={batch_tokens}, safetensors mmap, double-buffered")
    print(f"{'=' * 70}")

    if nvme_path is None:
        print("  Skipped (use --nvme /path/to/mount to test)")
        return None, None, None

    try:
        from safetensors.torch import save_file
        from safetensors import safe_open
    except ImportError:
        print("  Skipped (safetensors not installed: pip install safetensors)")
        return None, None, None

    from collections import OrderedDict

    K = hidden
    n_elem_layer = (layer_mb * 1024 * 1024) // 4  # int32 elements

    # ─ Create synthetic safetensors file ─
    fpath = os.path.join(nvme_path, f"_stream_bench_sf_{os.getpid()}.safetensors")
    print(f"  Creating {n_layers * layer_mb / 1024:.1f} GB safetensors file...")
    print(f"  Path: {fpath}")

    tensors = OrderedDict()
    for i in range(n_layers):
        # One flat tensor per layer (simulating concatenated packed weights)
        tensors[f"layer.{i}.packed"] = torch.randint(
            0, 2**31, (n_elem_layer,), dtype=torch.int32
        )
    save_file(tensors, fpath)
    del tensors

    file_size_gb = os.path.getsize(fpath) / 1024**3
    print(f"  File size: {file_size_gb:.2f} GB")

    # Drop page cache so we measure cold NVMe reads
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        print("  Page cache dropped.")
    except (PermissionError, FileNotFoundError):
        print("  Warning: cannot drop page cache (need root). First run may use cached data.")

    # ─ Open via mmap ─
    sf = safe_open(fpath, framework="pt", device="cpu")

    # Pre-allocate: pinned staging buffer + GPU double-buffer
    pinned_buf = torch.empty(n_elem_layer, dtype=torch.int32, pin_memory=True)
    gpu_slot = [
        torch.empty(n_elem_layer, dtype=torch.int32, device="cuda"),
        torch.empty(n_elem_layer, dtype=torch.int32, device="cuda"),
    ]
    copy_stream = torch.cuda.Stream()

    # Compute buffers (simulate MoE layer: attention + shared expert + active experts)
    # Attention: QKV+O → [M, 4*K] matmul
    # Shared expert: gate+up → [M, 2*inter], down → [M, K]
    # Active experts: n_active × (gate+up+down) with small intermediate
    N_shared = intermediate
    N_expert = expert_intermediate
    A = torch.randn(batch_tokens, K, dtype=torch.float16, device="cuda")
    W_attn = torch.randn(K, 4 * K, dtype=torch.float16, device="cuda")
    W_shared_gu = torch.randn(K, 2 * N_shared, dtype=torch.float16, device="cuda")
    W_shared_d = torch.randn(N_shared, K, dtype=torch.float16, device="cuda")
    # Expert weights (per active expert)
    W_expert_gu = torch.randn(K, 2 * N_expert, dtype=torch.float16, device="cuda")
    W_expert_d = torch.randn(N_expert, K, dtype=torch.float16, device="cuda")

    # Pre-alloc outputs
    O_attn = torch.empty(batch_tokens, 4 * K, dtype=torch.float16, device="cuda")
    O_shared_gu = torch.empty(batch_tokens, 2 * N_shared, dtype=torch.float16, device="cuda")
    O_shared_d = torch.empty(batch_tokens, K, dtype=torch.float16, device="cuda")
    O_expert_gu = torch.empty(batch_tokens, 2 * N_expert, dtype=torch.float16, device="cuda")
    O_expert_d = torch.empty(batch_tokens, K, dtype=torch.float16, device="cuda")

    def do_moe_compute():
        """Simulate one MoE layer forward (zero-alloc)."""
        # Attention
        torch.mm(A, W_attn, out=O_attn)
        # Shared expert
        torch.mm(A, W_shared_gu, out=O_shared_gu)
        torch.mm(O_shared_gu[:, :N_shared], W_shared_d, out=O_shared_d)
        # Active experts (simulate n_active_experts sequential expert forwards)
        for _ in range(n_active_experts):
            torch.mm(A, W_expert_gu, out=O_expert_gu)
            torch.mm(O_expert_gu[:, :N_expert], W_expert_d, out=O_expert_d)

    # ─ Warmup ─
    sync()
    for _ in range(3):
        do_moe_compute()
    sync()

    # ─ Baseline: compute only ─
    base_start = torch.cuda.Event(enable_timing=True)
    base_end = torch.cuda.Event(enable_timing=True)
    base_start.record()
    for _ in range(n_layers):
        do_moe_compute()
    base_end.record()
    sync()
    baseline_ms = base_start.elapsed_time(base_end)

    # ─ Transfer only: mmap → pinned → GPU for all layers ─
    # Re-drop cache
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
    except Exception:
        pass

    sync()
    xfer_wall_start = time.perf_counter()
    for i in range(n_layers):
        # Stage 1: mmap → pinned (CPU work, triggers NVMe page faults)
        tensor = sf.get_tensor(f"layer.{i}.packed")
        pinned_buf[:tensor.numel()].copy_(tensor)
        # Stage 2: pinned → GPU (sync for measurement)
        gpu_slot[0][:tensor.numel()].copy_(pinned_buf[:tensor.numel()])
    sync()
    xfer_wall_end = time.perf_counter()
    xfer_only_ms = (xfer_wall_end - xfer_wall_start) * 1000

    # ─ Full pipeline: mmap → pinned → GPU with threading + double-buffering ─
    # The mmap→pinned copy is CPU work. We run it on a background thread so
    # it overlaps with GPU compute. Two pinned buffers avoid contention.
    import threading

    # Re-drop cache
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
    except Exception:
        pass

    # Two pinned staging buffers for the background loader
    pinned_bufs = [
        pinned_buf,
        torch.empty(n_elem_layer, dtype=torch.int32, pin_memory=True),
    ]

    # Pre-load layer 0 into pinned_bufs[0] and GPU slot 0
    tensor0 = sf.get_tensor("layer.0.packed")
    n0 = tensor0.numel()
    pinned_bufs[0][:n0].copy_(tensor0)
    gpu_slot[0][:n0].copy_(pinned_bufs[0][:n0])
    sync()

    # Shared state for background loader
    load_ready = [threading.Event() for _ in range(n_layers)]
    load_numel = [0] * n_layers

    def bg_load(layer_idx, pinned_idx):
        """Background: mmap→pinned for one layer."""
        t = sf.get_tensor(f"layer.{layer_idx}.packed")
        n = t.numel()
        pinned_bufs[pinned_idx][:n].copy_(t)
        load_numel[layer_idx] = n
        load_ready[layer_idx].set()

    pipe_wall_start = time.perf_counter()
    pipe_start = torch.cuda.Event(enable_timing=True)
    pipe_end = torch.cuda.Event(enable_timing=True)
    pipe_start.record()

    # Start background load of layer 1 while we compute layer 0
    bg_thread = None
    if n_layers > 1:
        bg_thread = threading.Thread(target=bg_load, args=(1, 1))
        bg_thread.start()

    for i in range(n_layers):
        cur_slot = i % 2
        next_slot = 1 - cur_slot
        cur_pinned = i % 2

        # GPU compute on current layer (async launch)
        do_moe_compute()

        if i + 1 < n_layers:
            next_pinned = (i + 1) % 2

            # Wait for background mmap→pinned of layer i+1 to complete
            load_ready[i + 1].wait()
            if bg_thread is not None:
                bg_thread.join()

            # Queue async pinned→GPU copy on copy stream
            n_next = load_numel[i + 1]
            with torch.cuda.stream(copy_stream):
                gpu_slot[next_slot][:n_next].copy_(
                    pinned_bufs[next_pinned][:n_next], non_blocking=True
                )

            # Start background load of layer i+2 (if any) into the
            # pinned buffer we're NOT currently copying from
            if i + 2 < n_layers:
                future_pinned = (i + 2) % 2
                bg_thread = threading.Thread(
                    target=bg_load, args=(i + 2, future_pinned)
                )
                bg_thread.start()
            else:
                bg_thread = None

        # Wait for compute + copy before next iteration
        if i + 1 < n_layers:
            torch.cuda.current_stream().wait_stream(copy_stream)

    pipe_end.record()
    sync()
    pipe_wall_end = time.perf_counter()
    pipeline_ms = pipe_start.elapsed_time(pipe_end)
    pipeline_wall_ms = (pipe_wall_end - pipe_wall_start) * 1000

    # ─ Results ─
    sequential_ms = baseline_ms + xfer_only_ms
    overhead_gpu = (pipeline_ms / baseline_ms - 1) * 100
    # Use wall-clock for the pipeline since mmap→pinned is CPU work not
    # captured by CUDA events
    overhead_wall = (pipeline_wall_ms / baseline_ms - 1) * 100

    print("\n  Results:")
    print(
        f"  {'Compute only (GPU events):':40s} {fmt_time(baseline_ms):>10s}"
        f"  ({fmt_time(baseline_ms / n_layers)}/layer)"
    )
    print(
        f"  {'Transfer only (wall clock):':40s} {fmt_time(xfer_only_ms):>10s}"
        f"  ({fmt_time(xfer_only_ms / n_layers)}/layer)"
    )
    print(f"  {'Sequential (compute + transfer):':40s} {fmt_time(sequential_ms):>10s}")
    print(
        f"  {'Pipeline (GPU events):':40s} {fmt_time(pipeline_ms):>10s}"
        f"  ({fmt_time(pipeline_ms / n_layers)}/layer)"
    )
    print(
        f"  {'Pipeline (wall clock):':40s} {fmt_time(pipeline_wall_ms):>10s}"
        f"  ({fmt_time(pipeline_wall_ms / n_layers)}/layer)"
    )
    print()
    print(f"  {'Overhead vs compute (GPU events):':40s} {overhead_gpu:+.1f}%")
    print(f"  {'Overhead vs compute (wall clock):':40s} {overhead_wall:+.1f}%")

    # The wall clock overhead is the more accurate measure because it captures
    # CPU-side blocking on NVMe page faults that CUDA events miss
    effective_overhead = max(overhead_gpu, overhead_wall)

    if effective_overhead < 5:
        print(
            "\n  → EXCELLENT: NVMe→CPU→GPU transfer fully hidden behind compute."
        )
    elif effective_overhead < 20:
        print(
            f"\n  → GOOD: Most NVMe transfer hidden. {effective_overhead:.0f}% overhead."
        )
    elif effective_overhead < 50:
        print(
            f"\n  → MODERATE: Partial overlap. {effective_overhead:.0f}% overhead."
            f" Try increasing batch size."
        )
    else:
        print(
            f"\n  → POOR: NVMe transfer dominates. {effective_overhead:.0f}% overhead."
            f"\n    Increase batch size or use faster NVMe."
        )

    # Memory summary
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
    total_weights = n_layers * layer_mb / 1024
    print("\n  Memory:")
    print(f"  {'Total weight data on disk:':40s} {total_weights:.1f} GB")
    print(f"  {'GPU double-buffer (2 slots):':40s} {2 * layer_mb / 1024:.2f} GB")
    print(f"  {'CPU pinned staging:':40s} {layer_mb / 1024:.2f} GB")
    print(f"  {'GPU peak memory:':40s} {gpu_mem:.2f} GB")

    # Cleanup
    del sf, pinned_buf, gpu_slot, A, W_attn
    del W_shared_gu, W_shared_d, W_expert_gu, W_expert_d
    del O_attn, O_shared_gu, O_shared_d, O_expert_gu, O_expert_d
    torch.cuda.empty_cache()
    os.unlink(fpath)

    return baseline_ms, xfer_only_ms, pipeline_wall_ms


# ─── Main ───


def main():
    parser = argparse.ArgumentParser(description="CPU→GPU Weight Streaming Benchmark")
    parser.add_argument("--layer-mb", type=int, default=470, help="Layer size in MB (default: 470 for Llama-70B)")
    parser.add_argument("--n-layers", type=int, default=20, help="Number of layers for pipeline test (default: 20)")
    parser.add_argument("--hidden", type=int, default=8192, help="Hidden dimension (default: 8192)")
    parser.add_argument("--intermediate", type=int, default=28672, help="Intermediate dimension (default: 28672)")
    parser.add_argument("--batch-tokens", type=int, nargs="+", default=None, help="Batch token counts for matmul test")
    parser.add_argument("--pipeline-tokens", type=int, default=4096, help="Tokens for pipeline test (default: 4096)")
    parser.add_argument("--nvme", type=str, default=None, help="NVMe mount path for disk read test")
    parser.add_argument("--skip-matmul", action="store_true", help="Skip detailed matmul sweep")
    parser.add_argument(
        "--moe-experts", type=int, default=8,
        help="Number of active experts for MoE compute simulation (default: 8)",
    )
    parser.add_argument(
        "--expert-intermediate", type=int, default=1536,
        help="Expert MLP intermediate dim (default: 1536 for GLM-4.7)",
    )
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Test 1: PCIe bandwidth
    pinned_bw, async_bw = test_pcie_bandwidth(size_mb=max(512, args.layer_mb))

    # Test 2: NVMe bandwidth
    nvme_bw = test_nvme_bandwidth(args.nvme, size_mb=max(512, args.layer_mb))

    # Test 3: Matmul throughput
    if not args.skip_matmul:
        tokens_list = args.batch_tokens or [512, 1024, 2048, 4096, 8192, 16384]
        test_matmul_throughput(hidden=args.hidden, intermediate=args.intermediate, batch_tokens_list=tokens_list)

    # Test 4: Overlap
    transfer_ms, compute_ms, overlap_ms = test_overlap(
        layer_mb=args.layer_mb, batch_tokens=args.pipeline_tokens, hidden=args.hidden, intermediate=args.intermediate
    )

    # Test 5: Full pipeline
    baseline_ms, _xfer_ms, pipeline_ms = test_pipeline(
        n_layers=args.n_layers,
        layer_mb=args.layer_mb,
        batch_tokens=args.pipeline_tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
    )

    # Test 6: NVMe→CPU→GPU pipeline
    if args.nvme:
        test_nvme_pipeline(
            nvme_path=args.nvme,
            n_layers=args.n_layers,
            layer_mb=args.layer_mb,
            batch_tokens=args.pipeline_tokens,
            hidden=args.hidden,
            intermediate=args.intermediate,
            expert_intermediate=args.expert_intermediate,
            n_active_experts=args.moe_experts,
        )

    # ─ Summary ─
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  PCIe H2D (pinned):  {fmt_bw(pinned_bw)}")
    print(f"  PCIe H2D (async):   {fmt_bw(async_bw)}")
    if nvme_bw:
        print(f"  NVMe read:          {fmt_bw(nvme_bw)}")
    print(f"  Layer size:         {args.layer_mb} MB")
    print(f"  Transfer/layer:     {fmt_time(transfer_ms)} (measured)")
    print(f"  Compute/layer:      {fmt_time(compute_ms)} @ {args.pipeline_tokens} tokens")
    print(
        f"  Overlap ratio:      {(transfer_ms + compute_ms) / overlap_ms:.2f}x" if overlap_ms > 0 else "  Overlap: N/A"
    )

    ratio = compute_ms / transfer_ms if transfer_ms > 0 else float("inf")
    print(f"  Compute/transfer:   {ratio:.2f}x", end="")
    if ratio >= 1.0:
        print("  ← compute dominates, streaming should work well")
    else:
        print(f"  ← transfer dominates, need batch≥{int(args.pipeline_tokens / ratio)} tokens")

    overhead = (pipeline_ms / baseline_ms - 1) * 100
    print(f"  Pipeline overhead:  {overhead:+.1f}%")
    verdict = "VIABLE" if overhead < 20 else "MARGINAL" if overhead < 50 else "NOT VIABLE"
    print(f"\n  Verdict: {verdict} at {args.pipeline_tokens} tokens/step")


if __name__ == "__main__":
    main()
