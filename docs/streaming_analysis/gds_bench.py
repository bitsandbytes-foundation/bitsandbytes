"""GPUDirect Storage (GDS) benchmark: NVMe → GPU direct transfer.

Tests whether cuFile/kvikio can do NVMe-to-GPU DMA bypassing CPU,
and measures bandwidth compared to the traditional mmap→pinned→GPU path.

Tests:
  1. kvikio GDS status check (is P2P DMA active?)
  2. NVMe → GPU direct read bandwidth via kvikio
  3. NVMe → CPU pinned → GPU (traditional path) for comparison
  4. Pipelined layer streaming: GDS vs traditional
"""

import os
import time
import tempfile
from collections import OrderedDict

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


# ─── Test 1: GDS status ───

def test_gds_status():
    print(f"\n{'=' * 70}")
    print("  TEST 1: GPUDirect Storage Status")
    print(f"{'=' * 70}")

    import kvikio
    import kvikio.defaults

    print(f"  kvikio version:     {kvikio.__version__}")

    # Check if GDS is available
    gds_avail = kvikio.is_remote_file_available() if hasattr(kvikio, 'is_remote_file_available') else "N/A"
    print(f"  Remote file avail:  {gds_avail}")

    # Check compat mode vs GDS mode
    try:
        compat = kvikio.defaults.compat_mode()
        print(f"  Compat mode:        {compat}")
        if compat:
            print("  → Running in COMPATIBILITY mode (POSIX fallback, no P2P DMA)")
        else:
            print("  → Running in GDS mode (direct NVMe→GPU P2P DMA)")
    except Exception as e:
        print(f"  Compat mode check:  error — {e}")

    # Thread pool
    try:
        nthreads = kvikio.defaults.num_threads()
        print(f"  IO thread pool:     {nthreads} threads")
    except Exception:
        pass

    # Task size
    try:
        ts = kvikio.defaults.task_size()
        print(f"  Task size:          {ts / (1024*1024):.0f} MB")
    except Exception:
        pass

    return True


# ─── Test 2: GDS NVMe → GPU bandwidth ───

def test_gds_bandwidth(test_dir, size_mb=512, n_iter=5):
    print(f"\n{'=' * 70}")
    print(f"  TEST 2: GDS NVMe → GPU Direct Read ({size_mb} MB)")
    print(f"{'=' * 70}")

    import kvikio

    nbytes = size_mb * 1024 * 1024
    fpath = os.path.join(test_dir, f"_gds_bench_{os.getpid()}.bin")

    # Create test file
    print(f"  Writing {size_mb} MB test file to {fpath}...")
    cpu_data = torch.randint(0, 2**31, (nbytes // 4,), dtype=torch.int32)
    with open(fpath, "wb") as f:
        f.write(cpu_data.numpy().tobytes())
    del cpu_data

    # Allocate GPU buffer
    gpu_buf = torch.empty(nbytes // 4, dtype=torch.int32, device="cuda")

    # Warmup
    print("  Warming up...")
    with kvikio.CuFile(fpath, "r") as f:
        f.read(gpu_buf)
    sync()

    # Timed reads
    bandwidths = []
    for i in range(n_iter):
        sync()
        # Drop page cache if possible
        try:
            os.system("sync")
            with open("/proc/sys/vm/drop_caches", "w") as fc:
                fc.write("3")
        except Exception:
            pass

        gpu_buf.zero_()
        sync()

        start = time.perf_counter()
        with kvikio.CuFile(fpath, "r") as f:
            nbytes_read = f.read(gpu_buf)
        sync()
        elapsed = time.perf_counter() - start

        bw = (nbytes_read / (1024**3)) / elapsed
        bandwidths.append(bw)
        print(f"  Run {i+1}: {fmt_bw(bw)}  ({nbytes_read / 1e6:.0f} MB in {elapsed*1000:.1f}ms)")

    avg = sum(bandwidths) / len(bandwidths)
    peak = max(bandwidths)
    print(f"  Average: {fmt_bw(avg)}")
    print(f"  Peak:    {fmt_bw(peak)}")

    # Verify data integrity
    print("  Verifying data integrity...")
    cpu_check = torch.from_file(fpath, dtype=torch.int32, size=nbytes // 4)
    gpu_check = gpu_buf.cpu()
    if torch.equal(cpu_check, gpu_check):
        print("  → Data integrity OK")
    else:
        mismatches = (cpu_check != gpu_check).sum().item()
        print(f"  → DATA MISMATCH: {mismatches} elements differ!")

    del gpu_buf
    os.unlink(fpath)
    torch.cuda.empty_cache()

    return avg, peak


# ─── Test 3: Traditional mmap → pinned → GPU for comparison ───

def test_traditional_bandwidth(test_dir, size_mb=512, n_iter=5):
    print(f"\n{'=' * 70}")
    print(f"  TEST 3: Traditional NVMe → CPU → GPU ({size_mb} MB)")
    print(f"{'=' * 70}")

    nbytes = size_mb * 4  # int32
    n_elem = size_mb * 1024 * 1024 // 4
    fpath = os.path.join(test_dir, f"_trad_bench_{os.getpid()}.bin")

    # Create test file
    print(f"  Writing {size_mb} MB test file...")
    cpu_data = torch.randint(0, 2**31, (n_elem,), dtype=torch.int32)
    with open(fpath, "wb") as f:
        f.write(cpu_data.numpy().tobytes())

    pinned_buf = torch.empty(n_elem, dtype=torch.int32, pin_memory=True)
    gpu_buf = torch.empty(n_elem, dtype=torch.int32, device="cuda")

    # Warmup
    pinned_buf.copy_(cpu_data)
    gpu_buf.copy_(pinned_buf)
    sync()
    del cpu_data

    bandwidths = []
    for i in range(n_iter):
        try:
            os.system("sync")
            with open("/proc/sys/vm/drop_caches", "w") as fc:
                fc.write("3")
        except Exception:
            pass

        total_bytes = size_mb * 1024 * 1024
        start = time.perf_counter()
        # Stage 1: NVMe → CPU (mmap read)
        data = torch.from_file(fpath, dtype=torch.int32, size=n_elem)
        pinned_buf.copy_(data)
        # Stage 2: CPU pinned → GPU
        gpu_buf.copy_(pinned_buf)
        sync()
        elapsed = time.perf_counter() - start

        bw = (total_bytes / (1024**3)) / elapsed
        bandwidths.append(bw)
        print(f"  Run {i+1}: {fmt_bw(bw)}  ({elapsed*1000:.1f}ms)")

    avg = sum(bandwidths) / len(bandwidths)
    print(f"  Average: {fmt_bw(avg)}")

    del pinned_buf, gpu_buf
    os.unlink(fpath)
    torch.cuda.empty_cache()

    return avg


# ─── Test 4: Pipelined layer streaming comparison ───

def test_pipeline_comparison(test_dir, n_layers=5, layer_mb=200, batch_tokens=4096,
                              hidden=5120, intermediate=12288,
                              expert_intermediate=1536, n_active_experts=8):
    print(f"\n{'=' * 70}")
    print(f"  TEST 4: Pipelined Streaming — {n_layers} layers × {layer_mb} MB")
    print(f"  GDS (NVMe→GPU direct) vs Traditional (NVMe→CPU→GPU)")
    print(f"{'=' * 70}")

    import kvikio

    n_elem = (layer_mb * 1024 * 1024) // 4  # int32 elements
    nbytes_layer = n_elem * 4

    # Create test file with N layers
    fpath = os.path.join(test_dir, f"_pipe_bench_{os.getpid()}.bin")
    total_mb = n_layers * layer_mb
    print(f"  Creating {total_mb / 1024:.1f} GB test file ({n_layers} layers)...")

    with open(fpath, "wb") as f:
        for i in range(n_layers):
            data = torch.randint(0, 2**31, (n_elem,), dtype=torch.int32)
            f.write(data.numpy().tobytes())
    file_size = os.path.getsize(fpath)
    print(f"  File: {file_size / 1e9:.2f} GB")

    # Compute simulation (same MoE workload as stream_bench.py Test 6)
    K = hidden
    N_shared = intermediate
    N_expert = expert_intermediate
    A = torch.randn(batch_tokens, K, dtype=torch.float16, device="cuda")
    W_attn = torch.randn(K, 4 * K, dtype=torch.float16, device="cuda")
    W_shared_gu = torch.randn(K, 2 * N_shared, dtype=torch.float16, device="cuda")
    W_shared_d = torch.randn(N_shared, K, dtype=torch.float16, device="cuda")
    W_expert_gu = torch.randn(K, 2 * N_expert, dtype=torch.float16, device="cuda")
    W_expert_d = torch.randn(N_expert, K, dtype=torch.float16, device="cuda")
    O_attn = torch.empty(batch_tokens, 4 * K, dtype=torch.float16, device="cuda")
    O_shared_gu = torch.empty(batch_tokens, 2 * N_shared, dtype=torch.float16, device="cuda")
    O_shared_d = torch.empty(batch_tokens, K, dtype=torch.float16, device="cuda")
    O_expert_gu = torch.empty(batch_tokens, 2 * N_expert, dtype=torch.float16, device="cuda")
    O_expert_d = torch.empty(batch_tokens, K, dtype=torch.float16, device="cuda")

    def do_moe_compute():
        torch.mm(A, W_attn, out=O_attn)
        torch.mm(A, W_shared_gu, out=O_shared_gu)
        torch.mm(O_shared_gu[:, :N_shared], W_shared_d, out=O_shared_d)
        for _ in range(n_active_experts):
            torch.mm(A, W_expert_gu, out=O_expert_gu)
            torch.mm(O_expert_gu[:, :N_expert], W_expert_d, out=O_expert_d)

    # Warmup compute
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

    # ─ GDS pipeline: read directly to GPU + compute ─
    gpu_slots = [
        torch.empty(n_elem, dtype=torch.int32, device="cuda"),
        torch.empty(n_elem, dtype=torch.int32, device="cuda"),
    ]

    # Drop page cache
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as fc:
            fc.write("3")
    except Exception:
        pass

    # Pre-load layer 0 via GDS
    with kvikio.CuFile(fpath, "r") as f:
        f.pread(gpu_slots[0], size=nbytes_layer, file_offset=0).get()
    sync()

    gds_start = time.perf_counter()
    gds_gpu_start = torch.cuda.Event(enable_timing=True)
    gds_gpu_end = torch.cuda.Event(enable_timing=True)
    gds_gpu_start.record()

    for i in range(n_layers):
        cur_slot = i % 2
        next_slot = 1 - cur_slot

        # Kick off GDS read for next layer (async on kvikio thread pool)
        future = None
        if i + 1 < n_layers:
            f_handle = kvikio.CuFile(fpath, "r")
            future = f_handle.pread(
                gpu_slots[next_slot],
                size=nbytes_layer,
                file_offset=(i + 1) * nbytes_layer,
            )

        # Compute on current layer
        do_moe_compute()

        # Wait for GDS read to finish
        if future is not None:
            future.get()
            f_handle.close()

    gds_gpu_end.record()
    sync()
    gds_wall_ms = (time.perf_counter() - gds_start) * 1000
    gds_gpu_ms = gds_gpu_start.elapsed_time(gds_gpu_end)

    # ─ Traditional pipeline: mmap → pinned → GPU + compute ─
    import threading

    pinned_bufs = [
        torch.empty(n_elem, dtype=torch.int32, pin_memory=True),
        torch.empty(n_elem, dtype=torch.int32, pin_memory=True),
    ]
    trad_gpu_slots = [
        torch.empty(n_elem, dtype=torch.int32, device="cuda"),
        torch.empty(n_elem, dtype=torch.int32, device="cuda"),
    ]
    copy_stream = torch.cuda.Stream()

    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as fc:
            fc.write("3")
    except Exception:
        pass

    import mmap as mmap_mod

    fd = os.open(fpath, os.O_RDONLY)
    mm = mmap_mod.mmap(fd, 0, access=mmap_mod.ACCESS_READ)

    load_ready = [threading.Event() for _ in range(n_layers)]

    def bg_load_mmap(layer_idx, pinned_idx):
        offset = layer_idx * nbytes_layer
        src = torch.frombuffer(mm[offset:offset + nbytes_layer], dtype=torch.int32).clone()
        pinned_bufs[pinned_idx][:n_elem].copy_(src)
        del src
        load_ready[layer_idx].set()

    # Pre-load layer 0
    bg_load_mmap(0, 0)
    load_ready[0].wait()
    trad_gpu_slots[0].copy_(pinned_bufs[0])
    sync()

    trad_start = time.perf_counter()
    trad_gpu_start = torch.cuda.Event(enable_timing=True)
    trad_gpu_end = torch.cuda.Event(enable_timing=True)
    trad_gpu_start.record()

    bg_thread = None
    if n_layers > 1:
        bg_thread = threading.Thread(target=bg_load_mmap, args=(1, 1))
        bg_thread.start()

    for i in range(n_layers):
        cur_slot = i % 2
        next_slot = 1 - cur_slot

        # Compute
        do_moe_compute()

        if i + 1 < n_layers:
            next_pinned = (i + 1) % 2
            load_ready[i + 1].wait()
            if bg_thread is not None:
                bg_thread.join()

            with torch.cuda.stream(copy_stream):
                trad_gpu_slots[next_slot].copy_(pinned_bufs[next_pinned], non_blocking=True)

            if i + 2 < n_layers:
                future_pinned = (i + 2) % 2
                bg_thread = threading.Thread(target=bg_load_mmap, args=(i + 2, future_pinned))
                bg_thread.start()
            else:
                bg_thread = None

        if i + 1 < n_layers:
            torch.cuda.current_stream().wait_stream(copy_stream)

    trad_gpu_end.record()
    sync()
    trad_wall_ms = (time.perf_counter() - trad_start) * 1000

    mm.close()
    os.close(fd)

    # ─ Results ─
    gds_overhead = (gds_wall_ms / baseline_ms - 1) * 100
    trad_overhead = (trad_wall_ms / baseline_ms - 1) * 100

    print("\n  Results:")
    print(f"  {'Compute only (baseline):':40s} {fmt_time(baseline_ms):>10s}  ({fmt_time(baseline_ms / n_layers)}/layer)")
    print()
    print(f"  {'GDS pipeline (wall clock):':40s} {fmt_time(gds_wall_ms):>10s}  ({fmt_time(gds_wall_ms / n_layers)}/layer)")
    print(f"  {'GDS pipeline (GPU events):':40s} {fmt_time(gds_gpu_ms):>10s}  ({fmt_time(gds_gpu_ms / n_layers)}/layer)")
    print(f"  {'GDS overhead vs compute:':40s} {gds_overhead:+.1f}%")
    print()
    print(f"  {'Traditional pipeline (wall clock):':40s} {fmt_time(trad_wall_ms):>10s}  ({fmt_time(trad_wall_ms / n_layers)}/layer)")
    print(f"  {'Traditional overhead vs compute:':40s} {trad_overhead:+.1f}%")
    print()

    if gds_overhead < trad_overhead:
        speedup = trad_wall_ms / gds_wall_ms
        print(f"  → GDS is {speedup:.2f}x faster than traditional pipeline")
    else:
        slowdown = gds_wall_ms / trad_wall_ms
        print(f"  → Traditional is {slowdown:.2f}x faster (GDS not helping here)")

    # Cleanup
    del gpu_slots, trad_gpu_slots, pinned_bufs
    del A, W_attn, W_shared_gu, W_shared_d, W_expert_gu, W_expert_d
    del O_attn, O_shared_gu, O_shared_d, O_expert_gu, O_expert_d
    torch.cuda.empty_cache()
    os.unlink(fpath)

    return gds_wall_ms, trad_wall_ms, baseline_ms


# ─── Main ───

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPUDirect Storage Benchmark")
    parser.add_argument("--test-dir", type=str, default="/home/tim",
                        help="Directory on NVMe for test files (default: /home/tim = RAID0)")
    parser.add_argument("--size-mb", type=int, default=512,
                        help="Size for bandwidth tests (default: 512 MB)")
    parser.add_argument("--layer-mb", type=int, default=200,
                        help="Layer size for pipeline test (default: 200 MB)")
    parser.add_argument("--n-layers", type=int, default=5,
                        help="Layers for pipeline test (default: 5)")
    parser.add_argument("--tokens", type=int, default=4096,
                        help="Batch tokens for compute simulation (default: 4096)")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Test dir: {args.test_dir}")

    test_gds_status()
    gds_avg, gds_peak = test_gds_bandwidth(args.test_dir, size_mb=args.size_mb)
    trad_avg = test_traditional_bandwidth(args.test_dir, size_mb=args.size_mb)
    gds_pipe, trad_pipe, compute = test_pipeline_comparison(
        args.test_dir, n_layers=args.n_layers, layer_mb=args.layer_mb,
        batch_tokens=args.tokens,
    )

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  GDS bandwidth:         {fmt_bw(gds_avg)} avg, {fmt_bw(gds_peak)} peak")
    print(f"  Traditional bandwidth: {fmt_bw(trad_avg)} avg")
    print(f"  Speedup (raw BW):      {gds_avg / trad_avg:.2f}x")
    print()
    print(f"  Pipeline (GDS):        {fmt_time(gds_pipe)} ({(gds_pipe / compute - 1) * 100:+.1f}% overhead)")
    print(f"  Pipeline (trad):       {fmt_time(trad_pipe)} ({(trad_pipe / compute - 1) * 100:+.1f}% overhead)")
    print(f"  Compute baseline:      {fmt_time(compute)}")


if __name__ == "__main__":
    main()
