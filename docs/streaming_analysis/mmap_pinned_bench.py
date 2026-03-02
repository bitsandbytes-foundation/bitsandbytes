"""Benchmark: mmap → pinned copy overhead for NVMe weight streaming.

Measures the bandwidth of copying data from an mmap'd file (pageable memory)
into CPU pinned buffers, which is the critical path for low-RAM machines that
can't pre-load all weights into pinned memory.

The pipeline for low-RAM streaming is:
  NVMe → OS page cache (via mmap page fault) → CPU pinned staging buffer → GPU

This benchmark measures the first two stages (NVMe → pinned) to determine
whether the extra pageable→pinned memcpy is a bottleneck.

Tests:
  1. mmap → pinned copy at varying chunk sizes
  2. safetensors safe_open → get_tensor → copy to pinned
  3. Direct file read (O_RDONLY) → pinned copy for comparison
  4. Estimated layer transfer times at realistic MoE sizes

Usage:
    python docs/streaming_analysis/mmap_pinned_bench.py [--file-path PATH] [--file-size-gb N]
"""

import argparse
import mmap
import os
import struct
import sys
import tempfile
import time

import numpy as np
import torch

# ─── Helpers ───

def fmt_bw(gb_per_s):
    if gb_per_s >= 1:
        return f"{gb_per_s:.2f} GB/s"
    return f"{gb_per_s * 1000:.1f} MB/s"


def fmt_time(sec):
    if sec >= 1:
        return f"{sec:.2f}s"
    return f"{sec * 1000:.1f}ms"


def drop_caches():
    """Try to drop OS page caches. Requires sudo or appropriate permissions."""
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except PermissionError:
        return False


def create_test_file(path: str, size_bytes: int):
    """Create a test file filled with random data."""
    chunk = 64 * 1024 * 1024  # 64 MB write chunks
    written = 0
    with open(path, "wb") as f:
        while written < size_bytes:
            to_write = min(chunk, size_bytes - written)
            f.write(os.urandom(to_write))
            written += to_write
    print(f"  Created {path} ({size_bytes / 1e9:.1f} GB)")


def create_safetensors_file(path: str, tensor_sizes_bytes: list[int]):
    """Create a minimal safetensors file with tensors of given byte sizes.

    Each tensor is stored as int32 (matches quantized packed format).
    """
    import json

    header = {}
    metadata = {"format": "benchmark"}
    header["__metadata__"] = metadata

    offset = 0
    for i, sz in enumerate(tensor_sizes_bytes):
        numel = sz // 4  # int32
        header[f"tensor_{i}"] = {
            "dtype": "I32",
            "shape": [numel],
            "data_offsets": [offset, offset + sz],
        }
        offset += sz

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_size = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_json)
        # Write random data for each tensor
        for sz in tensor_sizes_bytes:
            f.write(os.urandom(sz))

    print(f"  Created safetensors {path} ({offset / 1e6:.0f} MB data, {len(tensor_sizes_bytes)} tensors)")


# ─── Test 1: mmap → pinned copy ───

def test_mmap_to_pinned(file_path: str, chunk_sizes_mb: list[int], n_repeats: int = 5):
    """Copy chunks from mmap'd file to pinned CPU buffer."""
    print(f"\n{'=' * 70}")
    print("Test 1: mmap → pinned copy bandwidth")
    print(f"{'=' * 70}")

    file_size = os.path.getsize(file_path)
    fd = os.open(file_path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

    results = []
    for chunk_mb in chunk_sizes_mb:
        chunk_bytes = chunk_mb * 1024 * 1024
        if chunk_bytes > file_size:
            print(f"  {chunk_mb:>6} MB: SKIP (> file size)")
            continue

        # Allocate pinned buffer
        numel = chunk_bytes // 4
        pinned = torch.empty(numel, dtype=torch.int32, device="cpu", pin_memory=True)
        pinned_np = pinned.numpy()

        # Drop caches if possible
        can_drop = drop_caches()
        cache_status = "cold" if can_drop else "warm"

        times = []
        for r in range(n_repeats):
            if can_drop:
                drop_caches()
                # Re-create mmap to ensure fresh page faults
                mm.close()
                os.close(fd)
                fd = os.open(file_path, os.O_RDONLY)
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

            offset = 0  # always read from start
            t0 = time.perf_counter()
            pinned_np[:] = np.frombuffer(mm[offset:offset + chunk_bytes], dtype=np.int32)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg = sum(times) / len(times)
        bw = chunk_bytes / avg / 1e9
        results.append((chunk_mb, avg, bw, cache_status))
        print(f"  {chunk_mb:>6} MB ({cache_status}): {fmt_bw(bw):>12}  ({fmt_time(avg)} avg, n={n_repeats})")

        del pinned

    mm.close()
    os.close(fd)
    return results


# ─── Test 2: safetensors safe_open → get_tensor → copy to pinned ───

def test_safetensors_to_pinned(st_path: str, n_repeats: int = 5):
    """Load tensors via safetensors safe_open, then copy to pinned."""
    print(f"\n{'=' * 70}")
    print("Test 2: safetensors safe_open → get_tensor → copy to pinned")
    print(f"{'=' * 70}")

    from safetensors import safe_open

    # Get tensor names and sizes
    f = safe_open(st_path, framework="pt", device="cpu")
    tensor_names = [k for k in f.keys()]
    print(f"  {len(tensor_names)} tensors in file")

    results = []
    for name in tensor_names:
        t_ref = f.get_tensor(name)
        sz_bytes = t_ref.numel() * t_ref.element_size()
        sz_mb = sz_bytes / 1e6

        # Pre-allocate pinned buffer
        pinned = torch.empty_like(t_ref, device="cpu", pin_memory=True)

        can_drop = drop_caches()
        cache_status = "cold" if can_drop else "warm"

        times = []
        for r in range(n_repeats):
            if can_drop:
                drop_caches()

            # Re-open to avoid caching in safe_open
            f2 = safe_open(st_path, framework="pt", device="cpu")
            t0 = time.perf_counter()
            tensor = f2.get_tensor(name)
            pinned.copy_(tensor)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            del tensor, f2

        avg = sum(times) / len(times)
        bw = sz_bytes / avg / 1e9
        results.append((name, sz_mb, avg, bw, cache_status))
        print(f"  {name}: {sz_mb:.0f} MB ({cache_status}): {fmt_bw(bw):>12}  ({fmt_time(avg)} avg)")

        del pinned

    return results


# ─── Test 3: Direct file read → pinned ───

def test_direct_read_to_pinned(file_path: str, chunk_sizes_mb: list[int], n_repeats: int = 5):
    """Read file directly into a numpy view of pinned memory."""
    print(f"\n{'=' * 70}")
    print("Test 3: Direct file read → pinned copy")
    print(f"{'=' * 70}")

    file_size = os.path.getsize(file_path)

    results = []
    for chunk_mb in chunk_sizes_mb:
        chunk_bytes = chunk_mb * 1024 * 1024
        if chunk_bytes > file_size:
            print(f"  {chunk_mb:>6} MB: SKIP (> file size)")
            continue

        numel = chunk_bytes // 4
        pinned = torch.empty(numel, dtype=torch.int32, device="cpu", pin_memory=True)
        pinned_np = pinned.numpy()

        can_drop = drop_caches()
        cache_status = "cold" if can_drop else "warm"

        times = []
        for r in range(n_repeats):
            if can_drop:
                drop_caches()

            t0 = time.perf_counter()
            with open(file_path, "rb") as fobj:
                data = fobj.read(chunk_bytes)
                pinned_np[:] = np.frombuffer(data, dtype=np.int32)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg = sum(times) / len(times)
        bw = chunk_bytes / avg / 1e9
        results.append((chunk_mb, avg, bw, cache_status))
        print(f"  {chunk_mb:>6} MB ({cache_status}): {fmt_bw(bw):>12}  ({fmt_time(avg)} avg, n={n_repeats})")

        del pinned

    return results


# ─── Test 4: Estimated layer transfer times ───

def test_layer_estimates(mmap_results: list, direct_results: list):
    """Estimate per-layer transfer times at realistic MoE sizes."""
    print(f"\n{'=' * 70}")
    print("Test 4: Estimated layer transfer times")
    print(f"{'=' * 70}")

    # Use the largest chunk size bandwidth as the representative rate
    if mmap_results:
        # Find the largest chunk bandwidth
        mmap_bw = max(r[2] for r in mmap_results)
        print(f"  Best mmap→pinned bandwidth: {fmt_bw(mmap_bw)}")
    else:
        mmap_bw = 0

    if direct_results:
        direct_bw = max(r[2] for r in direct_results)
        print(f"  Best direct read bandwidth: {fmt_bw(direct_bw)}")
    else:
        direct_bw = 0

    # Reference: PCIe Gen4 H2D on RTX 4090 = ~11 GB/s
    pcie_bw = 11.0

    layer_sizes = {
        "Dense layer (190 MB)": 190,
        "MoE layer NF4d+NF2e (1237 MB)": 1237,
        "MoE layer NF4 (2250 MB)": 2250,
    }

    print(f"\n  {'Layer type':<35} {'mmap→pin':>10} {'direct→pin':>12} {'PCIe H2D':>10} {'Bottleneck':>12}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")

    for name, size_mb in layer_sizes.items():
        size_gb = size_mb / 1000

        mmap_time = (size_gb / mmap_bw) if mmap_bw > 0 else float("inf")
        direct_time = (size_gb / direct_bw) if direct_bw > 0 else float("inf")
        pcie_time = size_gb / pcie_bw

        # Pipeline: mmap→pinned overlaps with PCIe H2D
        # Bottleneck is max(mmap→pinned, PCIe H2D)
        bottleneck = "mmap→pin" if mmap_time > pcie_time else "PCIe H2D"

        print(
            f"  {name:<35} {fmt_time(mmap_time):>10} {fmt_time(direct_time):>12} "
            f"{fmt_time(pcie_time):>10} {bottleneck:>12}"
        )


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Benchmark mmap → pinned copy")
    parser.add_argument(
        "--file-path",
        default="/media/tim/D/mmap_bench_test.bin",
        help="Path for test file (should be on NVMe)",
    )
    parser.add_argument("--file-size-gb", type=float, default=2.0, help="Test file size in GB")
    parser.add_argument("--n-repeats", type=int, default=5, help="Number of repeats per measurement")
    parser.add_argument("--skip-create", action="store_true", help="Skip file creation if it already exists")
    args = parser.parse_args()

    print("mmap → pinned copy benchmark")
    print(f"Machine: {os.uname().nodename}")
    print(f"RAM: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.0f} GB")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Check pinned memory works
    try:
        test_pin = torch.empty(1024, pin_memory=True)
        del test_pin
        print("Pinned memory: OK")
    except RuntimeError as e:
        print(f"Pinned memory FAILED: {e}")
        return

    file_size_bytes = int(args.file_size_gb * 1024 * 1024 * 1024)

    # Create test files
    print(f"\n--- Setup ---")
    raw_path = args.file_path
    st_path = raw_path.replace(".bin", ".safetensors")

    if not args.skip_create or not os.path.exists(raw_path):
        print("Creating raw test file...")
        create_test_file(raw_path, file_size_bytes)
    else:
        print(f"Using existing {raw_path} ({os.path.getsize(raw_path) / 1e9:.1f} GB)")

    # Create safetensors file with realistic layer sizes
    # MoE layer: ~1237 MB, Dense layer: ~190 MB
    st_tensor_sizes = [
        190 * 1024 * 1024,   # dense layer
        1237 * 1024 * 1024,  # MoE layer
    ]
    if not args.skip_create or not os.path.exists(st_path):
        print("Creating safetensors test file...")
        create_safetensors_file(st_path, st_tensor_sizes)
    else:
        print(f"Using existing {st_path}")

    # Run tests
    chunk_sizes = [1, 10, 100, 190, 500, 1000, 1237]
    # Filter to chunks that fit in the file
    chunk_sizes = [c for c in chunk_sizes if c * 1024 * 1024 <= file_size_bytes]

    mmap_results = test_mmap_to_pinned(raw_path, chunk_sizes, args.n_repeats)
    direct_results = test_direct_read_to_pinned(raw_path, chunk_sizes, args.n_repeats)
    test_safetensors_to_pinned(st_path, args.n_repeats)
    test_layer_estimates(mmap_results, direct_results)

    # Cleanup
    print(f"\n--- Cleanup ---")
    print(f"Test files left at:\n  {raw_path}\n  {st_path}")
    print("Delete manually when done.")


if __name__ == "__main__":
    main()
