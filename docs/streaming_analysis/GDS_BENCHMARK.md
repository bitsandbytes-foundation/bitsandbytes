# GPUDirect Storage Benchmark Results

Measured NVMe → GPU streaming bandwidth using kvikio/cuFile on dettmers-desktop.
Compares GDS (NVMe → GPU direct via DMA) to the traditional path (NVMe → CPU pinned → GPU).

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB VRAM) |
| GPU PCIe | Gen 5 x16 |
| NVMe | 6× WD_BLACK SN8100 4TB (Gen5 x4, ~12 GB/s each) |
| RAID | 5 drives in md RAID0 (XFS), mounted at `/home/tim` |
| 6th NVMe | Boot drive (nvme1, separate from RAID) |
| RAM | 256 GB DDR5 |
| CPU | AMD Ryzen Threadripper PRO 9975WX 32-Cores (128 PCIe 5.0 lanes) |
| Kernel | 6.14.0-33-generic |
| Driver | 580.95.05 |
| PyTorch | 2.9.1+cu130 |
| kvikio | 26.02.000 |
| nvidia-cufile | 1.15.0.42 (pip) |

## Key Findings

### 1. RAID0 bandwidth requires parallel IO threads

kvikio's default single-threaded IO hits only one RAID stripe at a time, capping
at single-drive bandwidth (~13 GB/s). Setting `KVIKIO_NTHREADS=16` and
`KVIKIO_TASK_SIZE=1048576` (1 MB tasks) enables parallel reads across all RAID
members, achieving near-full RAID0 bandwidth:

| Configuration | Read BW (2 GB file) |
|---------------|-------------------|
| kvikio default (1 thread) | 13.1 GB/s |
| kvikio 16 threads, 1 MB tasks | **49.0 GB/s** |
| fio baseline (4 jobs, iodepth=16) | 52.5 GB/s |

**Critical setting:** Always set these environment variables before import:
```bash
export KVIKIO_NTHREADS=16
export KVIKIO_TASK_SIZE=1048576
```

### 2. GDS raw bandwidth: 49 GB/s NVMe → GPU

With RAID0 parallelized, kvikio delivers 49 GB/s reading from NVMe directly
into GPU memory. A 1237 MB MoE layer (GLM-4.7 NF4d+NF2e) reads in **24.7ms**.

| Read size | Bandwidth | Time |
|-----------|-----------|------|
| 512 MB | 49.1 GB/s | 10.2ms |
| 1237 MB (MoE layer) | 48.9 GB/s | 24.7ms |
| 2048 MB | 49.0 GB/s | 40.7ms |

Data integrity verified: GPU buffer matches file contents after every read.

### 3. Pipeline overhead: near-zero at 8K tokens

Pipelined streaming test with realistic MoE compute simulation (attention +
shared expert + 8 active routing experts at GLM-4.7 dimensions):

**1237 MB layers (full MoE layer), GDS path:**

| Tokens | Compute/layer | Pipeline overhead | Verdict |
|--------|--------------|-------------------|---------|
| 4096 | 10.8ms | +85% | Transfer dominates |
| 8192 | 21.8ms | **+7.9%** | Nearly hidden |
| ~10K+ | ~27ms+ | ~0% | Fully hidden |

Note: this is forward-only matmul compute. Real training does forward + backward
recompute per layer (~3× the forward compute), so 4096 tokens in real training
would yield ~33ms compute/layer — enough to hide the 25ms GDS read.

**200 MB layers (dense layer), GDS path:**

| Tokens | Compute/layer | Pipeline overhead |
|--------|--------------|-------------------|
| 4096 | 10.8ms | +33.5% |
| 8192 | ~21ms | ~0% |

### 4. GDS vs traditional: 4-14× faster in pipeline

The traditional path (mmap → CPU pinned → GPU) is bottlenecked by CPU-side
memory copies. Even with threading and double-buffering, it's much slower:

| Layer size | GDS pipeline | Traditional pipeline | Speedup |
|-----------|-------------|---------------------|---------|
| 200 MB (5 layers, 4K tok) | 71.9ms | 292.9ms | **4.1×** |
| 1237 MB (3 layers, 4K tok) | 60.3ms | 819.7ms | **13.6×** |

### 5. RTX PRO 6000 is GDS-compatible

The RTX PRO 6000 (Blackwell Workstation Edition) is a workstation-class GPU,
successor to the Quadro line. GDS requires Quadro or Data Center GPUs — GeForce
is not supported. This GPU works with kvikio out of the box.

Consumer GeForce GPUs (RTX 4090, 5090) **cannot use GDS**. For those cards,
the CPU pinned RAM path (CPU → GPU DMA) is the only option.

## Comparison: GDS vs CPU Pinned vs Traditional

For a 1237 MB MoE layer at 4096 tokens:

| Path | Transfer time/layer | Pipeline overhead | Requirements |
|------|-------------------|-------------------|-------------|
| **GDS (RAID0, 16 threads)** | ~25ms | +85% @ 4K, +8% @ 8K | Pro/Quadro GPU, kvikio |
| **CPU pinned RAM** | ~12ms (PCIe Gen5) | ~0% @ 4K | 110+ GB RAM |
| **Traditional (mmap)** | ~270ms | +2400% | Unusable at any batch size |

- **GDS** is best when system RAM is limited (can't hold model in pinned RAM)
  or for avoiding startup load time.
- **CPU pinned RAM** is best when system RAM is abundant (256 GB on this machine).
  Lower latency than NVMe, zero overhead at smaller batch sizes.
- For the dettmers-desktop with 256 GB RAM, CPU pinned is simpler and faster.
  GDS becomes valuable when targeting machines with less RAM.

## PCIe Topology Note

The NVMe drives and GPU are on different PCIe root complexes (NVMe on buses
10/40/c0, GPU on bus f1). P2P DMA goes through the Threadripper PRO's Infinity
Fabric rather than a direct PCIe switch. Despite this, kvikio achieves 49 GB/s —
the Infinity Fabric has ample bandwidth to handle these transfers.

For machines with a dedicated PCIe switch between NVMe and GPU, slightly higher
bandwidth may be achievable.

## Reproducing

```bash
# On dettmers-desktop (or any machine with Pro/Quadro GPU + NVMe RAID)
pip install kvikio-cu12

# Run the benchmark
export KVIKIO_NTHREADS=16
export KVIKIO_TASK_SIZE=1048576
python docs/streaming_analysis/gds_bench.py \
    --test-dir /home/tim \
    --size-mb 1237 \
    --layer-mb 1237 \
    --n-layers 3 \
    --tokens 8192
```

## Software Stack

| Component | Version | Notes |
|-----------|---------|-------|
| kvikio-cu12 | 26.02.000 | Python bindings for cuFile |
| nvidia-cufile | 1.15.0.42 | cuFile library (via pip) |
| nvidia-fs kernel module | Not installed | Not needed — kvikio uses POSIX compat or CUDA 12.8+ P2P mode |
| CUDA toolkit | 13.1 (via conda) | Bundled with PyTorch |

kvikio runs in "compatibility mode" by default (POSIX pread with internal
thread pool). Setting `KVIKIO_COMPAT_MODE=OFF` enables native GDS, but in our
tests both modes deliver the same ~49 GB/s bandwidth, suggesting the POSIX
path with 16 threads is already saturating the RAID0 and PCIe link.
