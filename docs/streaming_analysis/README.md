# CPU→GPU Weight Streaming for QLoRA Training

This document analyzes the feasibility of streaming frozen base weights from
CPU DRAM (or NVMe) to GPU during QLoRA training, eliminating the need to hold
the full model in VRAM. The analysis covers a theoretical bandwidth model,
empirical validation on an RTX 4090, and configuration grids for multiple
hardware configurations.

## Key Result

**Weight streaming works with near-zero overhead** once per-layer compute time
exceeds the PCIe transfer time. On an RTX 4090 with PCIe 3.0 x16, the
crossover is approximately 4K tokens per step. Above this, the double-buffered
pipeline hides 100% of the transfer latency with <0.5% measured overhead.

## Architecture

QLoRA freezes the base model weights and only trains low-rank adapters. The
frozen weights are read-only during both forward and backward passes, making
them ideal candidates for streaming from slower storage tiers.

### Three-tier pipeline

```
NVMe SSD ──(3.5-14 GB/s)──> CPU DRAM ──(11-50 GB/s PCIe)──> GPU VRAM
  cold storage               bandwidth buffer                 compute
```

The GPU maintains a **double buffer** (2 layer slots). While computing on one
layer, the next layer transfers asynchronously from CPU DRAM via PCIe DMA on a
dedicated CUDA stream. The CPU DRAM buffer decouples the NVMe and GPU rates.

### Double-buffer operation

```
Time ──────────────────────────────────────────────────────>
GPU:   [compute L0] [compute L1] [compute L2] [compute L3]
PCIe:  [xfer L1   ] [xfer L2   ] [xfer L3   ] [xfer L4   ]
NVMe→CPU:  [read L2      ] [read L3      ] [read L4      ]
```

Each layer occupies one slot. After the GPU finishes computing a layer, that
slot is freed for the next incoming transfer. No GPU idle time occurs as long
as `compute_time >= transfer_time`.

## Theoretical Model

### Per-layer timing

For a transformer layer with `P_active` active parameters:

```
compute_ms = tokens × 3 × 2 × P_active / GPU_FLOPS × 1000
                      ↑   ↑
                      │   └─ 2 FLOPs per multiply-accumulate
                      └───── 3× for training (forward + backward ≈ 3× forward)

transfer_ms = layer_size_bytes / pcie_bandwidth × 1000
```

For MoE models, `P_active` is the active subset (routed experts + attention +
shared expert), while `layer_size_bytes` includes **all** experts since the
routing decision is token-dependent.

### GPU ring buffer sizing

```
K_ring = max(2, ceil(transfer_ms / compute_ms) + 1)
```

The ring buffer holds `K_ring` layers. The GPU processes `K_ring - 1` layers
while one layer transfers. When `compute_ms > transfer_ms`, `K_ring = 2`
(double buffer) suffices.

### NVMe CPU buffer sizing

The CPU DRAM buffer must absorb the rate mismatch between NVMe reads and GPU
consumption. Over the total processing time, NVMe delivers:

```
nvme_delivered = n_layers × layer_cycle_ms / nvme_ms
cpu_buffer_layers = n_layers - K_ring - nvme_delivered
```

Where `layer_cycle_ms = max(compute_ms, transfer_ms)`.

### Pipeline throughput

```
step_time = n_layers × max(compute_ms, transfer_ms, nvme_ms)
```

The slowest leg (compute, PCIe, or NVMe) determines throughput. Buffering
shifts when the bottleneck hits, but doesn't change the steady-state rate.

## Measured Hardware Parameters (RTX 4090 + PCIe 3.0)

| Parameter | Theoretical | Measured |
|---|---|---|
| PCIe Gen3 x16 H2D (pinned) | 13 GB/s | **11 GB/s** (85% eff.) |
| PCIe Gen3 x16 H2D (pageable) | — | **7 GB/s** |
| GPU FP16 tensor throughput | 330 TFLOPS | **160 TFLOPS** |
| Transfer/layer (470 MB, Llama-70B) | 36ms | **43ms** |
| Compute/layer @ 4K tokens | 61ms | **50ms** |
| Compute/layer @ 8K tokens | 122ms | **100ms** |
| Pipeline overhead (compute > transfer) | 0% | **<0.5%** |

The GPU achieves ~160 TFLOPS on these matmul shapes (not the peak 330 TFLOPS,
which requires ideal tile sizes). PCIe runs at 85% of theoretical due to
protocol overhead. Both deviations are consistent and predictable.

## Benchmark Results

### Pipeline overhead vs. batch size (Llama-70B, 470 MB/layer)

| Tokens | Compute/layer | Transfer/layer | Overhead | Verdict |
|--------|--------------|----------------|----------|---------|
| 512 | 6.4ms | 42ms | +536% | PCIe-limited |
| 1024 | 12.6ms | 43ms | +224% | PCIe-limited |
| 2048 | 24.5ms | 43ms | +67% | PCIe-limited |
| **4096** | **50ms** | **43ms** | **+0.2%** | **Fully hidden** |
| 8192 | 100ms | 42ms | +0.4% | Fully hidden |

The crossover is sharp: below ~4K tokens the GPU idles waiting for PCIe;
above it, transfers are completely hidden behind compute.

### Memory savings

For the pipeline test with 20 × 470 MB layers (9.2 GB total weights):

- GPU ring buffer (2 layers): **0.92 GB**
- GPU peak memory: **3.55 GB** (ring + activations + compute buffers)
- VRAM savings: **90%**

Extrapolated to full Llama-70B (80 layers, 38 GB total):

- GPU ring buffer: **0.94 GB** (2 layers)
- Remaining 78 layers: in CPU DRAM or NVMe
- VRAM savings: **97.5%**

## Configuration Grids

### Llama-70B (dense, 80 layers, 856M params/layer, 38 GB NF4)

Minimum feasible batch size per hardware configuration:

| NVMe config | Gen3 x16 PCIe | Gen4 x16 | Gen5 x16 |
|---|---|---|---|
| 1× Gen3 (3.5 GB/s) | 1K (11s/step) | 4K (11s) | 4K (11s) |
| 2× Gen3 (7 GB/s) | 1K (5s) | 1K (5s) | 2K (5s) |
| 1× Gen4 (7 GB/s) | 1K (5s) | 1K (5s) | 2K (5s) |
| 2× Gen5 (28 GB/s) | n/a | n/a | 1K (1s) |

Dense models are straightforward — 100% of transferred weights contribute to
compute, so even slow NVMe works at small batch sizes.

### GLM-4.7 (355B MoE, 92 layers, 4.1B params/layer, 207 GB NF4)

The MoE architecture creates a poor weight-to-compute ratio: each layer
transfers 2.25 GB (all 160 experts) but only 12.5% (8 active experts +
attention + shared) contributes FLOPs. This makes the model significantly
harder to stream.

**With 32 GB system RAM:**

| NVMe config | Gen4 x16 | Gen5 x16 |
|---|---|---|
| 1× Gen4 (7 GB/s) | 32K (30s) | 32K (30s) |
| 2× Gen4 (14 GB/s) | 16K (15s) | 16K (15s) |
| 2× Gen5 (28 GB/s) | n/a | **8K (7s)** |
| 4× Gen4 (28 GB/s) | n/a | **8K (7s)** |

**With 128 GB system RAM** (larger CPU buffer absorbs NVMe rate mismatch):

| NVMe config | Gen4 x16 | Gen5 x16 |
|---|---|---|
| 2× Gen4 (14 GB/s) | 2K (15s) | 8K (15s) |
| 2× Gen5 (28 GB/s) | n/a | **1K (7s)** |
| 4× Gen4 (28 GB/s) | n/a | **1K (7s)** |

### Effect of lower-bit quantization on GLM-4.7

Lower quantization reduces transfer size without changing compute (weights
are dequantized to FP16 before matmul):

| Quantization | Layer size | Model size | Transfer/layer | Min tokens (0% overhead) |
|---|---|---|---|---|
| NF4 (4-bit) | 2.25 GB | 207 GB | 205ms | ~11K |
| NF3 (3-bit) | 1.64 GB | 151 GB | 149ms | ~8K |
| NF2 (2-bit) | 1.15 GB | 106 GB | 104ms | ~5K |

At NF2, the 355B MoE model's per-layer transfer time approaches that of a 70B
dense model at NF4, making streaming much more practical.

## Implementation Notes

### Critical for correct overlap

1. **Pinned memory**: CPU buffers must use `pin_memory=True`. Pageable memory
   drops bandwidth from 11 GB/s to 7 GB/s and prevents true async DMA.

2. **Pre-allocated output buffers**: Use `torch.mm(A, B, out=C)` instead of
   `C = torch.mm(A, B)`. Temporary tensor allocations cause implicit CUDA
   synchronizations that serialize the pipeline. In testing, this single change
   reduced pipeline overhead from 78% to <0.5%.

3. **Dedicated copy stream**: Use a separate `torch.cuda.Stream()` for H2D
   transfers. The default stream serializes all operations.

4. **Stream synchronization**: After compute, call
   `torch.cuda.current_stream().wait_stream(copy_stream)` before the next
   iteration to ensure the incoming layer is ready.

### What doesn't work

- **torch.mm() without `out=`** in the pipeline loop — causes CUDA allocator
  syncs, defeating the overlap.
- **Pageable (non-pinned) CPU memory** — the CUDA runtime copies through an
  internal staging buffer, halving bandwidth and preventing overlap.
- **Single CUDA stream** — serializes compute and transfer.

## Running the Benchmark

```bash
# Llama-70B layer size, 4K tokens (should show ~0% overhead)
python docs/streaming_analysis/stream_bench.py --layer-mb 470 --pipeline-tokens 4096

# GLM-4.7 layer size (all experts), 8K tokens
python docs/streaming_analysis/stream_bench.py --layer-mb 2250 --pipeline-tokens 8192

# With NVMe read test
python docs/streaming_analysis/stream_bench.py --layer-mb 470 --nvme /mnt/nvme

# Custom model dimensions
python docs/streaming_analysis/stream_bench.py \
  --layer-mb 470 --hidden 8192 --intermediate 28672 \
  --pipeline-tokens 4096 --n-layers 20
```

### Interpreting results

- **Test 1** (PCIe bandwidth): Should show ~11 GB/s pinned on Gen3, ~24 GB/s
  on Gen4. Pageable should be noticeably slower.
- **Test 3** (matmul throughput): Shows actual TFLOPS on your GPU. Use this
  instead of the theoretical peak for planning.
- **Test 4** (overlap): Single-shot overlap test. Should show >2x speedup when
  compute dominates transfer.
- **Test 5** (full pipeline): The definitive test. Compare "pipeline overhead
  vs compute-only" — should be <5% when compute > transfer per layer.
