# NVMe Weight Streaming: Partial-Resident QLoRA Training

Training 70B–355B models on consumer GPUs by keeping a fraction of quantized
weights on-GPU and streaming the rest from CPU pinned memory or NVMe.

## Key Results

- **Llama-70B NF4 on 1× RTX 4090**: 49% resident on GPU, 51% streamed.
  Zero overhead at 1024 tokens (Gen4 NVMe + 32 GB RAM). A single $1600 GPU
  fine-tunes a 70B model.
- **GLM-4.7 355B MoE on 1× RTX 4090 (BF16)**: 15% resident, 85% streamed.
  Zero overhead at 8K tokens (batch 8 × 1024). A 355B model on one consumer
  GPU with a Gen4 NVMe and 32 GB RAM.
- **GLM-4.7 355B MoE on 1× RTX 5090 (NVFP4)**: 22% resident, 78% streamed.
  Zero overhead at 16K tokens with 1× Gen5 NVMe, or 8K tokens with 2× Gen5
  RAID or CPU pinned RAM.
- **GLM-4.7 355B on 1× RTX PRO 6000 Blackwell**: 79% resident, 21% streamed.
  Zero overhead at 8K tokens with GDS + NVMe RAID0 (49 GB/s measured).

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Consumer vs Workstation/Datacenter GPUs](#consumer-vs-workstationdatacenter-gpus)
3. [Memory Hierarchy and DMA](#memory-hierarchy-and-dma)
4. [Partial-Resident Streaming](#partial-resident-streaming)
5. [Theoretical Model](#theoretical-model)
6. [Dense Models: Llama-70B](#dense-models-llama-70b)
7. [MoE Models: GLM-4.7 355B](#moe-models-glm-47-355b)
8. [Mixed Quantization](#mixed-quantization)
9. [NVMe Streaming with 32 GB RAM](#nvme-streaming-with-32-gb-ram)
10. [RAID Convergence](#raid-convergence)
11. [Multi-GPU Pipeline Parallelism](#multi-gpu-pipeline-parallelism)
12. [Batch Size and Token Counts](#batch-size-and-token-counts)
13. [Hardware Recommendations](#hardware-recommendations)
14. [Implementation Notes](#implementation-notes)

---

## Architecture Overview

QLoRA freezes base model weights and trains only low-rank adapters. The frozen
weights are read-only during forward and backward passes, making them candidates
for streaming from slower storage tiers.

### Two streaming paths

The streaming path depends on the GPU class:

**Consumer GPUs (GeForce — RTX 4090, 5090):** NVMe → CPU → GPU

```
NVMe SSD ──(3.5–13 GB/s)──> CPU pinned buffer ──(11–22 GB/s PCIe DMA)──> GPU
  cold storage               4 rotating layer slots                  compute + LoRA
```

GeForce GPUs do not support GPUDirect Storage P2P DMA. All NVMe data must pass
through CPU memory as a staging area. The effective bandwidth is
`min(NVMe_aggregate, PCIe_bandwidth)`.

**Workstation/Datacenter GPUs (RTX PRO, Quadro, A100, H100):** NVMe → GPU direct

```
NVMe SSD ──(up to 49 GB/s via kvikio)──> GPU VRAM
  GDS P2P DMA bypasses CPU entirely        compute + LoRA
```

GPUDirect Storage (GDS) enables NVMe controllers to DMA directly into GPU
memory via PCIe peer-to-peer. With NVMe RAID0 and kvikio's parallel IO threads,
this achieves up to 49 GB/s (measured on 5× Gen5 NVMe RAID0).

### Pipeline stages (consumer path)

```
Time ───────────────────────────────────────────────────────────>
NVMe→CPU:  [read L(i+2)     ] [read L(i+3)     ] [read L(i+4)     ]
CPU→GPU:        [DMA  L(i+1)] [DMA  L(i+2)     ] [DMA  L(i+3)     ]
GPU:       [compute L(i)    ] [compute L(i+1)  ] [compute L(i+2)  ]
```

The GPU's DMA engine handles CPU→GPU transfers on a dedicated CUDA copy stream,
overlapping with compute on the default stream. No GPU idle time occurs when
compute time ≥ effective transfer time per layer.

### What stays on GPU

| Component | Location | Size (GLM-4.7 example) |
|---|---|---|
| Resident quantized weights | GPU VRAM | 14–73 layers (varies by GPU) |
| GPU double-buffer (2 layer slots) | GPU VRAM | 2.4 GB |
| LoRA adapters (all layers) | GPU VRAM | 0.4–0.7 GB |
| LoRA gradients | GPU VRAM | 0.4–0.7 GB |
| Adam optimizer states | GPU VRAM | 1.5–2.7 GB |
| Activations (1 layer, grad ckpt) | GPU VRAM | ~0.1 GB |
| CUDA context | GPU VRAM | ~1.5 GB |

### What stays off GPU

| Component | Location | Size |
|---|---|---|
| Streamed quantized weights | CPU pinned RAM or NVMe | Remaining layers |
| CPU staging buffer (4 layer slots) | CPU pinned RAM | 4.8 GB |

---

## Consumer vs Workstation/Datacenter GPUs

The GPU class determines which streaming path is available and what bandwidth
is achievable:

| Feature | Consumer (GeForce) | Workstation (RTX PRO/Quadro) | Datacenter (A100/H100) |
|---|---|---|---|
| GPUDirect Storage P2P | No (compat mode only) | Yes | Yes |
| NVMe path | NVMe → CPU → GPU | NVMe → GPU direct | NVMe → GPU direct |
| Effective BW ceiling | PCIe bandwidth | NVMe aggregate BW | NVMe aggregate BW |
| NVFP4 tensor cores | RTX 5090 only | RTX PRO 6000 Blackwell | No (Hopper: FP8) |
| Max single-GPU VRAM | 32 GB (5090) | 96 GB (RTX PRO 6000) | 80 GB (H100) |

### Consumer GPU streaming strategy

On consumer GPUs, the optimal approach depends on available system RAM:

| System RAM | Strategy | Effective bandwidth |
|---|---|---|
| ≥ model size (e.g., 128 GB for 111 GB model) | Load into CPU pinned RAM at startup | PCIe bandwidth (11–22 GB/s) |
| < model size (e.g., 32 GB) | Stream from NVMe every step | min(NVMe, PCIe) |

With sufficient RAM, NVMe is only used at startup. During training, the loop
is purely PCIe DMA from pinned DRAM — NVMe speed is irrelevant.

With insufficient RAM, NVMe feeds the pipeline every step. The effective
bandwidth is `min(NVMe_aggregate, PCIe)`, and RAID0 can close the gap
(see [RAID Convergence](#raid-convergence)).

### Workstation/Datacenter GPU streaming strategy

With GDS, the GPU reads directly from NVMe. The effective bandwidth is the
NVMe aggregate read speed (not capped by PCIe since there's no CPU bounce).

```bash
# Required setup for GDS with RAID0
pip install kvikio-cu12
export KVIKIO_NTHREADS=16        # Parallelize across RAID stripes
export KVIKIO_TASK_SIZE=1048576   # 1 MB task granularity
```

Measured bandwidth (RTX PRO 6000 + 5× WD SN8100 Gen5 RAID0): **49 GB/s**.
See `GDS_BENCHMARK.md` for full results.

---

## Memory Hierarchy and DMA

### Why pinned memory bypasses the CPU

With **pinned (page-locked) memory**, the GPU's DMA engine reads directly from
DRAM without CPU involvement:

```
GPU DMA engine ──> PCIe bus ──> CPU memory controller ──> DRAM chips
                                (NOT through CPU cores)
```

The CPU cores are completely uninvolved. The memory controller services the
PCIe read requests directly from DRAM. Since DRAM bandwidth (DDR4 dual-channel:
~50 GB/s, DDR5: ~80 GB/s) far exceeds PCIe bandwidth (11–22 GB/s), DRAM is
never the bottleneck.

With **regular (pageable) memory**, the CUDA runtime must:
1. Check if the page is in physical RAM (not swapped)
2. Copy data through an internal pinned staging buffer
3. DMA from the staging buffer to GPU

This halves effective bandwidth (measured: 7 GB/s vs 11 GB/s on Gen3) and
prevents true async overlap.

### Bandwidth hierarchy (measured values where available)

| Link | Bandwidth | Notes |
|---|---|---|
| DDR4 dual-channel | ~50 GB/s | Never the bottleneck |
| DDR5 dual-channel | ~80 GB/s | Never the bottleneck |
| PCIe Gen3 x16 | 11 GB/s (measured) | RTX 3090 era |
| PCIe Gen4 x16 | ~11 GB/s (measured H2D) | RTX 4090 |
| PCIe Gen5 x16 | ~27 GB/s (measured H2D) | RTX 5090 / Blackwell workstation |
| Gen3 NVMe (e.g., 970 EVO) | 3.5 GB/s (measured) | Sequential read |
| Gen4 NVMe (e.g., 980 PRO) | 7 GB/s | Sequential read |
| Gen5 NVMe (e.g., SN8100) | 13 GB/s (measured) | Sequential read, sustained |
| 5× Gen5 NVMe RAID-0 | 49 GB/s (measured) | Via kvikio with 16 IO threads |
| 5× Gen5 NVMe RAID-0 (fio) | 52.5 GB/s (measured) | Raw OS-level ceiling |

---

## Partial-Resident Streaming

Instead of streaming all layers from CPU/NVMe, keep a fraction **permanently
resident on GPU**. The resident layers require zero transfer — the GPU accesses
them directly from VRAM.

### Why partial residency helps

For every streamed layer, the GPU must wait for its transfer. But resident
layers compute "for free" (no transfer needed), creating windows where the
copy stream can work on upcoming streamed layers.

If fraction `f` of layers are streamed (and `1 - f` are resident), each
streamed layer gets `1/f` compute periods to complete its transfer:

```
Zero overhead when: (1/f) × compute_time ≥ transfer_time
Equivalently:       compute_time ≥ f × transfer_time
```

| Resident fraction | Streamed f | Effective threshold |
|---|---|---|
| 0% (all streamed) | 1.00 | compute ≥ 1.00 × transfer |
| 25% | 0.75 | compute ≥ 0.75 × transfer |
| 50% | 0.50 | compute ≥ 0.50 × transfer |
| 67% | 0.33 | compute ≥ 0.33 × transfer |
| 79% (RTX PRO 6000 + GLM-4.7) | 0.21 | compute ≥ 0.21 × transfer |

Higher residency lowers the batch size threshold for zero overhead. The GPU
VRAM budget determines how many layers can be resident.

### GPU VRAM budget

```
resident_weight = max_resident_layers × layer_size
gpu_double_buffer = 2 × layer_size
lora_training = lora_params + gradients + adam_states
overhead = cuda_context + activations

VRAM = resident_weight + gpu_double_buffer + lora_training + overhead
```

The double buffer is small (2 layers) regardless of how many layers are
streamed. This is the key insight: the buffer cost is O(1), not O(n_layers).

---

## Theoretical Model

### Per-layer timing

```
compute_ms = total_tokens × 3 × 2 × P_active / GPU_TFLOPS × 1000
             ^               ^   ^
             B × S           |   └─ 2 FLOPs per multiply-accumulate
                             └───── 3× for training (fwd + bwd ≈ 3× fwd)

transfer_ms = layer_size_bytes / effective_bandwidth × 1000
```

Where `total_tokens = batch_size × sequence_length`. The GPU doesn't
distinguish between more sequences and longer sequences — compute scales
with the product.

For MoE models, `P_active` includes only the routed experts, attention, and
shared expert. `layer_size_bytes` includes **all** experts.

### GPU compute throughput

| GPU | BF16 TFLOPS | NVFP4 TFLOPS | Compute/token (GLM-4.7) |
|---|---|---|---|
| RTX 4090 | 160 | — | 0.0193 ms |
| RTX 5090 (BF16) | 210 | — | 0.0147 ms |
| RTX 5090 (NVFP4) | — | 567 | 0.0054 ms |
| RTX PRO 6000 Blackwell (BF16) | 210 | — | 0.0147 ms |
| RTX PRO 6000 Blackwell (NVFP4) | — | 567 | 0.0054 ms |
| A100 | 156 | — | 0.0198 ms |
| H100 | 330 | — | 0.0094 ms |

**NVFP4 on MoE models:** The 2.7× NVFP4 throughput boost primarily accelerates
the dense layers (attention + shared expert), which are compute-bound large
matmuls. The routing expert matmuls are smaller (expert_intermediate=1536) and
tend toward memory-bandwidth-bound, benefiting less from NVFP4. For MoE, the
effective per-layer speedup is closer to ~2× than 2.7×, but we use the full
2.7× for conservative threshold planning (ensures zero overhead even in the
best-case compute scenario).

### Bytes per FLOP: the key metric

The ratio of bytes transferred to FLOPs computed determines how hard a model
is to stream:

```
bytes_per_FLOP = layer_size_bytes / (3 × 2 × P_active)
```

| Model | Layer size | Active params | Bytes/FLOP | Streaming difficulty |
|---|---|---|---|---|
| Llama-70B NF4 | 470 MB | 1.05B | 0.075 | Easy |
| Llama-70B NF3 | 342 MB | 1.05B | 0.054 | Very easy |
| GLM-4.7 NF4 | 2250 MB | 514M | 0.730 | Hard |
| GLM-4.7 NF4d+NF2e | 1237 MB | 514M | 0.401 | Moderate |
| GLM-4.7 NF2 | 1150 MB | 514M | 0.373 | Moderate |

Dense models have bytes/FLOP < 0.1 — nearly every transferred byte does
useful work. MoE models at NF4 have bytes/FLOP > 0.7 — most transferred
data is inactive experts. Mixed quantization (NF4 dense + NF2 experts)
brings MoE models down to ~0.4.

---

## Dense Models: Llama-70B

### Layer characteristics

| Property | Value |
|---|---|
| Layers | 80 |
| Layer size (NF4) | 470 MB |
| Layer size (NF3) | 342 MB |
| Total model (NF4) | 36.7 GB |
| Active params/layer | 1.05B (100% — dense) |

### Streaming configurations (32 GB RAM, NVMe in the loop)

| GPU | Compute | Resident / Streamed | Gen3 NVMe | Gen4 NVMe | Gen5 NVMe |
|---|---|---|---|---|---|
| 1× RTX 4090 (24G), NF4 | BF16 | 41 / 39 (49%) | 0% @ 2048t | **0% @ 1024t** | 0% @ 512t |
| 1× RTX 4090 (24G), NF3 | BF16 | 57 / 23 (29%) | 0% @ 1024t | **0% @ 512t** | 0% @ 256t |
| 1× RTX 5090 (32G), NF4 | BF16 | 58 / 22 (28%) | 0% @ 2048t | 0% @ 1024t | 0% @ 512t |
| 1× RTX 5090 (32G), NF4 | NVFP4 | 58 / 22 (28%) | 0% @ 4096t | 0% @ 2048t | 0% @ 1024t |
| 1× RTX 5090 (32G), NF3 | any | ALL ON GPU | — | — | — |
| 1× A100/H100 (80G) | any | ALL ON GPU | — | — | — |
| 1× RTX PRO 6000 (96G) | any | ALL ON GPU | — | — | — |

Dense models are the streaming sweet spot. A single RTX 4090 with a Gen4
NVMe reaches zero overhead at just 1024 tokens — that's batch=1 with 1K
context. NVFP4 on the RTX 5090 raises the threshold by 2.7×, but dense models
are so compute-efficient per byte that even 2048–4096 tokens is easy.

With NF3, the model shrinks to 27 GB. A single RTX 5090 (32G) fits it
entirely with no streaming needed.

### Measured results (RTX 4090 + PCIe Gen3)

| Total tokens | Compute/layer | Transfer/layer | Overhead |
|---|---|---|---|
| 512 | 6.4 ms | 42 ms | +536% |
| 1024 | 12.6 ms | 43 ms | +224% |
| 2048 | 24.5 ms | 43 ms | +67% |
| **4096** | **50 ms** | **43 ms** | **<1%** |
| 8192 | 100 ms | 42 ms | <1% |

Note: this is on Gen3 PCIe (11 GB/s). On Gen4 (22 GB/s) the crossover
halves to ~2048 tokens. With 49% residency, it halves again to ~1024.

---

## MoE Models: GLM-4.7 355B

### Layer characteristics

| Property | Value |
|---|---|
| Layers | 92 |
| Total params/layer | 4.10B |
| Routing experts | 160 (each ~23.6M params) |
| Active experts/token | 8 |
| Dense params (attn + shared) | 324M (7.9% of layer) |
| Expert params | 3776M (92.1% of layer) |
| Active params/layer | 514M (12.5% of total) |

### The MoE streaming challenge

GLM-4.7 transfers 2.6× more data per layer than Llama-70B but computes on
only 0.5× the parameters. This 5× worse compute-to-transfer ratio means
GLM-4.7 needs 5× more tokens to hide the same transfer latency.

| Metric | Llama-70B NF4 | GLM-4.7 NF4d+NF2e | Ratio |
|---|---|---|---|
| Layer size | 470 MB | 1237 MB | 2.6× |
| Active params | 1050M | 514M | 0.5× |
| Transfer time (Gen4 NVMe) | 66 ms | 173 ms | 2.6× |
| Compute time @ 1K tokens | 40 ms | 20 ms | 0.5× |
| Bytes per FLOP | 0.075 | 0.401 | 5.4× |

### Streaming configurations: NF4d+NF2e (1237 MB/layer, recommended)

**Token thresholds for 0% streaming overhead (consumer GPUs — NVMe via CPU):**

| GPU | Compute | Res / Str | % str | Gen3 | Gen4 | Gen5 | 2×Gen5 RAID |
|---|---|---|---|---|---|---|---|
| 1× RTX 4090 (24G) | BF16 | 14 / 78 | 85% | 16K | **8K** | 8K | 4K |
| 1× RTX 5090 (32G) | BF16 | 20 / 72 | 78% | >16K | 16K | **8K** | 4K |
| 1× RTX 5090 (32G) | NVFP4 | 20 / 72 | 78% | >32K | 32K | **16K** | 8K |

**Token thresholds (workstation/datacenter GPUs — GDS or CPU pinned):**

| GPU | Compute | Res / Str | % str | Gen5 GDS | RAID GDS | CPU pinned |
|---|---|---|---|---|---|---|
| 1× RTX PRO 6000 (96G) | BF16 | 73 / 19 | 21% | 2K | **1K** | 1K |
| 1× RTX PRO 6000 (96G) | NVFP4 | 73 / 19 | 21% | 4K | **2K** | 2K |
| 1× A100 (80G) | BF16 | 60 / 32 | 35% | 2K | **1K** | 1K |
| 1× H100 (80G) | BF16 | 60 / 32 | 35% | 4K | **2K** | 2K |

The H100 shows higher token requirements than the A100 despite having 80 GB
because its higher compute throughput (330 vs 156 TFLOPS) means it finishes
each layer faster, spending more time waiting for the transfer. Same effect
as NVFP4 — faster compute = harder to hide transfer.

### Streaming configurations: NF4 all (2250 MB/layer)

All-NF4 nearly doubles the layer size, pushing thresholds significantly higher.
This is the main argument for mixed quantization.

**Token thresholds (consumer GPUs):**

| GPU | Compute | Res / Str | % str | Gen3 | Gen4 | Gen5 | 2×Gen5 RAID |
|---|---|---|---|---|---|---|---|
| 1× RTX 4090 (24G) | BF16 | 6 / 86 | 93% | 32K | 16K | 16K | 8K |
| 1× RTX 5090 (32G) | BF16 | 9 / 83 | 90% | 32K | 16K | **16K** | 8K |
| 1× RTX 5090 (32G) | NVFP4 | 9 / 83 | 90% | >32K | >32K | 32K | 16K |

**NF4 vs NF4d+NF2e: the streaming impact**

| GPU + Compute | NF4d+NF2e threshold | NF4 threshold | Increase |
|---|---|---|---|
| RTX 4090 BF16 + Gen4 NVMe | 8K | 16K | 2× |
| RTX 5090 BF16 + Gen5 NVMe | 8K | 16K | 2× |
| RTX 5090 NVFP4 + Gen5 NVMe | 16K | 32K | 2× |
| RTX 5090 NVFP4 + 2×Gen5 RAID | 8K | 16K | 2× |

All-NF4 consistently doubles the token threshold because the layer is 1.82×
bigger while compute stays the same, and fewer layers fit on GPU (raising the
streamed fraction from ~80% to ~90%).

### Why NVFP4 raises thresholds on consumer GPUs

The NVFP4 tensor cores (RTX 5090, Blackwell) provide 2.7× compute throughput
over BF16. This makes each layer complete in ~4ms instead of ~11ms at 4K tokens.
But NVMe transfer time is unchanged — a 1237 MB layer still takes 103ms to
read from a Gen5 drive. The faster compute leaves more idle time waiting for
data:

| RTX 5090 compute mode | Compute @ 8K tokens | Gen5 transfer | Overhead |
|---|---|---|---|
| BF16 (210T) | 21.8 ms/layer | 103 ms | hidden by residency |
| NVFP4 (567T) | 8.1 ms/layer | 103 ms | not fully hidden |

For MoE models specifically, NVFP4 primarily accelerates the dense compute
(attention + shared expert), which are compute-bound large matmuls. The routing
expert matmuls (8 active experts × 1536 intermediate) are smaller and tend
toward memory-bandwidth-bound, limiting the effective speedup to ~2× per
MoE layer rather than the full 2.7×. Using NVFP4 for the dense layers and
standard BF16 for the NF2 expert matmuls is a natural fit.

### Min GPUs: streaming vs all-on-GPU

Streaming reduces the number of GPUs needed. With NF4d+NF2e (1237 MB/layer):

| GPU | Without streaming | With streaming | GPUs saved |
|---|---|---|---|
| RTX 4090 (24G) | 6 GPUs | **1 GPU** | 5 |
| RTX 5090 (32G) | 4 GPUs | **1 GPU** | 3 |
| A100 / H100 (80G) | 2 GPUs | **1 GPU** | 1 |
| RTX PRO 6000 (96G) | 2 GPUs | **1 GPU** | 1 |

---

## Mixed Quantization

For MoE models, the experts dominate layer size (92% of params for GLM-4.7)
but are rarely all active. Using lower-bit quantization for experts while
keeping attention at higher precision gives nearly all the memory savings
with better quality for the always-active components.

### GLM-4.7 layer size by quantization scheme

| Quantization | Dense (7.9%) | Experts (92.1%) | Layer total | Model total |
|---|---|---|---|---|
| All NF4 | 178 MB | 2072 MB | 2250 MB | 202 GB |
| All NF3 | 130 MB | 1510 MB | 1640 MB | 147 GB |
| **NF4 dense + NF2 experts** | **178 MB** | **1059 MB** | **1237 MB** | **111 GB** |
| NF4 dense + NF3 experts | 178 MB | 1510 MB | 1688 MB | 152 GB |
| All NF2 | 91 MB | 1059 MB | 1150 MB | 103 GB |

**NF4 dense + NF2 experts** is only 7% larger than all-NF2 but preserves NF4
quality for the attention layers and shared expert that every token passes
through. This is the recommended mixed quantization for MoE streaming.

### Mixed quantization and NVFP4

Mixed quantization is particularly well-suited for NVFP4 compute on Blackwell:

1. **Dense layers (NF4, 7.9% of params):** Compute-bound, large matmuls.
   NVFP4 tensor cores provide the full 2.7× throughput boost here.
2. **Expert layers (NF2, 92.1% of params):** Bandwidth-bound, small matmuls
   (1536 intermediate per expert). NVFP4 helps less — these are already
   limited by memory bandwidth, not compute.

Using NVFP4 for dense and BF16 for expert matmuls is the natural strategy.
The half-layer size from NF4d+NF2e (1237 MB vs 2250 MB) also halves the
streaming token threshold — critical when NVFP4's fast compute makes
streaming harder to hide.

---

## NVMe Streaming with 32 GB RAM

With 32 GB of system RAM (~24 GB usable), most configurations cannot hold all
streamed weights in CPU pinned memory. The NVMe SSD feeds the pipeline during
every training step.

### Three-stage pipeline with NVMe

```
NVMe ──read──> CPU pinned buffer ──DMA──> GPU double-buffer ──compute──>
                4 rotating slots         2 rotating slots
                (4.8 GB)                 (2.4 GB)
```

The CPU buffer holds only 4 layer slots (not the full streamed weights). This
is a small rotating staging area that keeps all three pipeline stages busy
simultaneously:

| Stage | Activity | Bandwidth |
|---|---|---|
| 1: NVMe → CPU | Read layer i+2 from disk | NVMe BW (3.5–49 GB/s) |
| 2: CPU → GPU | DMA layer i+1 via PCIe | PCIe BW (11–27 GB/s) |
| 3: GPU compute | Process layer i | GPU TFLOPS |

CPU pinned buffer: 4 × 1.21 GB = **4.8 GB** (for NF4d+NF2e). Fits easily in
32 GB RAM with ample headroom for the OS and PyTorch.

### NVMe is usually the bottleneck (consumer GPUs)

On consumer GPUs, the effective bandwidth is `min(NVMe, PCIe)`. Since NVMe is
typically slower than PCIe, it determines the crossover batch size.

**Per-layer transfer time at different effective bandwidths**
(GLM-4.7 NF4d+NF2e, 1237 MB/layer):

| Effective bandwidth | Transfer/layer | Bottleneck |
|---|---|---|
| 3.5 GB/s (1× Gen3 NVMe) | 345 ms | NVMe |
| 7.0 GB/s (1× Gen4 NVMe) | 173 ms | NVMe |
| 11 GB/s (PCIe Gen4 x16 cap) | 112 ms | PCIe |
| 12 GB/s (1× Gen5 NVMe) | 103 ms | NVMe (exceeds Gen4 PCIe) |
| 13 GB/s (2× Gen4 RAID) | 95 ms | PCIe Gen4 caps at 11 |
| 22 GB/s (PCIe Gen5 x16 cap) | 56 ms | PCIe |
| 49 GB/s (5× Gen5 RAID + GDS) | 25 ms | GDS only, not consumer |

### Full crossover matrix: 1× RTX 4090 (24G), NF4d+NF2e, BF16

14 resident + 78 streamed (85% streamed), 160 TFLOPS:

| Storage config | 1K t | 2K t | 4K t | 8K t | 16K t |
|---|---|---|---|---|---|
| 1× Gen3 NVMe (3.5 GB/s) | >10× | +641% | +271% | +85% | **0%** |
| 1× Gen4 NVMe (7 GB/s) | +641% | +271% | +85% | **0%** | 0% |
| 1× Gen5 NVMe (capped at 11 GB/s PCIe) | +354% | +127% | +14% | **0%** | 0% |
| 2× Gen4 RAID (capped at 11 GB/s PCIe) | +354% | +127% | +14% | **0%** | 0% |

### Full crossover matrix: 1× RTX 5090 (32G), NF4d+NF2e, BF16

20 resident + 72 streamed (78% streamed), 210 TFLOPS:

| Storage config | 1K t | 2K t | 4K t | 8K t | 16K t |
|---|---|---|---|---|---|
| 1× Gen4 NVMe (7 GB/s) | +800% | +350% | +125% | +13% | **0%** |
| 1× Gen5 NVMe (12 GB/s) | +435% | +168% | +34% | **0%** | 0% |
| 2× Gen5 RAID (capped at 22 GB/s PCIe) | +194% | +47% | **0%** | 0% | 0% |

### Full crossover matrix: 1× RTX 5090 (32G), NF4d+NF2e, NVFP4

20 resident + 72 streamed (78% streamed), 567 TFLOPS:

| Storage config | 4K t | 8K t | 16K t | 32K t |
|---|---|---|---|---|
| 1× Gen5 NVMe (12 GB/s) | +260% | +80% | **0%** | 0% |
| 2× Gen5 RAID (capped at 22 GB/s PCIe) | +100% | **0%** | 0% | 0% |
| CPU pinned RAM (22 GB/s PCIe) | +100% | **0%** | 0% | 0% |

NVFP4 raises the threshold by 2.7× compared to BF16: Gen5 NVMe goes from
8K → 16K, and 2× Gen5 RAID goes from 4K → 8K.

### Full crossover matrix: 1× RTX PRO 6000 Blackwell (96G), NF4d+NF2e

73 resident + 19 streamed (21% streamed), 210 TFLOPS BF16 / 567 TFLOPS NVFP4:

| Storage config | Compute | 1K t | 2K t | 4K t | 8K t |
|---|---|---|---|---|---|
| 1× Gen5 NVMe GDS (12 GB/s) | BF16 | +5% | **0%** | 0% | 0% |
| 5× Gen5 RAID GDS (49 GB/s) | BF16 | **0%** | 0% | 0% | 0% |
| 1× Gen5 NVMe GDS (12 GB/s) | NVFP4 | +180% | +40% | **0%** | 0% |
| 5× Gen5 RAID GDS (49 GB/s) | NVFP4 | +20% | **0%** | 0% | 0% |

### Full crossover matrix: 1× A100 (80G), NF4d+NF2e, BF16

60 resident + 32 streamed (35% streamed), 156 TFLOPS:

| Storage config | 512t | 1K t | 2K t | 4K t | 8K t |
|---|---|---|---|---|---|
| Gen4 NVMe GDS (7 GB/s) | +493% | +197% | +48% | **0%** | 0% |
| Gen5 NVMe GDS (12 GB/s) | +246% | +73% | **0%** | 0% | 0% |
| RAID GDS (24 GB/s) | +73% | **0%** | 0% | 0% | 0% |
| CPU pinned (22 GB/s) | +73% | **0%** | 0% | 0% | 0% |

---

## RAID Convergence

On consumer GPUs, the NVMe path always goes through CPU memory. The effective
bandwidth is `min(NVMe_aggregate, PCIe_bandwidth)`. Once the RAID aggregate
read speed exceeds the PCIe H2D bandwidth, NVMe streaming becomes equivalent
to CPU pinned RAM — the bottleneck shifts from the drive to the PCIe link.

### RAID size needed to match CPU pinned performance

| GPU | PCIe BW | RAID to match | Result |
|---|---|---|---|
| RTX 4090 (Gen4 x16) | ~11 GB/s | 1× Gen5 NVMe (13 GB/s) | **Matches pinned** |
| RTX 4090 (Gen4 x16) | ~11 GB/s | 2× Gen4 NVMe (14 GB/s) | **Matches pinned** |
| RTX 5090 (Gen5 x16) | ~22 GB/s | 2× Gen5 NVMe (26 GB/s) | **Matches pinned** |
| RTX 5090 (Gen5 x16) | ~22 GB/s | 3× Gen4 NVMe (21 GB/s) | ~Matches pinned |

**Key insight:** A single ~$80 Gen5 NVMe on an RTX 4090 already saturates the
PCIe Gen4 link. At that point, having 128 GB of RAM provides zero bandwidth
advantage over a 32 GB machine with NVMe streaming — the token thresholds
are identical.

On the RTX 5090 with PCIe Gen5, two Gen5 drives in RAID0 are needed to
saturate the link. This is still a very affordable upgrade (~$160).

### When RAID doesn't help (diminishing returns)

Once aggregate NVMe ≥ PCIe, adding more drives provides no additional
bandwidth. The PCIe link is the ceiling on consumer GPUs:

| RTX 4090 config | Effective BW | Improvement over 1× Gen4 |
|---|---|---|
| 1× Gen4 NVMe | 7 GB/s | baseline |
| 1× Gen5 NVMe | 11 GB/s (PCIe cap) | 1.6× |
| 2× Gen5 RAID | 11 GB/s (PCIe cap) | 1.6× (no gain over 1×) |
| 4× Gen5 RAID | 11 GB/s (PCIe cap) | 1.6× (no gain over 1×) |

On workstation/datacenter GPUs with GDS, there is no PCIe cap because
the NVMe reads bypass the CPU entirely. Adding more RAID drives continues
to increase bandwidth up to the GPU's PCIe link capacity or RAID controller
limits.

---

## Multi-GPU Pipeline Parallelism

With pipeline parallelism, each GPU handles a subset of layers. This reduces
both the resident weight footprint and the number of layers to stream per GPU.

### GLM-4.7 NF4d+NF2e: all-on-GPU thresholds

When enough GPUs are used, the model fits entirely in VRAM with no streaming:

| GPU | Min GPUs (all on GPU) | Free VRAM | Notes |
|---|---|---|---|
| RTX 4090 (24G) | 6 | 3.2 GB | Tight |
| RTX 4090 (24G) | 8 | 8.1 GB | Comfortable |
| RTX 5090 (32G) | 5 | ~3 GB | Tight |
| RTX 5090 (32G) | 6 | 11.2 GB | Comfortable |
| RTX PRO 6000 (96G) | 2 | 37.3 GB | Generous |
| A100 / H100 (80G) | 2 | 21.3 GB | Comfortable |

### Streaming with pipeline parallelism

When fewer GPUs are available, streaming fills the gap:

| Config | Compute | Layers/GPU | Resident | Streamed | Gen4 NVMe 0% at |
|---|---|---|---|---|---|
| 2× RTX 4090 | BF16 | 46 | 15 | 31 (67%) | 8K |
| 4× RTX 4090 | BF16 | 23 | 15 | 8 (35%) | 4K |
| 2× RTX 5090 | BF16 | 46 | 21 | 25 (54%) | 8K |
| 2× RTX 5090 | NVFP4 | 46 | 21 | 25 (54%) | 16K |

Pipeline parallelism helps in two ways:
1. Fewer layers per GPU → more can be resident
2. Lower streamed fraction → lower batch size threshold

---

## Batch Size and Token Counts

The "total tokens" in all tables refers to `batch_size × sequence_length`.
The GPU performs the same FLOPs regardless of how tokens are arranged:

| Batch size | Seq length | Total tokens | Equivalent |
|---|---|---|---|
| 1 | 8192 | 8192 | Same compute |
| 8 | 1024 | 8192 | Same compute |
| 4 | 2048 | 8192 | Same compute |
| 16 | 512 | 8192 | Same compute |

### Practical configurations for GLM-4.7 on 1× RTX 4090 (BF16)

The 8K token threshold with Gen4 NVMe can be reached many ways:

| Scenario | Batch | Seq len | Total | Use case |
|---|---|---|---|---|
| Long context | 1 | 8192 | 8192 | Document fine-tuning |
| Standard SFT | 8 | 1024 | 8192 | Instruction tuning |
| Short-context SFT | 16 | 512 | 8192 | Chat fine-tuning |
| Multi-turn dialog | 4 | 2048 | 8192 | Conversation tuning |

### Practical configurations for GLM-4.7 on 1× RTX 5090 (NVFP4)

The 16K token threshold (Gen5 NVMe) or 8K threshold (2×Gen5 RAID):

| Scenario | Batch | Seq len | Total | Use case |
|---|---|---|---|---|
| Standard SFT (RAID) | 8 | 1024 | 8192 | Instruction tuning |
| Standard SFT (single NVMe) | 16 | 1024 | 16384 | Instruction tuning |
| Long context (single NVMe) | 4 | 4096 | 16384 | Document fine-tuning |

### Activation memory with gradient checkpointing

With gradient checkpointing, only one layer's activations are in VRAM at a
time. For B=8, S=1024, H=4096:

```
Hidden states: 8 × 1024 × 4096 × 2 bytes = 64 MB per checkpoint
```

With chunked flash attention and chunked MLP, intermediate tensors are further
bounded. The ~2–3 GB of free VRAM on an RTX 4090 is sufficient for these
batch sizes.

---

## Hardware Recommendations

### Dense models (Llama-70B, Qwen-72B)

| Budget | Hardware | Quant | Streaming? | Min tokens |
|---|---|---|---|---|
| $1,600 | 1× RTX 4090 + Gen4 NVMe | NF4 | Yes (49% streamed) | 1024 |
| $1,600 | 1× RTX 4090 + Gen4 NVMe | NF3 | Yes (29% streamed) | 512 |
| $2,000 | 1× RTX 5090 | NF3 | No (all on GPU) | — |
| $3,200 | 2× RTX 4090 | NF4 | No (all on GPU) | — |

Dense models are easy to stream. A single RTX 4090 is sufficient.

### MoE models (GLM-4.7 355B)

**Consumer GPUs:**

| Budget | Hardware | Quant | Compute | Streaming | Min tokens |
|---|---|---|---|---|---|
| $1,800 | 1× RTX 4090 + Gen4 NVMe | NF4d+NF2e | BF16 | 85% | **8K** |
| $1,880 | 1× RTX 4090 + Gen5 NVMe | NF4d+NF2e | BF16 | 85% | **8K** |
| $2,200 | 1× RTX 5090 + Gen5 NVMe | NF4d+NF2e | BF16 | 78% | **8K** |
| $2,200 | 1× RTX 5090 + Gen5 NVMe | NF4d+NF2e | NVFP4 | 78% | **16K** |
| $2,360 | 1× RTX 5090 + 2× Gen5 RAID | NF4d+NF2e | NVFP4 | 78% | **8K** |
| $9,600 | 6× RTX 4090 | NF4d+NF2e | BF16 | None | — |

**Workstation/Datacenter GPUs (GDS available):**

| Budget | Hardware | Quant | Compute | Streaming | Min tokens |
|---|---|---|---|---|---|
| ~$7,000 | 1× RTX PRO 6000 + NVMe RAID | NF4d+NF2e | BF16 | 21% | **1K** |
| ~$7,000 | 1× RTX PRO 6000 + NVMe RAID | NF4d+NF2e | NVFP4 | 21% | **2K** |
| ~$15,000 | 2× RTX PRO 6000 | NF4d+NF2e | any | None | — |
| ~$25,000 | 1× A100 + Gen5 NVMe | NF4d+NF2e | BF16 | 35% | **2K** |

For consumer hardware, the RTX 4090 at 8K tokens (batch 8 × 1024) remains
the most accessible path. The RTX 5090 with NVFP4 is faster per step but
needs 16K tokens (or a second Gen5 NVMe for RAID to bring it back to 8K).

### NVMe selection guide

| GPU config | Min NVMe | Ideal NVMe | Notes |
|---|---|---|---|
| RTX 4090 (85% streamed) | Gen4 | Gen5 (saturates PCIe) | Single Gen5 = max BW |
| RTX 5090 BF16 (78% streamed) | Gen5 | Gen5 | Single drive sufficient |
| RTX 5090 NVFP4 (78% streamed) | Gen5 | 2× Gen5 RAID | RAID halves token req |
| RTX PRO 6000 (21% streamed) | Gen4 | NVMe RAID + GDS | GDS unlocks full BW |
| A100 (35% streamed) | Gen4 | Gen5 + GDS | GDS bypasses CPU |

Higher GPU residency makes you more tolerant of slow NVMe. The RTX PRO 6000
keeps 79% resident, so even a Gen4 NVMe works at low token counts.

---

## Implementation Notes

### Critical for correct overlap

1. **Pinned memory**: CPU buffers must use `pin_memory=True`. Pageable memory
   drops bandwidth from 11 GB/s to 7 GB/s and prevents true async DMA.

2. **Pre-allocated output buffers**: Use `torch.mm(A, B, out=C)` instead of
   `C = torch.mm(A, B)`. Temporary allocations cause implicit CUDA
   synchronizations that serialize the pipeline.

3. **Dedicated copy stream**: Use a separate `torch.cuda.Stream()` for H2D
   transfers. The default stream serializes all operations.

4. **Stream synchronization**: Call
   `torch.cuda.current_stream().wait_stream(copy_stream)` before accessing
   the transferred data.

### NVMe triple-buffering implementation

For NVMe streaming (32 GB RAM path), the CPU side needs its own buffering:

```python
# 4 CPU pinned slots (rotating)
cpu_slots = [torch.empty(layer_size, pin_memory=True) for _ in range(4)]

# Async NVMe read (via io_uring or mmap + madvise)
# Slot 0,1: being read from NVMe
# Slot 2,3: being DMA'd to GPU
```

The NVMe reads should use `O_DIRECT` or `io_uring` for maximum bandwidth.
Standard `read()` syscalls go through the page cache, which wastes memory and
adds copies. For sequential reads of 1+ GB per layer, direct I/O achieves
near-theoretical NVMe bandwidth.

### GPUDirect Storage (GDS) path

On workstation GPUs (RTX PRO / Quadro / Data Center), kvikio enables NVMe → GPU
transfers that bypass CPU memory entirely. This eliminates the CPU bounce buffer
and allows the GPU to read directly from the NVMe controller via PCIe P2P DMA.

**Requirements:**
- Workstation or Data Center GPU (not GeForce — GeForce falls back to compat mode)
- `pip install kvikio-cu12`
- RAID0 requires parallel IO: `KVIKIO_NTHREADS=16 KVIKIO_TASK_SIZE=1048576`
- Kernel ≥ 6.2 for native PCI P2PDMA (no nvidia-fs module needed with CUDA 12.8+)

**Measured results** (RTX PRO 6000 Blackwell + 5× WD SN8100 Gen5 RAID0):
- Raw bandwidth: 49 GB/s (vs 52.5 GB/s OS-level ceiling)
- 1237 MB MoE layer: 24.7ms per read
- Pipeline overhead: +7.9% at 8K tokens, ~0% at 10K+ tokens
- 4-14× faster than the traditional mmap → pinned → GPU path

**Consumer GPU fallback:** kvikio works on GeForce GPUs in compatibility mode
(POSIX pread + CPU bounce buffer). Measured at 3.3 GB/s on RTX 4090 — same as
the traditional mmap path, no improvement. Use the CPU pinned RAM path instead.

See `GDS_BENCHMARK.md` for full benchmark results.

### LoRA placement for MoE models

For MoE models, LoRA adapters go on the **dense** components only:

- Attention projections (q, k, v, o) — always
- Shared expert (gate, up, down) — recommended
- Routing experts — **never** (160 × 3 = 480 projections per layer)

This keeps LoRA memory at 0.4–0.7 GB for the full model (92 layers, r=64),
with optimizer states adding another 1.5–2.7 GB.

### What doesn't work

- **Pageable (non-pinned) CPU memory** — halves bandwidth, prevents async
  overlap
- **Single CUDA stream** — serializes compute and transfer
- **torch.mm() without `out=`** — causes CUDA allocator syncs
- **Standard file I/O without O_DIRECT** — page cache overhead on large
  sequential reads
- **Full-model NF4 for MoE** — 2.25 GB/layer with 12.5% utilization;
  use mixed NF4d+NF2e instead
- **kvikio with default thread count on RAID** — caps at single-drive
  bandwidth; must set KVIKIO_NTHREADS=16
- **GDS on GeForce GPUs** — falls back to compat mode, no P2P DMA
