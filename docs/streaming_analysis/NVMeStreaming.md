# NVMe Weight Streaming: Partial-Resident QLoRA Training

Training 70B–355B models on consumer GPUs by keeping a fraction of quantized
weights on-GPU and streaming the rest from CPU pinned memory or NVMe.

## Key Results

- **Llama-70B NF4 on 1× RTX 4090**: 49% resident on GPU, 51% streamed.
  Zero overhead at 1024 tokens (Gen4 NVMe + 32 GB RAM). A single $1600 GPU
  fine-tunes a 70B model.
- **GLM-4.7 355B MoE on 1× RTX 4090**: 15% resident, 85% streamed.
  Zero overhead at 8192 tokens (batch 8 × 1024 seq). A 355B model on one
  consumer GPU with a Gen4 NVMe and 32 GB RAM.
- **GLM-4.7 355B on 1× A100**: 65% resident, 35% streamed.
  Zero overhead at 2048 tokens with Gen5 NVMe.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Memory Hierarchy and DMA](#memory-hierarchy-and-dma)
3. [Partial-Resident Streaming](#partial-resident-streaming)
4. [Theoretical Model](#theoretical-model)
5. [Dense Models: Llama-70B](#dense-models-llama-70b)
6. [MoE Models: GLM-4.7 355B](#moe-models-glm-47-355b)
7. [Mixed Quantization](#mixed-quantization)
8. [NVMe Streaming with 32 GB RAM](#nvme-streaming-with-32-gb-ram)
9. [Multi-GPU Pipeline Parallelism](#multi-gpu-pipeline-parallelism)
10. [Batch Size and Token Counts](#batch-size-and-token-counts)
11. [Hardware Recommendations](#hardware-recommendations)
12. [Implementation Notes](#implementation-notes)

---

## Architecture Overview

QLoRA freezes base model weights and trains only low-rank adapters. The frozen
weights are read-only during forward and backward passes, making them candidates
for streaming from slower storage tiers.

### Three-stage pipeline

```
NVMe SSD ──(3.5–28 GB/s)──> CPU pinned buffer ──(11–44 GB/s PCIe DMA)──> GPU
  cold storage               4 rotating layer slots                  compute + LoRA
```

The pipeline has three concurrent stages:

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
| Resident quantized weights | GPU VRAM | 60–73 layers (varies by GPU) |
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
~50 GB/s, DDR5: ~80 GB/s) far exceeds PCIe bandwidth (11–44 GB/s), DRAM is
never the bottleneck.

With **regular (pageable) memory**, the CUDA runtime must:
1. Check if the page is in physical RAM (not swapped)
2. Copy data through an internal pinned staging buffer
3. DMA from the staging buffer to GPU

This halves effective bandwidth (measured: 7 GB/s vs 11 GB/s on Gen3) and
prevents true async overlap.

### Bandwidth hierarchy

| Link | Bandwidth | Notes |
|---|---|---|
| DDR4 dual-channel | ~50 GB/s | Never the bottleneck |
| DDR5 dual-channel | ~80 GB/s | Never the bottleneck |
| PCIe Gen3 x16 | 11 GB/s (measured) | 85% of theoretical 13 GB/s |
| PCIe Gen4 x16 | ~22 GB/s | 2× Gen3 |
| PCIe Gen5 x16 | ~27 GB/s (measured H2D) | Blackwell workstation |
| Gen3 NVMe (e.g., 970 EVO) | 3.5 GB/s | Sequential read |
| Gen4 NVMe (e.g., 980 PRO) | 7 GB/s | Sequential read |
| Gen5 NVMe (e.g., SN8100) | 13 GB/s (measured) | Sequential read, sustained |
| 5× Gen5 NVMe RAID-0 | 49 GB/s (measured) | Via kvikio with 16 IO threads |
| 5× Gen5 NVMe RAID-0 (fio) | 52.5 GB/s (measured) | Raw OS-level ceiling |

### When NVMe is in the loop

NVMe bandwidth is **irrelevant during training** if all streamed weights fit in
CPU RAM. In that case, NVMe reads once at startup, and the training loop is
purely PCIe DMA from pinned DRAM.

NVMe bandwidth **matters during training** only when CPU RAM is too small to
hold all streamed weights. With 32 GB RAM, the usable portion (~24 GB after
OS and PyTorch) often cannot hold 40–95 GB of streamed weights. In this case,
the pipeline reads from NVMe every step, and the effective bandwidth is:

```
effective_bandwidth = min(NVMe_read_bandwidth, PCIe_bandwidth)
```

With triple-buffering on the CPU side, NVMe reads and PCIe DMA overlap, but
throughput is still capped by the slower link.

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
| 79% (RTX 6000P + GLM-4.7) | 0.21 | compute ≥ 0.21 × transfer |

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

| GPU | Resident / Streamed | Gen3 NVMe | Gen4 NVMe | Gen5 NVMe |
|---|---|---|---|---|
| 1× RTX 4090 (24G), NF4 | 41 / 39 (49%) | 0% @ 2048t | **0% @ 1024t** | 0% @ 512t |
| 1× RTX 4090 (24G), NF3 | 57 / 23 (29%) | 0% @ 1024t | **0% @ 512t** | 0% @ 256t |
| 1× RTX 5090 (32G), NF4 | 58 / 22 (28%) | 0% @ 2048t | 0% @ 1024t | 0% @ 512t |
| 1× RTX 5090 (32G), NF3 | ALL ON GPU | — | — | — |
| 1× A100/H100 (80G) | ALL ON GPU | — | — | — |
| 1× RTX 6000P (96G) | ALL ON GPU | — | — | — |

Dense models are the streaming sweet spot. A single RTX 4090 with a Gen4
NVMe reaches zero overhead at just 1024 tokens — that's batch=1 with 1K
context. The compute-to-transfer ratio is favorable because every byte
transferred contributes to active computation.

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

### Streaming configurations (32 GB RAM)

The resident fraction varies by GPU — more VRAM means more layers stay on-GPU,
and fewer tokens are needed for zero overhead.

**NVMe crossover: total tokens for 0% streaming overhead**

| GPU | Res / Str | % streamed | Gen3 NVMe | Gen4 NVMe | Gen5 NVMe | 2×Gen5 RAID |
|---|---|---|---|---|---|---|
| 1× RTX 4090 (24G) | 14 / 78 | 85% | 16K | 8K | 8K | 4K |
| 2× RTX 4090 (24G) | 15 / 31 | 67% | 16K | 8K | 4K | 2K |
| 4× RTX 4090 (24G) | 15 / 8 | 35% | 8K | 4K | 2K | 1K |
| 1× RTX 5090 (32G) | 20 / 72 | 78% | >16K | 16K | 8K | 4K |
| 1× RTX 6000P (96G) | 73 / 19 | 21% | 4K | 2K | 2K | 1K |
| 1× A100 (80G) | 60 / 32 | 35% | 8K | 4K | 2K | 1K |
| 1× H100 (80G) | 60 / 32 | 35% | 16K | 8K | 4K | 2K |

The H100 shows higher token requirements despite having 80 GB because its
higher compute throughput (330 vs 156 TFLOPS) means it finishes each layer
faster, spending more time waiting for the transfer. Faster compute + same
transfer = more idle time.

### Min GPUs: streaming vs all-on-GPU

Streaming reduces the number of GPUs needed. With NF4d+NF2e (1237 MB/layer):

| GPU | Without streaming | With streaming | GPUs saved |
|---|---|---|---|
| RTX 4090 (24G) | 6 GPUs | **1 GPU** | 5 |
| RTX 5090 (32G) | 4 GPUs | **1 GPU** | 3 |
| A100 / H100 (80G) | 2 GPUs | **1 GPU** | 1 |
| RTX 6000 Pro (96G) | 2 GPUs | **1 GPU** | 1 |

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
| 1: NVMe → CPU | Read layer i+2 from disk | NVMe BW (3.5–24 GB/s) |
| 2: CPU → GPU | DMA layer i+1 via PCIe | PCIe BW (11–44 GB/s) |
| 3: GPU compute | Process layer i | GPU TFLOPS |

CPU pinned buffer: 4 × 1.21 GB = **4.8 GB** (for NF4d+NF2e). Fits easily in
32 GB RAM with ample headroom for the OS and PyTorch.

### NVMe is usually the bottleneck

With 32 GB RAM, the effective bandwidth is `min(NVMe, PCIe)`. Since NVMe is
typically slower than PCIe, it determines the crossover batch size.

**Per-layer transfer time at different effective bandwidths**
(GLM-4.7 NF4d+NF2e, 1237 MB/layer):

| Effective bandwidth | Transfer/layer | Bottleneck |
|---|---|---|
| 3.5 GB/s (Gen3 NVMe, any PCIe) | 345 ms | NVMe |
| 7.0 GB/s (Gen4 NVMe, any PCIe) | 173 ms | NVMe |
| 12 GB/s (Gen5 NVMe, Gen4+ PCIe) | 101 ms | NVMe |
| 14 GB/s (2× Gen4 RAID, Gen4+ PCIe) | 86 ms | NVMe |
| 22 GB/s (2× Gen5 RAID, Gen4 PCIe) | 55 ms | PCIe |
| 24 GB/s (2× Gen5 RAID, Gen5 PCIe) | 50 ms | NVMe |

### Full crossover matrix: 1× A100 (80G), NF4d+NF2e

60 resident + 32 streamed (35% streamed), 156 TFLOPS:

| Storage config | 512t | 1K t | 2K t | 4K t | 8K t |
|---|---|---|---|---|---|
| Gen3 NVMe + Gen3 PCIe (3.5 GB/s) | +1086% | +493% | +197% | +48% | **0%** |
| Gen4 NVMe + Gen4 PCIe (7 GB/s) | +493% | +197% | +48% | **0%** | 0% |
| Gen5 NVMe + Gen4 PCIe (12 GB/s) | +246% | +73% | **0%** | 0% | 0% |
| 2× Gen4 RAID + Gen4 PCIe (14 GB/s) | +197% | +48% | **0%** | 0% | 0% |
| 2× Gen5 RAID + Gen5 PCIe (24 GB/s) | +73% | **0%** | 0% | 0% | 0% |

### Full crossover matrix: 1× RTX 4090 (24G), NF4d+NF2e

14 resident + 78 streamed (85% streamed), 160 TFLOPS:

| Storage config | 512t | 1K t | 2K t | 4K t | 8K t | 16K t |
|---|---|---|---|---|---|---|
| Gen3 NVMe + Gen3 PCIe (3.5 GB/s) | >10× | >10× | +641% | +271% | +85% | **0%** |
| Gen4 NVMe + Gen4 PCIe (7 GB/s) | >10× | +641% | +271% | +85% | **0%** | 0% |
| Gen5 NVMe + Gen4 PCIe (12 GB/s) | +765% | +332% | +116% | +8% | **0%** | 0% |
| 2× Gen4 RAID + Gen4 PCIe (14 GB/s) | +641% | +271% | +85% | **0%** | 0% | 0% |
| 2× Gen5 RAID + Gen5 PCIe (24 GB/s) | +332% | +116% | +8% | **0%** | 0% | 0% |

### Full crossover matrix: 1× RTX 6000P (96G), NF4d+NF2e

73 resident + 19 streamed (21% streamed), 160 TFLOPS:

| Storage config | 512t | 1K t | 2K t | 4K t | 8K t |
|---|---|---|---|---|---|
| Gen3 NVMe + Gen3 PCIe (3.5 GB/s) | +622% | +261% | +81% | **0%** | 0% |
| Gen4 NVMe + Gen4 PCIe (7 GB/s) | +261% | +81% | **0%** | 0% | 0% |
| Gen5 NVMe + Gen4 PCIe (12 GB/s) | +111% | +5% | **0%** | 0% | 0% |
| 2× Gen4 RAID + Gen4 PCIe (14 GB/s) | +81% | **0%** | 0% | 0% | 0% |
| 2× Gen5 RAID + Gen5 PCIe (24 GB/s) | +5% | **0%** | 0% | 0% | 0% |

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
| RTX 6000P (96G) | 2 | 37.3 GB | Generous |
| A100 / H100 (80G) | 2 | 21.3 GB | Comfortable |

### Streaming with pipeline parallelism

When fewer GPUs are available, streaming fills the gap:

| Config | Layers/GPU | Resident | Streamed | Gen4 NVMe 0% at |
|---|---|---|---|---|
| 2× RTX 4090 | 46 | 15 | 31 (67%) | 8192 tokens |
| 4× RTX 4090 | 23 | 15 | 8 (35%) | 4096 tokens |
| 2× RTX 5090 | 46 | 21 | 25 (54%) | 8192 tokens |

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

### Practical configurations for GLM-4.7 on 1× RTX 4090

The 8192 token threshold with Gen4 NVMe can be reached many ways:

| Scenario | Batch | Seq len | Total | Use case |
|---|---|---|---|---|
| Long context | 1 | 8192 | 8192 | Document fine-tuning |
| Standard SFT | 8 | 1024 | 8192 | Instruction tuning |
| Short-context SFT | 16 | 512 | 8192 | Chat fine-tuning |
| Multi-turn dialog | 4 | 2048 | 8192 | Conversation tuning |

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

| Budget | Hardware | Quant | Streaming? | Min tokens |
|---|---|---|---|---|
| $1,800 | 1× RTX 4090 + Gen4 NVMe | NF4d+NF2e | Yes (85%) | 8192 |
| $3,600 | 2× RTX 4090 + Gen4 NVMe | NF4d+NF2e | Yes (67%) | 8192 |
| $7,200 | 4× RTX 4090 + Gen4 NVMe | NF4d+NF2e | Yes (35%) | 4096 |
| $9,600 | 6× RTX 4090 | NF4d+NF2e | No (all on GPU) | — |
| ~$7,000 | 1× RTX 6000P + Gen4 NVMe | NF4d+NF2e | Yes (21%) | 2048 |
| ~$15,000 | 2× RTX 6000P | NF4d+NF2e | No (all on GPU) | — |
| ~$25,000 | 1× A100 + Gen5 NVMe | NF4d+NF2e | Yes (35%) | 2048 |

For consumer hardware, the single RTX 4090 at 8K tokens (batch 8 × 1024) is
the most accessible path to fine-tuning a 355B parameter model.

### NVMe selection guide

| GPU config | Min NVMe for practical use | Ideal NVMe |
|---|---|---|
| 1× RTX 6000P (21% streamed) | Gen3 (3.5 GB/s) | Gen4 (7 GB/s) |
| 1× A100 (35% streamed) | Gen4 (7 GB/s) | Gen5 (12 GB/s) |
| 4× RTX 4090 (35% streamed) | Gen4 (7 GB/s) | Gen5 (12 GB/s) |
| 1× RTX 4090 (85% streamed) | Gen4 (7 GB/s) | 2× Gen4 RAID (14 GB/s) |

Higher GPU residency makes you more tolerant of slow NVMe. The RTX 6000P
keeps 79% resident, so even a Gen3 NVMe works at 4K tokens.

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

### LoRA placement for MoE models

For MoE models, LoRA adapters go on the **dense** components only:

- Attention projections (q, k, v, o) — always
- Shared expert (gate, up, down) — recommended
- Routing experts — **never** (160 × 3 = 480 projections per layer)

This keeps LoRA memory at 0.4–0.7 GB for the full model (92 layers, r=64),
with optimizer states adding another 1.5–2.7 GB.

### GPUDirect Storage (GDS) path

On workstation GPUs (RTX PRO / Quadro / Data Center), kvikio enables NVMe → GPU
transfers that bypass CPU memory entirely. This eliminates the CPU bounce buffer
and allows the GPU to read directly from the NVMe controller via PCIe P2P DMA.

**Requirements:**
- Workstation or Data Center GPU (not GeForce)
- `pip install kvikio-cu12`
- RAID0 requires parallel IO: `KVIKIO_NTHREADS=16 KVIKIO_TASK_SIZE=1048576`

**Measured results** (RTX PRO 6000 Blackwell + 5× WD SN8100 Gen5 RAID0):
- Raw bandwidth: 49 GB/s (vs 52.5 GB/s OS-level ceiling)
- 1237 MB MoE layer: 24.7ms per read
- Pipeline overhead: +7.9% at 8K tokens, ~0% at 10K+ tokens
- 4-14× faster than the traditional mmap → pinned → GPU path

See `docs/streaming_analysis/GDS_BENCHMARK.md` for full benchmark results.

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
