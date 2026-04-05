# bitsandbytes 2: Efficient Agent Training and Inference

## What is bitsandbytes 2?

bitsandbytes 2 is a GPU software library that makes it possible to train and run very large AI models on hardware that would normally be far too small. A model like GLM-4.7 has 355 billion parameters and would normally require a rack of expensive datacenter GPUs. bitsandbytes 2 fits it on a single consumer graphics card — an NVIDIA RTX 5090 with 32 GB of memory — by compressing the model's weights, streaming them from disk, and optimizing every layer of the computation stack.

The library is the successor to the original bitsandbytes, which introduced QLoRA (4-bit quantization for fine-tuning) and 8-bit optimizers. bitsandbytes 2 generalizes these ideas to 2-5 bit quantization, adds custom CUDA kernels for fast inference, and introduces NVMe weight streaming for training models that far exceed GPU memory.

---

## The problem: agent training breaks the entire stack

Training AI coding agents is fundamentally different from training conventional models:

| | Regular training | Agent training |
|---|---|---|
| Context length | 2K-8K tokens | 50K-258K tokens |
| Activation memory | Modest | Exceeds 32 GB alone |
| Fits on one GPU? | Usually | Never (without bitsandbytes 2) |
| Batch size | Standard | Often just one long trajectory |

At 256K context on a 355B-parameter model, the intermediate computations (activations) alone exceed 32 GB. The frozen model weights are hundreds of gigabytes even when compressed. Optimizer states, gradients, and adapter weights all compete for the same memory. Training compute scales quadratically with context length because attention — the mechanism that lets the model look at all prior tokens — grows as O(n^2). A 355B model at 256K context requires roughly 143 PetaFLOPs per training step.

bitsandbytes 2 solves this with three interlocking systems: quantization, NVMe weight streaming, and a training optimization stack.

---

## Feature 1: Block Normal Float (BNF) quantization

### The idea

Neural network weights follow an approximately normal (bell-curve) distribution. BNF exploits this by choosing quantization levels so that each level covers an equal slice of probability mass under the bell curve. This is information-theoretically optimal — it squeezes the maximum information into each bit.

The original QLoRA paper introduced NF4 (4-bit Normal Float). bitsandbytes 2 generalizes this to arbitrary bit widths: k = 2, 3, 4, and 5 bits per weight, as well as fractional widths like 2.66-bit (for MoE experts) and 3.33-bit (for dense layers).

### Why this matters

- **No calibration data required.** Methods like GPTQ and AWQ need a representative dataset to calibrate quantization ranges. If the deployment data differs from the calibration data, accuracy degrades (out-of-distribution bias). BNF is purely statistical — it assumes only that weights are approximately normally distributed, which is true across virtually all modern models.
- **Deterministic and reproducible.** The same model always quantizes to the same result.
- **Matches or exceeds GPTQ at 4-bit** while being simpler and faster to apply.

### Hadamard rotations

At very low bit widths (2-3 bits), outlier weights cause problems: a single large value wastes an entire quantization level. Hadamard rotation is a mathematical transformation that spreads outlier energy evenly across all weights in a block, making the distribution more Gaussian and improving quantization accuracy.

The key insight is that Hadamard rotation is orthogonal — if you rotate both the weights and the input activations, the final result is unchanged: H(A) x H(B)^T = A x B^T. This means existing inference kernels need no modification. Weights are rotated once offline; activations are rotated per forward pass with a dedicated kernel that adds less than 1 microsecond of overhead at typical batch sizes.

### NVFP4 on Blackwell (B200 / RTX 5090)

NVIDIA's Blackwell architecture (RTX 5090, B200) includes native FP4 tensor cores — hardware units that perform 4-bit floating-point matrix multiplications directly. bitsandbytes 2 targets these via the NVFP4 format, achieving 970-1,583 TFLOPS on a single RTX 5090.

The Hadamard rotation is fused into the NVFP4 quantization kernel at zero additional cost — the rotation matrix is applied as one operand of a CUTLASS GEMM, which is already being executed.

---

## Feature 2: Custom CUDA inference kernels

### The 4-kernel strategy

Different batch sizes need different GPU strategies. bitsandbytes 2 uses four specialized CUDA kernels, automatically dispatched based on the current workload:

| Kernel | Batch size (M) | Use case | How it works |
|---|---|---|---|
| Scalar GEMV | 1-4 | Autoregressive decode (token-by-token generation) | 64 threads per warp, warp-shuffle codebook lookup, no tensor cores. Avoids 94% waste that tensor cores have at M=1. |
| MMA dequant | 5-16 | Small batch inference | Tensor core m16n8k16 with inline dequantization via async copy pipeline. |
| Dequant + cuBLAS | 17+ | Large batch / prefill | Separate dequant kernel writes FP16, then cuBLAS handles the matrix multiply at full efficiency. |
| Grouped MMA | 1-16 | MoE expert layers | Same as MMA but batches all active experts in one kernel launch. |

All four kernels read a tiled packed format (via `repack_kbit`) with E4M4 absmax scaling factors (1 byte per quantization block, down from 4 bytes in the original bitsandbytes).

### Inference benchmarks: RTX 4090 (Ada Lovelace)

Per-kernel timings measured with NVIDIA Nsight Compute (NCU) on an RTX 4090, using GLM-4.7 layer shapes.

**Single-token decode (M=1) — the dominant workload for code assistants:**

| Layer | k=2 | k=3 | k=4 | k=5 | FP16 | Speedup vs FP16 |
|---|---|---|---|---|---|---|
| gate/up (2048x5120) | 9.5 us | 10.8 us | 13.0 us | 14.4 us | 19.1 us | 1.47-2.00x |
| down (5120x2048) | 10.2 us | 11.6 us | 13.1 us | 14.4 us | 19.1 us | 1.32-1.87x |
| O projection (4096x2048) | 8.0 us | 9.1 us | 10.2 us | 11.1 us | 15.8 us | 1.42-1.99x |
| KV projection (2048x512) | 3.5 us | 3.7 us | 4.3 us | 4.1 us | 10.9 us | 2.56-3.11x |
| MoE gate/up (8 experts) | 9.0 us | 10.2 us | 11.3 us | 12.7 us | 11.7 us | 0.92-1.30x |
| MoE down (8 experts) | 8.9 us | 10.7 us | 12.1 us | 13.1 us | 13.1 us | 1.00-1.47x |

Dense layers see large speedups because the scalar GEMV reads 2-5x less data from memory. MoE layers are roughly at parity because individual expert matrices are small.

**Per-block totals (all 7 layer types summed):**

| Quantization | kbit time (us) | FP16 time (us) | Speedup |
|---|---|---|---|
| k=2 (2-bit) | 57.8 | 100.4 | **1.74x faster** |
| k=3 (3-bit) | 65.7 | 100.4 | **1.53x faster** |
| k=4 (4-bit) | 75.1 | 100.4 | **1.34x faster** |
| k=5 (5-bit) | 82.3 | 100.4 | **1.22x faster** |

**Workload-weighted performance (k=4, real Claude Code session distribution):**

| Concurrent users | Dominant kernel | Speed vs FP16 |
|---|---|---|
| 1 | Scalar GEMV (87%) | **43% faster** |
| 4 | Scalar + dq+cuBLAS | **24% faster** |
| 8 | MMA + dq+cuBLAS | **15% faster** |
| 16 | dq+cuBLAS (76%) | Break-even |
| 32+ | dq+cuBLAS (93%+) | FP16 wins |

The crossover is at ~16 concurrent users. For single-user and small-team use (the vast majority of code assistant deployments), kbit quantization is strictly faster than FP16 while using 4-8x less memory.

### MoE benchmarks: NVFP4 on Blackwell (B200)

On NVIDIA B200 datacenter GPUs with native FP4 tensor cores, the NVFP4 MoE pipeline achieves even larger speedups over BF16:

| Configuration | BF16 (ms) | NVFP4 (ms) | Speedup |
|---|---|---|---|
| 8 experts x 8 tokens (gate/up) | 0.501 | 0.267 | **1.87x** |
| 8 experts x 32 tokens (gate/up) | 0.533 | 0.321 | **1.66x** |
| 8 experts x 8 tokens (down) | 0.546 | 0.254 | **2.15x** |
| 8 experts x 32 tokens (down) | 0.588 | 0.271 | **2.17x** |
| 8 experts x 128 tokens (down) | 0.599 | 0.356 | **1.68x** |

Peak throughput reaches 322.8 TFLOPS on a single B200. The NVFP4 pipeline wins in every configuration tested, from 1.07x to 2.17x over BF16.

### Memory savings

Regardless of speed, quantization provides substantial memory compression:

| k (bits) | Compression vs FP16 | 70B model size | 355B model size |
|---|---|---|---|
| 2-bit | 8.0x smaller | ~17.5 GB | ~89 GB |
| 3-bit | 5.3x smaller | ~26.2 GB | ~133 GB |
| 4-bit | 4.0x smaller | ~35.0 GB | ~178 GB |
| FP16 | baseline | ~140 GB | ~710 GB |

At 2-bit, an entire 70B model fits in the 24 GB of a single RTX 4090.

---

## Feature 3: NVMe weight streaming

### The idea

During QLoRA training, the model's base weights are frozen — they are never modified, only read. This means they can be stored on slow storage (NVMe SSD or CPU RAM) and streamed to the GPU layer by layer, rather than occupying permanent GPU memory.

bitsandbytes 2 implements a double-buffered pipeline: while the GPU computes on layer N, the next layer's weights are being prefetched from storage into a second buffer. The GPU never waits for data as long as the compute time per layer exceeds the transfer time.

### Zero overhead at agent-length contexts

The critical insight is that agent training has long compute per layer (because of the long context), which means the streaming transfer is completely hidden behind computation. On a 355B MoE model (GLM-4.7):

| Hardware | Zero-overhead threshold |
|---|---|
| RTX 4090 + 1x Gen4 NVMe | ~8K context |
| RTX 5090 + 1x Gen5 NVMe | ~4K context |
| RTX PRO 6000 + GDS 5x Gen5 RAID0 | ~2K context |

All configurations reach zero streaming overhead well before the agent training zone (50K+ tokens). The same property that makes agent training expensive (long contexts = lots of compute per layer) is exactly what makes streaming free.

### Streaming backends and partial residency

- **CPU pinned RAM / mmap staged buffers:** 10-27 GB/s via PCIe. Works on any system.
- **GPU Direct Storage (GDS):** Direct NVMe-to-GPU DMA, up to 49 GB/s with 5x Gen5 NVMe RAID0.
- **Partial residency:** The system automatically detects available GPU memory and keeps as many layers resident as possible, streaming only the overflow. A 355B model on a 32 GB GPU might keep 20% of layers on-GPU and stream the other 80%.
- **Zero configuration:** The backend and residency strategy are auto-detected. No tuning required.

---

## Feature 4: Training optimization stack

Beyond quantization and streaming, bitsandbytes 2 includes a stack of training optimizations that collectively achieve **4x faster training** versus naive QLoRA:

- **Chunked cross-entropy:** Never materializes the full [batch x sequence x vocabulary] logits tensor, which at 256K context would be enormous.
- **Chunked MLP:** Splits the sequence dimension with gradient checkpointing, reducing peak memory.
- **Chunked flash attention:** Memory-efficient attention chunked along the context dimension.
- **CPU-offloaded gradient checkpointing:** Asynchronously moves activation checkpoints between GPU and CPU during forward and backward passes.
- **NVFP4 + BNF quantized weights:** 2-5 bit base weights with native tensor core acceleration on Blackwell.
- **8-bit optimizer states:** Compresses Adam optimizer state from 32-bit to 8-bit per parameter.

---

## Training results

All results below are on a **single RTX 5090** (32 GB VRAM), using QLoRA with NVFP4 base weights, rank-64 LoRA adapters, 8-bit optimizer, and gradient checkpointing.

### LoRA with BF16 base weights (full-precision, NVMe streaming)

| Model | Params | Context | Tok/s | TFLOPS | MFU |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | 35B | 256K | 15,623 | 894 | 51.9% |
| Qwen3-Next-80B-A3B | 80B | 256K | 13,017 | 880 | 51.1% |
| GLM-4.7 | 355B | 198K | 848 | 894 | 51.9% |
| MiniMax-M2.5 | 230B | 192K | 2,578 | 890 | 51.6% |
| DeepSeek-V3 | 671B | 128K | 1,297 | 878 | 51.0% |
| Qwen3.5-397B-A17B | 397B | 256K | 4,056 | 806 | 46.7% |

Up to 51.9% model FLOPs utilization (MFU) with zero NVMe overhead at agent-length contexts.

### QLoRA with NVFP4 base weights (Blackwell FP4 tensor cores)

| Model | Params | Context | Tok/s | TFLOPS | MFU |
|---|---|---|---|---|---|
| Qwen3.5-9B | 9B | 64K | 35,200 | 1,583 | 23.0% |
| Qwen3-32B | 32B | 40K | 6,431 | 1,410 | 20.5% |
| GLM-5 | 754B | 128K | 3,578 | 1,204 | 17.5% |
| Kimi-K2.5 | 1.1T | 256K | 1,549 | 1,020 | 14.8% |
| Qwen3.5-397B-A17B | 397B | 256K | 5,838 | 1,160 | 16.8% |
| DeepSeek-V3 | 671B | 160K | 1,259 | 1,017 | 14.8% |

Peak throughput: **1,583 TFLOPS** on a single RTX 5090 (Qwen3.5-9B at 64K context). For reference, the RTX 5090's theoretical FP4 peak is ~6,900 TFLOPS, so the system achieves 14.8-23% utilization even while streaming weights from NVMe and offloading gradients.

### Headline result: 397B at 256K context on one GPU

| Model | Params | Context | Tok/s | TFLOPS |
|---|---|---|---|---|
| Qwen3.5-397B-A17B | 397B | 256K | 5,838 | 1,160 |
| GLM-4.7 | 355B | 198K | 928 | 979 |
| MiniMax-M2.5 | 230B | 192K | 2,816 | 972 |
| Qwen3-32B | 32B | 32K | 7,322 | 1,480 |

A 397-billion-parameter model, training at 256K context length, on a single consumer GPU. This setup replaces what would otherwise require 16 GPUs.

---

## Efficiency vs. SERA agent training

SERA (Soft-Verified Efficient Repository Agents) is a state-of-the-art method for training coding agents. It achieved 49.5% on SWE-bench Verified — matching frontier proprietary models — using supervised fine-tuning with a novel soft verification technique that eliminates the need for test infrastructure.

The original SERA training used Axolotl (a standard training framework) on 16 GPUs:
- **18.7 GPU-days** of compute
- **16 GPUs required** simultaneously
- Trained Qwen3-32B at 32K context

With bitsandbytes 2 as the training backend:
- **< 1 GPU-day** of compute
- **1 GPU** (RTX 5090)
- Same model, same quality target

This is a **19x improvement in efficiency** — one consumer GPU replaces sixteen. The cost reduction makes agent training accessible to individual researchers and small teams, rather than requiring large-lab GPU clusters.

| | SERA + Axolotl | SERA + bitsandbytes 2 |
|---|---|---|
| GPU-days | 18.7 | < 1 |
| GPUs required | 16 | 1 (RTX 5090) |
| Model | Qwen3-32B | Same |
| Context | 32K | Same |
| SWE-bench Verified | 49.5% | Same quality |
| Efficiency gain | baseline | **19x** |

---

## What's next

- **Blackwell NVFP4 kernel optimization** for the RTX 5090's native FP4 path
- **Qwen3-Coder-Next** (512 experts, top-10 routing) as a stress test for the grouped GEMM kernels
- **VQ quantization benchmarks** for sub-2-bit MoE inference
- **RL integration** to use bitsandbytes 2 as the training backend for continual agent improvement via reinforcement learning
- **Open-source release**

---

## Summary

bitsandbytes 2 combines three systems — BNF quantization, NVMe weight streaming, and a training optimization stack — to enable training and inference of 300B-700B parameter models on a single consumer GPU.

**Quantization:** BNF is information-optimal, requires no calibration, and generalizes to 2-5 bits. Combined with Hadamard rotations and NVFP4 on Blackwell, it achieves up to 1.45x inference speedup at 2-bit on MoE models and 970-1,583 TFLOPS training throughput on a single RTX 5090.

**Streaming:** NVMe weight streaming adds zero overhead at agent-length contexts (50K+ tokens). Models up to 400B parameters train on 24-32 GB GPUs with automatic partial residency and backend selection.

**Training efficiency:** The full optimization stack delivers a 19x efficiency improvement over standard multi-GPU training for SERA agent training. A single RTX 5090 replaces sixteen GPUs. Agent training is no longer a large-lab activity.
