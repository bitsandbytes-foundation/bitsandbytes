# QLoRA Implementation Guide

A comprehensive analysis of QLoRA implementations across the ecosystem, with focus
on Unsloth's optimizations and their relevance to bitsandbytes.

## Table of Contents

1. [QLoRA Fundamentals](#1-qlora-fundamentals)
2. [Unsloth: Architecture and Implementation](#2-unsloth-architecture-and-implementation)
3. [Unsloth Open-Source vs. Commercial Code](#3-unsloth-open-source-vs-commercial-code)
4. [Unsloth Feature Catalog](#4-unsloth-feature-catalog)
5. [Other QLoRA Implementations](#5-other-qlora-implementations)
6. [Algorithms Worth Reimplementing](#6-algorithms-worth-reimplementing)
7. [bitsandbytes kbit-gemm: Beyond QLoRA](#7-bitsandbytes-kbit-gemm-beyond-qlora)
8. [Repository References](#8-repository-references)

---

## 1. QLoRA Fundamentals

QLoRA (Dettmers et al., NeurIPS 2023) backpropagates gradients through a frozen,
4-bit quantized pretrained LLM into Low Rank Adapters (LoRA). Three key algorithms:

### NF4 (4-bit NormalFloat) Quantization

An information-theoretically optimal data type for normally distributed weights.
Each quantization bin represents an equal expected number of values from N(0,1),
normalized to [-1, 1]. The 16 NF4 values are:

```
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

NF4 consistently outperforms FP4 by approximately 1 percentage point on MMLU.
Implemented in bitsandbytes via `create_normal_map()`, `Linear4bit`, and `Params4bit`.

### Double Quantization (DQ)

Quantizes the quantization constants (absmax scaling factors) themselves using an
8-bit float format with a block size of 256. Saves ~0.37 bits per parameter (~0.4
GB for a 65B model). The two-level dequantization process:

1. Dequantize absmax2 + code2 -> absmax (float32)
2. Add offset to absmax
3. Dequantize W using absmax -> output (float16/bfloat16)

### Paged Optimizers

Uses NVIDIA unified memory to manage memory spikes during gradient checkpointing.
When GPU memory is exhausted, optimizer states are automatically paged to CPU
memory and paged back when needed.

### Key Paper Settings

- LoRA rank r=64, alpha=16
- LoRA applied to **all linear layers** (not just Q/V attention)
- LoRA dropout 0.1 for models up to 13B
- Adam beta2=0.999, max grad norm 0.3

---

## 2. Unsloth: Architecture and Implementation

Unsloth (github.com/unslothai/unsloth, 52K+ stars) is the leading optimized QLoRA
framework. Created by Daniel Han-Chen and Michael Han-Chen, launched November 2023.

### Core Design Philosophy

Unsloth patches standard HuggingFace model code with custom, hand-optimized
operations. The key insight is that PyTorch's autograd is suboptimal for the
LoRA + quantized weight pattern because:

1. Triton kernels are opaque to autograd (appear as black boxes)
2. The low-rank structure of LoRA creates opportunities for bracket optimization
   in chained matrix multiplications
3. Many intermediate tensors can be eliminated through fusion

### Code Architecture

Two repositories form the complete system:

**`unsloth` (main)** -- Apache 2.0 (kernels dir: AGPLv3)
```
unsloth/
  kernels/
    cross_entropy_loss.py    # Triton CE loss (chunked for large vocabs)
    rope_embedding.py        # Triton RoPE (in-place, fwd+bwd fused)
    swiglu.py                # Triton SwiGLU activation
    geglu.py                 # Triton GeGLU activation
    rms_layernorm.py         # Triton RMSNorm
    layernorm.py             # Triton LayerNorm
    fast_lora.py             # Fused LoRA forward/backward (MLP, QKV, O)
    flex_attention.py        # Flex Attention backend
    fp8.py                   # FP8 quantization support
    utils.py                 # Dequantization, bitsandbytes interface
    moe/                     # Mixture-of-Experts Triton kernels
  models/
    loader.py                # FastLanguageModel / FastModel entry points
    llama.py                 # Llama-family model patching + get_peft_model
    mistral.py               # Mistral model patching
    qwen2.py, qwen3.py       # Qwen model patching
    gemma.py, gemma2.py       # Gemma model patching
    _utils.py                # prepare_model_for_kbit_training, compilation
    vision.py                # Vision model support (FastBaseModel)
    dpo.py, rl.py            # DPO / RL support
  trainer.py                 # Training loop patches
  save.py                    # Model saving / GGUF export
```

**`unsloth-zoo` (utilities)** -- LGPL-3.0
```
unsloth_zoo/
  compiler.py                # torch.compile orchestration
  compiler_replacements.py   # Replacement functions for compiled models
  gradient_checkpointing.py  # Custom gradient checkpointing + CPU offload
  loss_utils.py              # Fused linear cross-entropy, cut-cross-entropy
  peft_utils.py              # get_peft_regex, merge_and_overwrite_lora
  patching_utils.py          # BnB compilation patches, model patches
  training_utils.py          # Training loop utilities
  rl_replacements.py         # GRPO compiled Triton kernels
  saving_utils.py            # Model merging/saving
  tiled_mlp.py               # Tiled MLP for memory efficiency
  vllm_utils.py              # vLLM inference integration
  temporary_patches/         # Model-specific hotfixes
    bitsandbytes.py          # BnB-specific patches
    moe_bnb.py               # BnB patches for MoE models
```

### How Unsloth Patches Models for QLoRA

The loading sequence (from `loader.py`):

1. **`FastLanguageModel.from_pretrained()`** dispatches to model-specific loaders
   (e.g., `FastLlamaModel`) based on `model_config.model_type`
2. Base model is loaded with `load_in_4bit=True` via HuggingFace + bitsandbytes
3. Default quantization config: NF4, double quantization enabled, bfloat16 compute
4. **`get_peft_model()`** applies LoRA adapters to all linear layers
5. **`patch_peft_model()`** replaces standard forward passes with fused versions:
   - MLP forward -> `apply_lora_mlp_swiglu` (or `geglu` for Gemma)
   - QKV forward -> `apply_lora_qkv`
   - O projection forward -> `apply_lora_o`
6. `prepare_model_for_kbit_training()` freezes base weights, sets up gradient
   checkpointing, ensures LoRA params are in float32 for mixed precision
7. Loss functions are patched with Triton cross-entropy
8. RMSNorm layers are patched with Triton versions

### Manual Backpropagation: The Core Innovation

Unsloth implements `torch.autograd.Function` subclasses with hand-derived gradients.
Three main custom autograd functions:

**LoRA_MLP** (`fast_lora.py:28`): Handles the full gated MLP with LoRA on gate,
up, and down projections. Forward:
```
e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
g = matmul_lora(X, upW, upW_quant, upA, upB, upS)
h = swiglu_fg_kernel(e, g)   # Triton fused SwiGLU
i = matmul_lora(h, downW, downW_quant, downA, downB, downS)
```

Backward computes 6 adapter gradients (d_gateA, d_gateB, d_upA, d_upB, d_downA,
d_downB) plus dX. Key optimization: uses `addmm_` with alpha/beta for fused
scale-and-accumulate, avoiding temporary allocations:
```python
d_downA.addmm_(h.t(), dY @ downB.t(), alpha=downS, beta=0)
d_downB.addmm_(downA.t() @ h.t(), dY, alpha=downS, beta=0)
```

The dX gradient reuses input memory via in-place operations:
```python
dX = torch.matmul(df, upW.t(), out=X if ctx.inplace else None)
dX.addmm_(df @ upB.t(), upA.t(), alpha=upS)  # LoRA contribution
```

**LoRA_QKV** (`fast_lora.py:327`): Handles fused Q/K/V projections. Computes
Q, K, V in one forward, then backward produces 6 adapter gradients plus combined
dX from all three projections.

**LoRA_W** (`fast_lora.py:562`): Single-projection LoRA for the output projection.

### The `matmul_lora` Function

Central to everything (`utils.py:1000`):
```python
def matmul_lora(X, W, W_quant, A, B, s, out=None):
    # Dequantize 4-bit weights (uses global buffer to avoid allocation)
    W = fast_dequantize(W, W_quant, use_global_buffer=True)
    out = torch_matmul(X, W.t(), out=out)
    if A is not None:
        # LoRA: out += X @ A.t() @ B.t() * s
        A, B = A.t(), B.t()
        XA = torch_matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha=s)
    return out
```

Note the bracket optimization: `X @ A @ B` is computed as `(X @ A) @ B` because
A is small (hidden_dim x rank), so `X @ A` produces a small intermediate.

### Fast Dequantization

`fast_dequantize()` (`utils.py:462`) calls bitsandbytes C functions directly via
ctypes, bypassing Python overhead:

1. `cdequantize_blockwise_fp32()` -- dequantize absmax2 (the double-quant layer)
2. Add offset to absmax
3. `cdequantize_blockwise_fp16_nf4()` or `_bf16_nf4()` -- dequantize weights

Uses **global weight buffers** (`WEIGHT_BUFFERS`, `ABSMAX_BUFFERS`) to avoid
repeated allocation/deallocation of the dequantized weight matrix. This is
significant because dequantization happens on every forward and backward pass.

### Triton Kernel Details

**SwiGLU** (`swiglu.py`): Two kernels:
- `_fg_kernel`: Forward. `f = e * sigmoid(e); h = f * g` (the SiLU(gate) * up pattern)
- `_DWf_DW_dfg_kernel`: Backward. Fuses computation of df, dg, de into a single
  kernel pass. Stores results in-place into the DW, e, g buffers. Uses adaptive
  int32/int64 indexing via `LONG_INDEXING` constexpr for long-context support.

**RMSNorm** (`rms_layernorm.py`): Forward stores inverse variance for backward.
Backward uses a single kernel with special Gemma handling (`W_row + 1.0` for
Gemma's +1 layernorm convention). Both forward and backward use `calculate_settings()`
to select optimal BLOCK_SIZE and num_warps.

**Cross-Entropy Loss** (`cross_entropy_loss.py`): For vocab sizes <= 65536, a
single kernel computes logsumexp and loss. For larger vocabs (e.g., Gemma 256K),
uses `_chunked_cross_entropy_forward` with 2D grid `(n_rows, n_chunks)`, computing
per-chunk logsumexp then reducing with `torch.logsumexp(logsumexp, dim=1)`.
Backward: `dC/dx = exp[x - logsumexp] - 1` for the label, `exp[x - logsumexp]`
otherwise. Supports logit softcapping (Gemma 2) and logit scaling (Cohere).

**RoPE** (`rope_embedding.py`): In-place rotary embedding. Two implementations:
- `_rope_embedding`: Simple version with group processing (ROPE_GROUP_SIZE=4 heads
  per thread block)
- `_rope_embedding_QK`: Fused Q+K version for attention layers with rope indices
  support. Backward reuses forward kernel with `sin1 = -sin1`.

### Gradient Checkpointing with CPU Offload

Implemented in `unsloth_zoo/gradient_checkpointing.py`. Uses asynchronous
non-blocking GPU-to-CPU transfers (`tensor.to('cpu', non_blocking=True)`) to
overlap data movement with computation. Overhead: +1.9%. Results:
- H100 80GB: 228K tokens (4x over HF+FA2's 57.5K)
- RTX 4090 24GB: 56.4K tokens (vs 14.1K)
- Enables 1.7x larger batch sizes

---

## 3. Unsloth Open-Source vs. Commercial Code

### Open-Source Components

| Component | License | Contents |
|---|---|---|
| `unsloth/` (main) | Apache 2.0 | Model loading, patching, trainer |
| `unsloth/kernels/` | AGPLv3 | All Triton kernels (RoPE, SwiGLU, CE loss, RMSNorm, fast_lora) |
| `unsloth-zoo/` | LGPL-3.0 | PEFT utils, gradient checkpointing, loss utils, compiler, training utils, vLLM integration, RL replacements |

**Everything in the open-source repos is available for inspection and use.** The
AGPLv3 license on kernels requires sharing source if you distribute modified
versions. The LGPL on zoo allows linking without full copyleft.

### Commercial / Proprietary (Not in Public Repos)

The **Pro** and **Enterprise/Max** tiers contain proprietary optimizations:

| Feature | Free | Pro | Max |
|---|---|---|---|
| GPU support | Single NVIDIA | Up to 8 GPUs | Multi-node, NVIDIA/Intel/AMD |
| Speed | ~2x faster | ~2.5x faster | Up to 30x faster |
| Memory | ~70% less | 20% less (vs free) | Best-in-class |
| Accuracy | Baseline | Improved | +30% improvement |

The multi-GPU, multi-node, and cross-vendor hardware support are proprietary.
The "30% accuracy improvement" in Max likely involves proprietary training
techniques not visible in the open-source code.

### What Is *Not* Proprietary

The core algorithmic innovations are all visible in the open-source code:
- Manual backprop with bracket optimization
- All Triton kernels
- Fast dequantization with global buffers
- Gradient checkpointing with CPU offload
- Dynamic 4-bit quantization logic (in the model loader)
- The `matmul_lora` fused forward
- Cross-entropy chunking strategy

---

## 4. Unsloth Feature Catalog

### Triton Kernels (Open Source, AGPLv3)

| Kernel | File | What It Does |
|---|---|---|
| SwiGLU forward | `swiglu.py` | `h = silu(gate) * up` fused |
| SwiGLU backward | `swiglu.py` | `df, dg, de` fused in single pass |
| GeGLU forward/backward | `geglu.py` | Exact and approximate GELU variants |
| RMSNorm forward | `rms_layernorm.py` | `x * rsqrt(mean(x^2) + eps) * w` |
| RMSNorm backward | `rms_layernorm.py` | With Gemma `w+1` variant |
| Cross-Entropy forward | `cross_entropy_loss.py` | Logsumexp-based, chunked for >64K vocab |
| Cross-Entropy backward | `cross_entropy_loss.py` | `softmax(x) - one_hot(label)` |
| RoPE forward | `rope_embedding.py` | In-place `Q*cos - rotate(Q)*sin` |
| RoPE backward | `rope_embedding.py` | Reuses forward with `-sin` |
| MoE grouped GEMM | `moe/grouped_gemm/` | Forward + backward for MoE layers |

### Custom Autograd Functions (Open Source, Apache 2.0)

| Function | File | Fused Operations |
|---|---|---|
| `LoRA_MLP` | `fast_lora.py` | gate+up+down projections with LoRA |
| `LoRA_QKV` | `fast_lora.py` | Q+K+V projections with LoRA |
| `LoRA_W` | `fast_lora.py` | Single projection with LoRA |
| `Fast_RMS_Layernorm` | `rms_layernorm.py` | Triton-backed RMSNorm autograd |
| `Fast_RoPE_Embedding` | `rope_embedding.py` | Triton-backed RoPE autograd |
| `Fast_RoPE_Embedding_QK` | `rope_embedding.py` | Fused Q+K RoPE autograd |
| `Fast_CrossEntropyLoss` | `cross_entropy_loss.py` | Triton-backed CE autograd |

### Infrastructure Features (Open Source, LGPL-3.0 in zoo)

| Feature | Location | Description |
|---|---|---|
| Gradient checkpointing + CPU offload | `unsloth_zoo/gradient_checkpointing.py` | Async non-blocking offload, +1.9% overhead |
| Fused linear cross-entropy | `unsloth_zoo/loss_utils.py` | Avoids materializing full logits |
| PEFT regex targeting | `unsloth_zoo/peft_utils.py` | Smart layer selection for LoRA |
| torch.compile orchestration | `unsloth_zoo/compiler.py` | Model-aware compilation |
| RL/GRPO Triton kernels | `unsloth_zoo/rl_replacements.py` | Compiled GRPO loss (3 Triton kernels) |
| vLLM integration | `unsloth_zoo/vllm_utils.py` | Fast inference with LoRA adapters |
| Tiled MLP | `unsloth_zoo/tiled_mlp.py` | Memory-efficient MLP for large models |
| BnB patches | `unsloth_zoo/temporary_patches/bitsandbytes.py` | Fixes for bitsandbytes compilation |
| MoE BnB patches | `unsloth_zoo/temporary_patches/moe_bnb.py` | BnB support for MoE architectures |

### Model Loading Features

| Feature | Description |
|---|---|
| `load_in_4bit=True` | Standard QLoRA via bitsandbytes NF4 |
| `load_in_8bit=True` | 8-bit LoRA via bitsandbytes LLM.int8() |
| `load_in_fp8=True` | FP8 LoRA via torchao |
| `load_in_16bit=True` | 16-bit LoRA (no quantization) |
| `full_finetuning=True` | Full parameter training (no LoRA) |
| Dynamic 4-bit | Selective per-layer quantization |
| Pre-quantized models | Unsloth hosts `-bnb-4bit` variants on HuggingFace |
| QAT support | `qat_scheme` parameter for quantization-aware training |

---

## 5. Other QLoRA Implementations

### PEFT (HuggingFace)

The canonical LoRA/QLoRA library. Provides `LoraConfig` + `get_peft_model()`.
No custom kernels -- relies entirely on bitsandbytes for quantization and standard
PyTorch for training. Supports LoftQ initialization, rsLoRA, DoRA. This is what
most other frameworks build on top of.

- GitHub: github.com/huggingface/peft

### Axolotl

YAML-configuration wrapper around Transformers, PEFT, DeepSpeed. Added its own
LoRA optimizations in February 2025, **inspired by Unsloth**:

- SwiGLU/GeGLU Triton kernels (similar to Unsloth's)
- Fused LoRA MLP autograd functions
- Fused LoRA attention autograd functions
- Benchmarks on H100: 1.28x speedup (rank 16, seq 512), up to 1.76x for quantized

Also integrates Liger Kernel (LinkedIn's Triton kernels) for RMSNorm, RoPE,
SwiGLU, CrossEntropy, FusedLinearCrossEntropy.

- GitHub: github.com/axolotl-ai-cloud/axolotl

### TRL (HuggingFace)

Library for post-training via SFT, GRPO, DPO, reward modeling. Has first-class
QLoRA support through PEFT integration. Does not implement its own quantization
-- delegates entirely to bitsandbytes + PEFT. Value-add is seamless alignment
training on quantized models.

- GitHub: github.com/huggingface/trl

### LLaMA-Factory

Unified framework for 100+ LLMs/VLMs. Unique in supporting **six quantization
backends**: bitsandbytes, HQQ, EETQ, GPTQ, AWQ, AQLM. Bit widths from 2 to 8.
Both CLI and web UI. Integrates GaLore, BAdam, APOLLO, DoRA, LongLoRA, NEFTune.

- GitHub: github.com/hiyouga/LlamaFactory

### torchtune (Meta/PyTorch)

PyTorch-native fine-tuning. **Does not use bitsandbytes** -- uses torchao's
`NF4Tensor` for a pure-PyTorch NF4 implementation. Zero dependency on PEFT or
Transformers. Memory: ~9GB for Llama3-8B QLoRA (vs ~19GB LoRA).

- GitHub: github.com/meta-pytorch/torchtune

### FSDP-QLoRA (Answer.AI)

Enables training a 70B model on two 24GB consumer GPUs via FSDP + QLoRA.
Required three key changes to bitsandbytes:

1. `bnb_4bit_quant_storage` parameter (FSDP needs float dtypes for sharding)
2. Quantization metadata persistence across FSDP sharding
3. Prevention of double quantization during FSDP's CPU/GPU parameter movement

- GitHub: github.com/AnswerDotAI/fsdp_qlora

### Notable Alternatives and Extensions

| Method | Key Idea | Relation to QLoRA |
|---|---|---|
| **LoftQ** (ICLR 2024) | Quantization-aware LoRA initialization | Better init for QLoRA, 8%+ gains at 2-bit |
| **QDoRA** (Answer.AI + NVIDIA) | Weight decomposition + QLoRA | Magnitude/direction decomposition, outperforms QLoRA |
| **HQQ** (Mobius Labs) | Half-quadratic quantization | Drop-in BnB replacement, calibration-free |
| **GaLore** | Gradient low-rank projection | Alternative: full-param training with low memory |
| **APOLLO** (MLSys 2025) | Random projection + LR scaling | SGD-level memory, AdamW-level performance |
| **rsLoRA** | `alpha/sqrt(r)` scaling | Stabilizes high-rank LoRA |

---

## 6. Algorithms Worth Reimplementing

Based on the analysis above, these are the key algorithms from the ecosystem that
could be implemented in or alongside bitsandbytes:

### High Priority: From Unsloth

**1. Fast Dequantization with Global Buffers**

Unsloth's `fast_dequantize()` avoids repeated allocation of the dequantized weight
matrix by maintaining per-device global buffers (`WEIGHT_BUFFERS`, `ABSMAX_BUFFERS`).
It calls bitsandbytes C functions directly via ctypes with explicit CUDA streams.
This is a pure performance optimization that could be integrated into bitsandbytes
itself.

Key code path: `unsloth/kernels/utils.py:462-568`

**2. Triton Double-Dequantization Kernel**

External contributors have demonstrated 1.6-1.8x speedups by fusing the two-step
double dequantization (absmax2 -> absmax -> weights) into a single Triton kernel,
eliminating the intermediate absmax buffer and a kernel launch. This is directly
relevant to bitsandbytes' NF4 dequantization path.

**3. Fast GEMV for Inference**

Unsloth's `fast_gemv()` (`utils.py:649-954`) provides an optimized path for
single-token inference (seq_len=1) that calls bitsandbytes' 4-bit GEMV kernels
directly, bypassing the normal dequantize-then-matmul path.

### Medium Priority: Autograd Optimizations

**4. Fused LoRA Backward Pass**

The `LoRA_MLP`, `LoRA_QKV`, and `LoRA_W` custom autograd functions demonstrate
significant speedups from:
- Computing all LoRA adapter gradients in a single backward function
- Using `addmm_` with alpha/beta for fused scale-accumulate
- In-place dX computation to save memory
- Bracket-optimized matrix chain multiplication

These are training-side optimizations that could be provided as a bitsandbytes
utility or contributed to PEFT.

**5. Fused Activation Kernels**

The SwiGLU and GeGLU Triton kernels fuse the activation function forward and
backward into single kernel launches. The backward kernel (`_DWf_DW_dfg_kernel`)
is particularly clever: it computes three outputs (h, df, de) in a single pass
and stores them in-place into the input buffers.

### From the Broader Ecosystem

**6. LoftQ Initialization**

Already in PEFT, but the quantization-aware initialization could be tighter
integrated with bitsandbytes' quantization pipeline. Key algorithm: alternating
between weight quantization and SVD-based low-rank approximation of the
quantization error.

**7. HQQ-style Calibration-Free Quantization**

Half-quadratic quantization frames quantization as a robust optimization problem.
At 4-bit, it outperforms bitsandbytes in both perplexity and VRAM. Could inform
improvements to bitsandbytes' NF4 implementation.

**8. FSDP-Aware Quantization Metadata**

The metadata persistence patterns from FSDP-QLoRA are already partially in
bitsandbytes (>= 0.43.0) but the patterns for preventing double-quantization and
maintaining quant state across FSDP sharding could be further hardened.

---

## 7. bitsandbytes kbit-gemm: Beyond QLoRA

The `feature/kbit-gemm` branch in bitsandbytes contains new implementations
that go significantly beyond what Unsloth offers. While Unsloth relies on
bitsandbytes for NF4 dequantization and wraps it with Triton kernels, the
kbit-gemm work implements a full generalized k-bit quantization and inference
stack in pure CUDA.

### What kbit-gemm Does

Generalized k-bit quantization (k=2,3,4,5) with blocksize 32, using:

- **Bit-plane packing**: Unlike NF4's nibble packing (two 4-bit values per
  byte), kbit uses bit-plane representation. Each 32-element block produces
  k uint32 words where bit j of word b is bit b of element j's quantization
  index. This generalizes to any bit width without format changes.

- **E4M4 absmax encoding**: Per-block scale factors stored as a single byte
  (4-bit exponent, 4-bit mantissa, bias 11). Decoded branchlessly in the
  inner loop to avoid warp divergence. This is more compact than bitsandbytes'
  current float32 absmax.

- **Warp-shuffle codebook lookup**: The 2^k codebook entries (4 for k=2, 32
  for k=5) fit in warp lane registers. Lookup via `__shfl_sync` is a
  single-cycle register-to-register operation -- faster than shared memory
  lookup.

- **Repack tiling**: One-time data reorganization (TILE_K=64, TILE_N=128) for
  coalesced vector loads via `cp.async`. This is amortized at model load time.

### Three-Kernel Strategy

The branch implements three kernels optimized for different regimes:

**Kernel 1: Scalar GEMV** (highest priority, for autoregressive decode)
- For M=1-4 (token generation). No tensor cores -- pure scalar FMA.
- Eliminates MMA waste (93.75% of tensor core work is wasted at M=1 with
  TILE_M=16). Uses 1 warp per output column with K-dimension split.
- Projected 3-5x speedup over cuBLAS fp16 at batch=1-4.
- Current state: ~54% DRAM bandwidth on large shapes, with a path to 75%+
  by moving from shared-memory tiling to the bnb gemv_4bit register-file
  pattern.
- Supports both dense (single matrix) and grouped (MoE) dispatch.

**Kernel 2: Grouped Expert GEMM** (for MoE inference at batch >= 8)
- Batches all active MoE experts into a single kernel launch.
- Solves the fundamental MoE problem: individual expert GEMMs have 3-12% SM
  utilization (4-16 tiles on 128 SMs). Grouping 256+ expert invocations
  creates 1000+ tiles, achieving full SM utilization.
- Measured 1.6-2x speedup over cuBLAS at batch=16-64 for Qwen3 MoE layers.
- 3.6x data compression pays off when total expert data exceeds L2 cache.

**Kernel 3: Dequant + cuBLAS** (for prefill with large M)
- Dequantize kbit weights to fp16, then call cuBLAS for the GEMM.
- cuBLAS is unbeatable at large M (tensor core utilization near peak).
- Dequant kernel runs at 72-78% of peak DRAM bandwidth.

### How This Compares to Unsloth

| Aspect | Unsloth | bitsandbytes kbit-gemm |
|---|---|---|
| Bit widths | 4-bit only (NF4/FP4) | 2, 3, 4, 5-bit |
| Quantization format | Nibble packing (bitsandbytes) | Bit-plane packing |
| Absmax format | float32 (32 bytes/block) | E4M4 (1 byte/block) |
| Codebook lookup | Shared memory | Warp shuffle (faster) |
| Inference kernel | Relies on bitsandbytes gemv_4bit | Custom scalar GEMV |
| MoE support | Per-expert dispatch | Grouped GEMM (single launch) |
| Training kernels | Triton (SwiGLU, RoPE, CE loss) | Not yet (CUDA focus) |
| Language | Triton | CUDA |
| Hardware target | NVIDIA (Triton portability) | NVIDIA CUDA (sm_75+) |

The key difference: Unsloth optimizes the *training* path (backward passes,
gradient checkpointing, fused LoRA), while kbit-gemm optimizes the *inference*
path (GEMV for decode, grouped GEMM for MoE). They are complementary.

### Relevance to QLoRA

The kbit-gemm work extends the quantization beyond QLoRA's fixed 4-bit:
- **2-bit** quantization: 8x compression (vs 4x for NF4). Quality degrades
  but LoftQ initialization can partially recover it.
- **3-bit**: 5.3x compression. A sweet spot between quality and size.
- **5-bit**: 3.2x compression. Higher quality than NF4 at modest size increase.

For QLoRA training specifically, the dequantization improvements (E4M4 absmax,
warp shuffle codebook) could speed up the forward pass, and the global buffer
pattern from Unsloth could speed up the backward pass.

### Branch Files

Key guides on the `feature/kbit-gemm` branch:
- `guide.md` -- 1000-line comprehensive kernel development guide
- `optimization.md` -- Three-kernel strategy and performance analysis
- `optimization2.md` -- Phase 2 optimization (MoE grouping, instruction analysis)
- `progress.md` -- Full development log
- `agents/scalar_gemv_guide.md` -- Scalar GEMV implementation details

---

## 8. Repository References

### Cloned Locally

| Repository | Local Path | Description |
|---|---|---|
| unsloth | `/tmp/unsloth` | Main Unsloth framework |
| unsloth-zoo | `/tmp/unsloth-zoo` | Unsloth utilities package |

### Key Source Files to Study

| Purpose | File |
|---|---|
| LoRA fused backward | `/tmp/unsloth/unsloth/kernels/fast_lora.py` |
| Dequantization + GEMV | `/tmp/unsloth/unsloth/kernels/utils.py` |
| SwiGLU Triton kernels | `/tmp/unsloth/unsloth/kernels/swiglu.py` |
| RoPE Triton kernels | `/tmp/unsloth/unsloth/kernels/rope_embedding.py` |
| Cross-entropy Triton | `/tmp/unsloth/unsloth/kernels/cross_entropy_loss.py` |
| RMSNorm Triton | `/tmp/unsloth/unsloth/kernels/rms_layernorm.py` |
| Model loading / patching | `/tmp/unsloth/unsloth/models/loader.py` |
| Llama model patches | `/tmp/unsloth/unsloth/models/llama.py` |
| Model training prep | `/tmp/unsloth/unsloth/models/_utils.py` |
| Gradient checkpointing | `/tmp/unsloth-zoo/unsloth_zoo/gradient_checkpointing.py` |
| PEFT utilities | `/tmp/unsloth-zoo/unsloth_zoo/peft_utils.py` |
| Loss utilities | `/tmp/unsloth-zoo/unsloth_zoo/loss_utils.py` |
| BnB patches | `/tmp/unsloth-zoo/unsloth_zoo/temporary_patches/bitsandbytes.py` |

### bitsandbytes kbit-gemm Branch

| Purpose | File (on `feature/kbit-gemm` branch) |
|---|---|
| Full kernel dev guide | `guide.md` |
| Three-kernel strategy | `optimization.md` |
| Phase 2 optimization | `optimization2.md` |
| Development log | `progress.md` |
| Scalar GEMV guide | `agents/scalar_gemv_guide.md` |
| CUDA kernels | `csrc/ops.cu` |
| Python dispatch | `bitsandbytes/backends/cuda/ops.py` |
| Tests | `tests/test_scalar_gemv.py` |

### External Links

- QLoRA paper: https://arxiv.org/abs/2305.14314
- Unsloth technical blog: https://unsloth.ai/introducing
- Unsloth docs: https://unsloth.ai/docs
- PEFT: https://github.com/huggingface/peft
- Axolotl: https://github.com/axolotl-ai-cloud/axolotl
- LLaMA-Factory: https://github.com/hiyouga/LlamaFactory
- torchtune: https://github.com/meta-pytorch/torchtune
- FSDP-QLoRA: https://github.com/AnswerDotAI/fsdp_qlora
- LoftQ: https://arxiv.org/abs/2310.08659
- HQQ: https://github.com/mobiusml/hqq
- Triton double-dequant analysis: https://medium.com/@samdj0245/accelerating-nf4-double-dequantization-within-a-single-triton-kernel-f26a0f35b372
