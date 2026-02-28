# NVFP4 Implementation Guide

A comprehensive technical guide to the NVFP4 (NVIDIA FP4) data format, its CUDA-level
implementation, rotation-based quantization methods, and the state-of-the-art open-source
implementations from IST-DASLab (Dan Alistarh's lab) and others.

---

## Table of Contents

1. [Format Specification: E2M1](#1-format-specification-e2m1)
2. [Two-Level Micro-Block Scaling Architecture](#2-two-level-micro-block-scaling-architecture)
3. [NVFP4 vs MXFP4: Format Comparison](#3-nvfp4-vs-mxfp4-format-comparison)
4. [Blackwell Hardware: tcgen05.mma and Tensor Cores](#4-blackwell-hardware-tcgen05mma-and-tensor-cores)
5. [Rotation-Based Quantization](#5-rotation-based-quantization)
6. [MR-GPTQ: Micro-Rotated GPTQ](#6-mr-gptq-micro-rotated-gptq)
7. [QuTLASS: IST-DASLab CUDA Kernels](#7-qutlass-ist-daslab-cuda-kernels)
8. [FP-Quant: End-to-End Quantization Pipeline](#8-fp-quant-end-to-end-quantization-pipeline)
9. [Four Over Six: Adaptive Block Scaling](#9-four-over-six-adaptive-block-scaling)
10. [RaZeR: Redundant Zero Remapping](#10-razer-redundant-zero-remapping)
11. [Quartet: Native FP4 Training](#11-quartet-native-fp4-training)
12. [CUDA-Level Implementation Details](#12-cuda-level-implementation-details)
13. [Software Ecosystem and Deployment](#13-software-ecosystem-and-deployment)
14. [Implementation Considerations for bitsandbytes](#14-implementation-considerations-for-bitsandbytes)
15. [References](#15-references)

---

## 1. Format Specification: E2M1

NVFP4 is a 4-bit micro floating-point format using the **E2M1** encoding:

- **1 sign bit** (S)
- **2 exponent bits** (E), bias = 1
- **1 mantissa bit** (M)

### Encoding Formula

For exponent E and mantissa M:
- Normal values (E != 0): `(-1)^S * 2^(E-1) * (1 + M/2)`
- Subnormal values (E == 0): `(-1)^S * M/2`

### Complete Encoding Table

| Bits (SEMM) | Sign | Exp | Man | Value   |
|-------------|------|-----|-----|---------|
| `0000`      | +    | 00  | 0   | **+0.0** |
| `0001`      | +    | 00  | 1   | **+0.5** |
| `0010`      | +    | 01  | 0   | **+1.0** |
| `0011`      | +    | 01  | 1   | **+1.5** |
| `0100`      | +    | 10  | 0   | **+2.0** |
| `0101`      | +    | 10  | 1   | **+3.0** |
| `0110`      | +    | 11  | 0   | **+4.0** |
| `0111`      | +    | 11  | 1   | **+6.0** |
| `1000`      | -    | 00  | 0   | **-0.0** |
| `1001`      | -    | 00  | 1   | **-0.5** |
| `1010`      | -    | 01  | 0   | **-1.0** |
| `1011`      | -    | 01  | 1   | **-1.5** |
| `1100`      | -    | 10  | 0   | **-2.0** |
| `1101`      | -    | 10  | 1   | **-3.0** |
| `1110`      | -    | 11  | 0   | **-4.0** |
| `1111`      | -    | 11  | 1   | **-6.0** |

The representable magnitudes are: **{0, 0.5, 1, 1.5, 2, 3, 4, 6}**.

Note the non-uniform spacing: the gap between 0 and 0.5 is 0.5, but between 4 and 6 is 2.
This is characteristic of floating-point: relative precision is roughly constant while
absolute precision decreases with magnitude.

### Memory Packing

Two FP4 values are packed into a single byte. The first element occupies the 4 least
significant bits; the second occupies the 4 most significant bits:

```
Byte: [ elem1_high | elem1_low | elem0_high | elem0_low ]
      [ S E E M     | S E E M    ]
        ^^^^^^^^^^    ^^^^^^^^^^
        element 1     element 0
```

---

## 2. Two-Level Micro-Block Scaling Architecture

Raw E2M1 can only represent values in [-6, 6]. Real-world tensors have much wider ranges.
NVFP4 uses a **two-level hierarchical scaling** scheme to recover dynamic range:

### Level 1: Per-Block Scale (E4M3 FP8)

Every contiguous block of **16 FP4 values** shares a single **E4M3 FP8** scaling factor.
E4M3 has 1 sign bit, 4 exponent bits, and 3 mantissa bits, providing non-power-of-two
fractional precision (unlike MXFP4's E8M0 which is limited to powers of two).

```
Block of 16 values:  [v0, v1, ..., v15]  (each 4-bit E2M1)
Block scale:         s_block              (8-bit E4M3)
```

Reconstruction per element: `x_i = v_i * s_block`

### Level 2: Per-Tensor Scale (FP32)

A single **FP32** scalar normalizes the entire tensor's distribution before block-level
quantization. This compensates for E4M3's limited dynamic range (max ~448) compared to
the full FP32 range needed by real tensors.

```
Full reconstruction:  x_i = v_i * s_block * s_tensor
```

### Quantization Procedure

Given a tensor X:

1. **Compute per-tensor scale** `s_tensor` — typically the tensor's absmax or an
   MSE-optimized value.
2. **Divide** X by `s_tensor` to get the normalized tensor X'.
3. **For each block of 16 elements** in X':
   a. Compute block scale `s_block` — e.g., `max(|x'_i|) / 6.0`, quantized to E4M3.
   b. Divide each element by `s_block`.
   c. Round each result to the nearest E2M1 value.
   d. Pack two FP4 values per byte.
4. Store: packed FP4 data + E4M3 block scales + FP32 tensor scale.

### Memory Overhead

- 4 bits per value + 8 bits per 16 values for block scale = **4.5 bits per value** average
- Plus one FP32 per tensor (negligible for large tensors)
- **~3.5x compression** vs FP16, **~1.8x** vs FP8

---

## 3. NVFP4 vs MXFP4: Format Comparison

| Property            | NVFP4                  | MXFP4 (OCP MX)        |
|---------------------|------------------------|------------------------|
| Element format      | E2M1 (4-bit)          | E2M1 (4-bit)          |
| Block size          | **16 elements**        | **32 elements**        |
| Scale format        | **E4M3** (FP8)        | **E8M0** (power-of-2) |
| Scale precision     | Fractional (mantissa)  | Powers-of-two only     |
| Bits per element    | ~4.5                   | ~4.25                  |
| Per-tensor scale    | FP32 (required)        | None                   |

### Why This Matters

1. **Block size 16 vs 32**: Smaller blocks adapt more tightly to local value distributions,
   cutting quantization error roughly in half for heavy-tailed distributions common in LLMs.

2. **E4M3 vs E8M0 scales**: E8M0 can only represent powers of two (1, 2, 4, 8, ...).
   E4M3 supports fractional values like 1.5, 3.5, etc. Empirically, E4M3 reduces scale
   quantization MSE from ~0.72 to ~0.08 on typical weight distributions.

3. **The trade-off**: NVFP4 uses slightly more memory (4.5 vs 4.25 bits/element) but
   achieves materially better accuracy. On Llama-3.1-8B, NVFP4 with simple RTN recovers
   ~96% of FP16 accuracy; MXFP4 with RTN recovers only ~69%.

---

## 4. Blackwell Hardware: tcgen05.mma and Tensor Cores

NVFP4 has **native hardware support** on NVIDIA Blackwell GPUs (B200, B300, RTX 5090).
The 5th-generation Tensor Cores execute FP4 matrix multiplications directly.

### PTX Instruction: tcgen05.mma

Blackwell replaces Hopper's `wgmma.mma_async` with a new family of `tcgen05.*`
instructions. The MMA instruction format:

```asm
tcgen05.mma.cta_group.kind [d-tmem], a-desc, b-desc, idesc,
                           {disable-output-lane}, enable-input-d {,scale-input-d};
```

- **`d-tmem`**: Destination in Tensor Memory (TMEM), a dedicated 256KB on-SM memory
- **`a-desc`, `b-desc`**: 64-bit shared memory descriptors for operand tiles
- **`idesc`**: 32-bit instruction descriptor encoding data type, MMA shape, sparsity
- **`kind`**: Modifier specifying the data format — relevant kinds for FP4:
  - `mxf4`: MXFP4 block-scaled multiplication
  - `nvf4mxf4`: Mixed NVFP4/MXFP4 multiplication

### Block-Scaled MMA Semantics

For block-scaled formats, the hardware computes:

```
D = C + (A × SF_A) · (B × SF_B)
```

Scale factors `SF_A` and `SF_B` are applied along the K (contraction) dimension. For
NVFP4, every 16 elements in K share one E4M3 scale factor; for MXFP4, every 32 elements
share one E8M0 scale factor.

The hardware **automatically handles**:
- Unpacking of packed FP4 bytes
- Scale factor application per micro-block
- Dequantization to internal precision for accumulation
- Accumulation in FP32

### Tensor Memory (TMEM)

Blackwell introduces a dedicated 256KB Tensor Memory per SM:
- 512 columns × 128 lanes × 32 bits
- Read bandwidth: 16 TB/s per SM
- Write bandwidth: 8 TB/s per SM
- Used exclusively for MMA accumulator storage
- Accessed via `tcgen05.ld`, `tcgen05.st`, `tcgen05.cp` instructions

Unlike previous architectures, **no register file is used for MMA operands or
accumulators**. This frees registers for epilogue computation (quantization, scaling, etc.).

### Performance

- FP4 peak: **~7700 TFLOPS** at 2.4 GHz (per GPU)
- FP4 instructions are 2–4x faster than FP8 instructions
- Instruction latency: ~11 cycles regardless of precision
- Layer-wise speedups: 3.6x (B200) and 6x (RTX 5090) vs FP16
- End-to-end inference speedups: 2.2x (B200) and 4x (RTX 5090) vs FP16

### Block-Scale Memory Layout (Swizzle Format)

The hardware expects scale factors in a specific **block-scaled swizzle format** for
`tcgen05.mma`. The layout depends on the tile dimensions used by the MMA instruction.
CUTLASS and QuTLASS provide utility functions to reorder scale factors from a linear
layout into the hardware-expected format. The QuTLASS `to_blocked()` utility and Triton's
block-scaled matmul tutorial document these layouts.

---

## 5. Rotation-Based Quantization

Rotations are one of the most important techniques for making NVFP4 quantization practical.
Without rotations, naive round-to-nearest (RTN) quantization causes unacceptable accuracy
loss for many models, especially at 4-bit precision.

### The Outlier Problem

LLM weight and activation tensors have **heavy-tailed distributions** with large outliers.
Under block-wise quantization:
- The block scale is dominated by the largest element
- Smaller elements in the same block are severely under-represented
- A single outlier can waste most of the block's dynamic range

### How Rotations Help

An orthogonal rotation `H` applied to a vector preserves its L2 norm while redistributing
energy. Specifically, a **Hadamard transform** converts heavy-tailed (Laplace-like)
distributions into approximately **Gaussian distributions** where energy is spread evenly
across elements.

For a weight matrix W and activation matrix X:

```
Y = W · X = (W · H) · (H^T · X) = W_rot · X_rot
```

Since `H · H^T = I` (orthogonal), the output is unchanged. But now:
- `W_rot = W · H` has fewer outliers → quantizes better
- `X_rot = H^T · X` has fewer outliers → quantizes better

### Block-Diagonal Rotations

Full-matrix Hadamard transforms are O(n²) and impractical. Instead, **block-diagonal
Hadamard matrices** are used:

```
H_k = diag(H_k1, H_k2, ..., H_k(n/k))
```

where each `H_ki` is a k×k Hadamard matrix. This reduces cost to O(n·k) per vector.

Common block sizes: k ∈ {16, 32, 64, 128}.

### Optimal Block Size by Format

The interaction between rotation block size and quantization block size matters:

- **NVFP4 (block size 16)**: Had16 performs best. Larger rotations (Had32, Had64, Had128)
  can actually hurt because they spread information beyond the quantization block boundary,
  introducing inter-block error.
- **MXFP4 (block size 32)**: Had128 outperforms Had32. The larger quantization blocks
  benefit from stronger distribution normalization.

This is a key insight from the IST-DASLab "Bridging the Gap" paper: **rotation size should
be matched to the quantization group size for NVFP4**.

### Theoretical Foundation

For a Laplace-distributed (native) tensor, the "preservation rate" under absmax quantization
scales as:

```
R_Laplace(G) = Θ((log G)² · G^(-δ))
R_Normal(G) = Θ(√(log G) · G^(-δ²))
```

where G is the block size and δ < 1. Since δ² < δ, rotations hurt small G but help large G.
This explains the crossover between NVFP4 (G=16, rotation-sensitive) and MXFP4 (G=32,
rotation-friendly).

---

## 6. MR-GPTQ: Micro-Rotated GPTQ

MR-GPTQ (Micro-Rotated GPTQ) is the state-of-the-art post-training quantization method
for NVFP4, developed by IST-DASLab (Alistarh et al.). It combines three ingredients:

### Ingredient 1: MSE-Optimized Grids

Instead of using simple absmax scaling, MR-GPTQ solves an optimization problem to find
scales that minimize reconstruction error:

```
minimize:  Σ ||X̂_i - X_i||²
over:      s_tensor, s_block_1, ..., s_block_K

where:     X̂_i = s_tensor · s_block · Q_FP4(X_i / (s_tensor · s_block))
```

The optimization alternates between:
1. Fixing block scales, optimizing per-tensor scale
2. Fixing per-tensor scale, optimizing block scales

For NVFP4 without rotations, this consistently improves over absmax. For MXFP4 with
rotations, a single static value works well across layers.

### Ingredient 2: Static Activation Reordering

GPTQ benefits from processing columns in order of activation magnitude (largest first).
Dynamic reordering at inference time costs 10–20% throughput. MR-GPTQ applies reordering
**statically** during quantization:

1. Compute grid and scales in original column order
2. Shuffle columns by activation heuristic before GPTQ
3. Shuffle back after quantization

This preserves the microscaling block structure while getting GPTQ's accuracy benefits.

### Ingredient 3: Fused Online Micro-Rotations

The key innovation: block-wise Hadamard transforms are fused with quantization in a single
GPU kernel. For inference:

- **Weights**: Rotation is applied **offline** — `W_rot = W · H_k` is precomputed and
  stored. No runtime cost.
- **Activations**: Rotation is applied **online** — `X_rot = X · H_k` is computed on the
  fly via a lightweight fused kernel that performs rotation + quantization + scale
  computation in a single pass.

The overhead of online rotation is negligible because for block sizes k < 256, the
operation is memory-bound (not compute-bound). Any rotation matrix — Hadamard, DCT, or
arbitrary — can be applied at essentially the same cost.

### Results

On Llama-3.1-8B-Instruct with W4A4 NVFP4 quantization:
- RTN (no rotation): ~92% FP16 accuracy recovery
- GPTQ: ~95.7% recovery
- MR-GPTQ: ~95.8% recovery
- On 70B models: 98–99% recovery

---

## 7. QuTLASS: IST-DASLab CUDA Kernels

**QuTLASS** (CUTLASS-Powered Quantized BLAS) is the reference CUDA kernel library from
Dan Alistarh's lab at IST Austria. It provides high-performance kernels for NVFP4 and
MXFP4 on Blackwell GPUs.

**Repository**: [github.com/IST-DASLab/qutlass](https://github.com/IST-DASLab/qutlass)

### Architecture

```
qutlass/
├── csrc/
│   ├── fused_quantize_mx.cu    # Fused rotation + quantize + scale kernel
│   ├── gemm.cu                 # CUTLASS-backed matmul (large batches)
│   ├── gemm_ada.cu             # Prototype matmul (small batches, bs=1-32)
│   └── [backward kernels]      # QAT backward pass kernels
├── utils.py                    # Block-scale reordering (to_blocked)
├── benchmarks/
└── tests/
```

### Fused Quantization Kernel

The signature:

```python
aq, a_sf = qutlass.fusedQuantizeMx(a, h, method)
```

- **`a`**: Input tensor (BF16/FP16) to quantize
- **`h`**: Rotation matrix (Hadamard, DCT, identity, etc.) loaded at runtime
- **`method`**: `"quest"` (MSE-optimized) or `"abs_max"`
- **Returns**: `aq` (packed FP4 E2M1), `a_sf` (E4M3/E8M0 block scales)

The CUDA kernel (`fused_quantize_mx.cu`) performs in a single pass:
1. Load input tile from global memory
2. Apply rotation: multiply by H (block-diagonal, loaded into shared memory)
3. Compute block statistics (max, MSE grid search)
4. Compute block scale and quantize to E4M3
5. Quantize each element to E2M1 via round-to-nearest
6. Pack two FP4 values per byte
7. Write packed data and scales to global memory

Supported rotation sizes: 16, 32, 64, 128. The rotation matrix is loaded at runtime, so
**any orthogonal transform** works without recompilation.

### Matmul Kernels

**Large-batch CUTLASS kernel** (bs > 32):

```python
output = qutlass.matmul_mxf4_bf16_tn(aq, bq, a_sf, b_sf, alpha)
```

Requires block-scale reordering via `qutlass.to_blocked()` to match the hardware swizzle
format expected by `tcgen05.mma`.

**Small-batch prototype kernel** (bs = 1–32):

```python
output = qutlass.matmul_ada_mxf4_bf16_tn(aq, bq, a_sf, b_sf, alpha)
```

No block reordering needed; uses a custom CUDA kernel optimized for decode-time single-token
inference.

**NVFP4 variants** have identical signatures: `matmul_nvf4_bf16_tn`, etc.

### Performance

- Layer-wise speedup: 3.6x on B200, 6x on RTX 5090 (vs BF16)
- End-to-end inference: 2.2x on B200, 4x on RTX 5090
- Near-ideal throughput with negligible rotation overhead
- MXFP4 achieves ~15% higher throughput than NVFP4 due to power-of-two scales and larger
  block size reducing overhead

### Requirements

- NVIDIA Blackwell GPU (sm_100a or sm_120a)
- CUDA 12.8+
- PyTorch 2.8+
- CUTLASS 4.2.1

---

## 8. FP-Quant: End-to-End Quantization Pipeline

**FP-Quant** is the end-to-end model quantization and export tool from IST-DASLab, built
on top of QuTLASS.

**Repository**: [github.com/IST-DASLab/FP-Quant](https://github.com/IST-DASLab/FP-Quant)

### Quantization Workflow

```bash
python model_quant.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --format nvfp \
    --w_bits 4 --a_bits 4 \
    --w_group_size 16 --a_group_size 16 \
    --gptq \
    --transform_class hadamard \
    --hadamard_group_size 16 \
    --dataset_name_or_path fineweb-edu \
    --num_sequences 128 \
    --export_quantized_model realquant
```

### Supported Transforms

FP-Quant supports six rotation/transform types:
1. **`identity`** — No transformation
2. **`hadamard`** — Hadamard rotation (recommended for NVFP4)
3. **`dct`** — Discrete cosine transform
4. **`dst`** — Discrete sine transform
5. **`fast_food`** — Structured random projection
6. **`gsr`** — Grouped sequency-aligned transform

### Export Modes

1. **`realquant`**: Exports with QuTLASS kernels for actual FP4 computation. Requires
   Blackwell GPU at inference time.
2. **`pseudoquant`**: Exports with Triton "fake-quantized" kernels. Runs on any GPU but
   does not achieve FP4 speedups.

### Integration

- **HuggingFace Transformers**: Load quantized models with `FPQuantConfig`
- **vLLM**: Tensor-parallel inference with quantized models
- **Pre-quantized models**: Available on HuggingFace under the "MR-GPTQ" collection

```python
from transformers import AutoModelForCausalLM, FPQuantConfig

model = AutoModelForCausalLM.from_pretrained(
    "IST-DASLab/Llama-3.1-8B-Instruct-MR-GPTQ-nvfp",
    quantization_config=FPQuantConfig(forward_dtype="nvfp4"),
    device_map="auto"
)
```

---

## 9. Four Over Six: Adaptive Block Scaling

**Four Over Six** (MIT HAN Lab) proposes mixed FP4/FP6 quantization where each block
independently chooses between 4-bit and 6-bit representation based on quantization error.

**Repository**: [github.com/mit-han-lab/fouroversix](https://github.com/mit-han-lab/fouroversix)
**Paper**: [arxiv.org/abs/2512.02010](https://arxiv.org/abs/2512.02010)

### Key Insight

For many blocks, scaling to a maximum of 4 (2-bit exponent range) instead of 6 (full E2M1
range) introduces less quantization error. By adaptively selecting the "clipping" point per
block, the method achieves better accuracy than uniform NVFP4.

### CUDA Implementation

The algorithm is implemented as a register-resident CUDA kernel:

1. Quantize block to FP4 using standard NVFP4 procedure
2. Dequantize back to FP16 using `cvt` instructions
3. Compute per-block MSE error
4. Repeat with alternative scaling (scale-to-4 instead of scale-to-6)
5. Select the lower-error variant per block
6. Store a 1-bit flag per block indicating the choice

All intermediate values (quantized, dequantized, errors) stay in the **register file**,
keeping overhead under 15%.

### PTX Instructions Used

- **`cvt.rn`**: Convert with round-to-nearest — used for FP32→E4M3 scale quantization
  and for packed FP4 conversion/deconversion
- The `cvt` family handles both quantization (FP32/FP16 → packed FP4) and dequantization
  (packed FP4 → FP16) needed for error calculation

### Three Backend Implementations

1. **CUDA**: Highest performance, requires Blackwell (sm_100/sm_120)
2. **Triton**: Full feature set including stochastic rounding, Hadamard transforms, 2D
   block scaling, transposed inputs
3. **PyTorch**: Reference implementation for testing/education, runs on any GPU

### API

```python
from fouroversix import quantize_to_fp4, fp4_matmul, quantize_model

# Tensor-level quantization
q_tensor = quantize_to_fp4(tensor, scale_rule="adaptive_4_6")

# Matrix multiplication with quantized operands
output = fp4_matmul(q_a, q_b)

# Model-level quantization
quantize_model(model, ModelQuantizationConfig(...))
```

---

## 10. RaZeR: Redundant Zero Remapping

**RaZeR** exploits redundancies in the NVFP4 format to add extra quantization values
without increasing memory footprint.

**Repository**: [github.com/abdelfattah-lab/NVFP4-RaZeR](https://github.com/abdelfattah-lab/NVFP4-RaZeR)
**Paper**: [arxiv.org/abs/2501.04052](https://arxiv.org/abs/2501.04052)

### Two Redundancies

1. **Positive/negative zero in E2M1**: Bit patterns `0000` (+0.0) and `1000` (-0.0) both
   represent zero. One is redundant.
2. **Sign bit in E4M3 block scale**: Block scales are always positive, so the sign bit is
   wasted. Furthermore, LLM weights tolerate E3M3 (6-bit effective scale), freeing 2 bits.

### Mechanism

RaZeR repurposes these redundant bits to encode **special values** beyond the standard
E2M1 set:

- **Weights**: 2 freed bits → 4 possible special values (2-bit selector)
- **Activations**: 1 freed bit → 2 possible special values (1-bit selector)

The special values are stored as 4-bit offsets added to 6.0:
```
special_value = ±(offset + 6.0)
```
where offset is encoded with 1 sign bit, 2 integer bits, 1 fraction bit (range [-3.5, 3.5]).

Empirically, **±5** is optimal across most LLMs, sitting at the midpoint between FP4's
largest values (±4 and ±6). This fills a key gap in the E2M1 representation.

### Hardware Modification

RaZeR proposes a modified tensor core decoder:
1. Compare incoming FP4 value against binary zero
2. On match: route to special value path
3. Selector bit chooses which offset register
4. Add offset to 6.0, apply sign
5. Feed reconstructed value to MAC array

Silicon overhead: 3.7% area, 13.5% decoder power (0.37%/1.35% at chip level).

### Results

Compared to baseline NVFP4:
- 34.6% perplexity loss reduction (weight-only)
- 31.2% reduction (weight + activation)
- 4.47% improvement on GSM8K reasoning for Llama-3.1-8B

---

## 11. Quartet: Native FP4 Training

**Quartet** and **Quartet II** from IST-DASLab enable full FP4 training (not just inference)
using NVFP4.

**Repository**: [github.com/IST-DASLab/Quartet](https://github.com/IST-DASLab/Quartet)

### Key Training Techniques

1. **Random Hadamard Transforms**: Applied to all GEMM inputs (forward and backward) to
   normalize distributions before quantization.

2. **Stochastic Rounding**: Gradients are rounded probabilistically:
   ```
   P(round_up) = (x - floor(x)) / (ceil(x) - floor(x))
   P(round_down) = 1 - P(round_up)
   ```
   This eliminates systematic rounding bias that would accumulate over training steps.

3. **2D Block Scaling**: For weight matrices, a single scale factor covers a 16×16 block
   (inspired by DeepSeek-v3), providing finer granularity than 1D row-wise scaling.

4. **MS-EDEN** (Quartet II): A novel unbiased quantization routine for micro-scaled
   formats that achieves 2x lower quantization error than stochastic rounding.

### Integration

Quartet kernels are released as part of the QuTLASS library. A complete training pipeline
is available via:
- `main_setup.sh` for pseudo-quantized MXFP4 pre-training
- HuggingFace Transformers integration (PR #41897)
- Nanochat-QAT training recipes

### NVIDIA Transformer Engine Integration

```python
from transformer_engine.common.recipe import NVFP4BlockScaling
import transformer_engine.pytorch as te

nvfp4_recipe = NVFP4BlockScaling()
my_linear = te.Linear(768, 768)

with te.autocast(recipe=nvfp4_recipe):
    output = my_linear(input)
```

---

## 12. CUDA-Level Implementation Details

This section details how NVFP4 quantization is implemented at the CUDA kernel level.

### Quantization Kernel Pseudocode

```cuda
// Per-block quantization: 16 elements → 8 packed bytes + 1 E4M3 scale
__global__ void quantize_nvfp4_kernel(
    const half* __restrict__ input,    // [N] input tensor (pre-divided by tensor scale)
    uint8_t* __restrict__ output,      // [N/2] packed FP4 output
    fp8_e4m3* __restrict__ scales,     // [N/16] per-block scales
    int N
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = block_idx * 16;
    if (start >= N) return;

    // Step 1: Load 16 elements
    half vals[16];
    for (int i = 0; i < 16; i++)
        vals[i] = input[start + i];

    // Step 2: Compute block scale (absmax method)
    float amax = 0.0f;
    for (int i = 0; i < 16; i++)
        amax = fmaxf(amax, fabsf(__half2float(vals[i])));

    // Scale so that amax maps to 6.0 (max E2M1 magnitude)
    float scale = amax / 6.0f;

    // Quantize scale to E4M3 using cvt.rn
    fp8_e4m3 scale_fp8 = cvt_rn_fp32_to_e4m3(scale);
    float scale_dequant = cvt_e4m3_to_fp32(scale_fp8);
    scales[block_idx] = scale_fp8;

    // Step 3: Quantize each element to E2M1
    uint8_t packed[8];
    for (int i = 0; i < 16; i += 2) {
        float v0 = __half2float(vals[i]) / scale_dequant;
        float v1 = __half2float(vals[i+1]) / scale_dequant;

        uint8_t q0 = round_to_nearest_e2m1(v0);  // 4-bit value
        uint8_t q1 = round_to_nearest_e2m1(v1);  // 4-bit value

        packed[i/2] = (q1 << 4) | (q0 & 0x0F);
    }

    // Step 4: Store packed output
    for (int i = 0; i < 8; i++)
        output[block_idx * 8 + i] = packed[i];
}
```

### Round-to-Nearest E2M1

```cuda
__device__ uint8_t round_to_nearest_e2m1(float x) {
    // E2M1 representable positive magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    // Decision boundaries (midpoints): 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    float ax = fabsf(x);
    uint8_t sign = (x < 0) ? 0x8 : 0x0;
    uint8_t code;

    if      (ax < 0.25f)  code = 0x0;  // 0.0
    else if (ax < 0.75f)  code = 0x1;  // 0.5
    else if (ax < 1.25f)  code = 0x2;  // 1.0
    else if (ax < 1.75f)  code = 0x3;  // 1.5
    else if (ax < 2.50f)  code = 0x4;  // 2.0
    else if (ax < 3.50f)  code = 0x5;  // 3.0
    else if (ax < 5.00f)  code = 0x6;  // 4.0
    else                  code = 0x7;  // 6.0

    return sign | code;
}
```

### Dequantization

```cuda
__device__ float dequantize_e2m1(uint8_t code) {
    // Lookup table for E2M1 magnitudes
    static const float LUT[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float sign = (code & 0x8) ? -1.0f : 1.0f;
    return sign * LUT[code & 0x7];
}

// Full dequantization: value = e2m1_value * block_scale * tensor_scale
float x = dequantize_e2m1(code) * cvt_e4m3_to_fp32(block_scale) * tensor_scale;
```

### Fused Rotation + Quantization Kernel (QuTLASS Pattern)

```cuda
// Simplified structure of the fused kernel in qutlass/csrc/fused_quantize_mx.cu
__global__ void fused_rotate_quantize_kernel(
    const half* __restrict__ input,      // [M, K]
    const half* __restrict__ rotation,   // [rot_size, rot_size] loaded at runtime
    uint8_t* __restrict__ output_fp4,    // [M, K/2] packed
    fp8_e4m3* __restrict__ output_sf,    // [M, K/16] block scales
    int M, int K, int rot_size
) {
    // 1. Load tile of input into shared memory
    __shared__ half tile[TILE_M][TILE_K];
    load_tile(input, tile, ...);

    // 2. Load rotation matrix into shared memory
    __shared__ half rot[MAX_ROT_SIZE][MAX_ROT_SIZE];
    load_rotation(rotation, rot, rot_size);

    // 3. Apply block-diagonal rotation in-place
    //    Each rot_size-element chunk is multiplied by the rotation matrix
    for (int chunk = 0; chunk < TILE_K; chunk += rot_size) {
        // Matrix-vector multiply: tile[row][chunk:chunk+rot_size] *= rot
        apply_rotation_block(tile, rot, row, chunk, rot_size);
    }

    // 4. Quantize rotated values (per 16-element blocks)
    for (int block = 0; block < TILE_K; block += 16) {
        float amax = compute_block_amax(tile, row, block);
        fp8_e4m3 scale = quantize_scale(amax / 6.0f);
        float scale_f = dequantize_scale(scale);

        for (int i = 0; i < 16; i += 2) {
            uint8_t q0 = round_to_nearest_e2m1(tile[row][block+i] / scale_f);
            uint8_t q1 = round_to_nearest_e2m1(tile[row][block+i+1] / scale_f);
            store_packed(output_fp4, q1, q0);
        }
        store_scale(output_sf, scale);
    }
}
```

### PTX-Level Operations

Key PTX instructions used in NVFP4 kernels:

```asm
// FP32 to E4M3 scale conversion (round to nearest)
cvt.rn.satfinite.e4m3x2.f32  result, src0, src1;

// Packed FP4 conversion (two values at once)
cvt.rn.satfinite.e2m1x2.f32  result, src0, src1;
// or:  cvt.rn.satfinite.e2m1x2.f16x2  result, src;

// FP4 to FP16 dequantization
cvt.f16x2.e2m1x2  result, src;

// Tensor Core MMA with block-scaled FP4
tcgen05.mma.cta_group::1.nvf4mxf4
    [d_tmem], a_desc, b_desc, idesc, pred_disable, pred_enable;
```

The `cvt.rn.satfinite.e2m1x2` instruction converts two FP32 values to packed E2M1 in a
single instruction, with round-to-nearest-even and saturation to the representable range.

---

## 13. Software Ecosystem and Deployment

### Quantization Tools

| Tool              | Purpose                        | Rotation Support | GPTQ | Export Format |
|-------------------|--------------------------------|------------------|------|---------------|
| FP-Quant          | Post-training quantization     | Yes (6 types)    | Yes  | QuTLASS/Triton|
| LLM Compressor    | NVIDIA's PTQ tool              | SmoothQuant      | Yes  | TensorRT-LLM  |
| TensorRT ModelOpt | NVIDIA optimization toolkit    | Limited          | Yes  | TensorRT      |
| Four Over Six     | Adaptive 4/6 quantization      | Hadamard         | No   | Custom        |

### Inference Runtimes

| Runtime          | NVFP4 Support | Rotation Support | Backend           |
|------------------|---------------|------------------|-------------------|
| vLLM             | Yes           | Via FP-Quant     | QuTLASS/FlashInfer|
| TensorRT-LLM     | Yes           | Limited          | Native CUTLASS    |
| Transformer Engine| Yes          | Hadamard         | Native kernels    |
| SGLang           | Planned       | TBD              | TBD               |

### Pre-Quantized Models

IST-DASLab publishes pre-quantized models on HuggingFace:
- Llama-3.1-8B/70B-Instruct in NVFP4 and MXFP4 formats
- Quantized with MR-GPTQ for optimal accuracy
- Compatible with vLLM and Transformers

---

## 14. bitsandbytes NVFP4 Implementation

This section documents the actual NVFP4 implementation in bitsandbytes, targeting
SM_120 (Blackwell consumer GPUs like RTX PRO 6000).

### Architecture

The GEMM and fused quantize use **CUTLASS** (vendored from QuTLASS, compiled into
the shared library). Legacy quantize/dequantize/rotation kernels use raw CUDA with
inline PTX and serve as fallback for non-Blackwell GPUs.

```
csrc/
├── kernels.cu                     # Quantize/dequantize/Hadamard kernels (fallback)
├── kernels_nvfp4_sm120.cu         # Legacy hand-written GEMM (SM_120)
├── qutlass/gemm_nvfp4_sm120.cu    # CUTLASS-based GEMM (SM_120, from QuTLASS)
├── qutlass/fused_quantize_nv.cu   # CUTLASS-based fused quantize (SM_80+, from QuTLASS)
├── qutlass/include/               # Vendored CUTLASS extensions for quantize epilogue
├── qutlass/scale_reorder.cu       # Scale factor reordering for CUTLASS
├── ops.cu                         # Host-side launchers
└── pythonInterface.cpp            # extern "C" symbols for ctypes

third_party/cutlass/               # CUTLASS headers (submodule, header-only)

bitsandbytes/
├── _ops.py                    # torch.library op definitions
├── backends/cuda/ops.py       # CUDA backend dispatch (ctypes → C)
├── functional.py              # NVFP4QuantState, quantize/dequantize/gemm
└── nn/modules.py              # LinearNVFP4 module
```

### Key Design Decisions

1. **SM_120 consumer GPUs only**: CUTLASS GEMM uses `wgmma` (SM_120 Blackwell path).
   SM_100 datacenter uses `tcgen05.mma` with TMEM (separate implementation, future work).
2. **CUTLASS GEMM, owned quantization**: The GEMM is vendored from QuTLASS with
   PyTorch dependencies removed. Quantize/dequantize/rotation kernels are owned code.
3. **Block size fixed at 16**: Hardware requirement for NVFP4 (different from existing
   bitsandbytes variable block sizes of 32-4096).
4. **NVFP4=3 in DataType_t enum**: Separate from existing FP4=1 (custom bitsandbytes
   format, not E2M1). No breaking changes to existing API.
5. **Two-level scaling**: E4M3 block scales per 16 elements + FP32 tensor scale.
6. **Hadamard rotation always on**: Randomized Hadamard rotation is always applied.
   With the CUTLASS fused quantize, the rotation is applied via the B matrix in the
   GEMM at zero additional cost.
7. **CUTLASS fused quantize**: Quantization formulated as a GEMM (SM_80 CUTLASS 2.x).
   Each group of 16 elements becomes a GEMM row; B is the randomized Hadamard matrix.
   Falls back to the hand-written kernel on non-Blackwell builds.
8. **Scale reordering at quantize time**: CUTLASS expects block-scaled swizzled layout;
   computed once at quantization and stored in `NVFP4QuantState.block_scales_blocked`.
8. **BF16 output from CUTLASS**: Tensor scales folded into CUTLASS epilogue alpha;
   result converted to FP32 in Python dispatch for API compatibility.

### PTX Instruction

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
    .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

- MMA tile: m16 × n8 × k64
- A: 4× uint32 registers (32 packed E2M1 nibbles each)
- B: 2× uint32 registers
- C/D: 4× float registers (accumulator)
- SFA/SFB: 1× uint32 each (4 packed UE4M3 bytes)
- Requires `-gencode=arch=compute_120a,code=sm_120a` (the `a` suffix is critical)

### GEMM Kernel Optimizations

The GEMM kernel (`kGemmNVFP4_smem`) evolved through several optimization stages:

| Version | 4K×4K TFLOPS | Key Optimization |
|---------|-------------|-----------------|
| v1 | 18 | Correctness-first, per-nibble global loads |
| v2 | 111 | Vectorized uint32/uint4 bulk loads |
| v3 | 225 | Shared memory tiling (32×128 block tile, 8 warps) |
| v4 | 239 | Register-based load/compute pipeline |
| v5 | 240 | Auto split-K for small-batch GPU occupancy |

Final kernel features:
- **32×128 block tile**: 2 M-warps × 4 N-warps × 4 N-tiles/warp = 8 warps (256 threads)
- **Shared memory tiling**: 5760 bytes per K-step (A: 1024, B: 4096, SFA: 128, SFB: 512)
- **Register pipeline**: Issue global loads → compute MMA → sync → write to smem
- **Auto split-K**: Two-tier heuristic fills GPU for small-batch LLM inference
- **launch_bounds(256, 4)**: 4 blocks/SM for maximum occupancy

### Performance Results (RTX PRO 6000)

| Shape | NVFP4 TFLOPS | cuBLAS FP16 TFLOPS | Speedup |
|-------|-------------|-------------------|---------|
| 1×4096×4096 | 4.1 | 3.3 | **1.25x** |
| 8×4096×4096 | 26.2 | 24.9 | **1.05x** |
| 32×4096×4096 | 87.8 | 86.9 | **1.01x** |
| 32×4096×11008 | 156.5 | 127.7 | **1.23x** |
| 128×4096×4096 | 174.6 | 232.7 | 0.75x |
| 4096×4096×4096 | 239.7 | 359.5 | 0.67x |

For LLM inference (bs=1-32), the NVFP4 GEMM matches or exceeds cuBLAS FP16.
Memory compression: **3.6x** (FP4 + scales vs FP16).

### Quantization Error

- E2M1 round-trip on standard normal data: mean abs error ~0.074
- Hadamard rotation kurtosis reduction: 5.22 → 3.03 (Gaussian target: 3.0)
- GEMM matches dequant→torch.matmul reference: 0.000000 relative error
- LinearNVFP4 vs FP32 Linear: ~13.5% relative error (expected for FP4)

### Python API

```python
import bitsandbytes.functional as F
from bitsandbytes.nn import LinearNVFP4

# Quantize/dequantize
packed, state = F.quantize_nvfp4(tensor, tensor_scale)
recovered = F.dequantize_nvfp4(packed, state)

# GEMM
output = F.gemm_nvfp4(A_data, A_state, B_data, B_state)

# Linear module
layer = LinearNVFP4(4096, 11008)
output = layer(input)  # weight quantized lazily on first forward
```

### Memory Layout

```
Quantized tensor storage:
├── data:        [N/2] bytes     (packed FP4, 2 values per byte)
├── block_scales: [N/16] bytes   (E4M3 FP8, one per 16-element block)
└── tensor_scale: [1] float32    (per-tensor global scale)
```

### Future Optimizations

- **cp.async double buffering**: Overlap global→smem loads with MMA compute for large M
- **SM_100 datacenter kernel**: Uses tcgen05.mma with TMEM (separate implementation)
- **MR-GPTQ integration**: MSE grid search, static reordering, GPTQ pipeline
- **LoRA fusion**: FP16 LoRA adapters with FP4 base in GEMM epilogue

---

## 15. References

### Papers

1. **Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization**
   Egiazarian et al. (IST-DASLab), 2025. [arXiv:2509.23202](https://arxiv.org/abs/2509.23202)
   — MR-GPTQ algorithm, QuTLASS kernels, comprehensive NVFP4 vs MXFP4 analysis.

2. **Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling**
   MIT HAN Lab, 2025. [arXiv:2512.02010](https://arxiv.org/abs/2512.02010)
   — Adaptive FP4/FP6 block selection, CUDA kernel with register-resident error computation.

3. **RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping**
   Chen et al., 2025. [arXiv:2501.04052](https://arxiv.org/abs/2501.04052)
   — Exploiting format redundancies for extra quantization values.

4. **Quartet: Native FP4 Training Can Be Optimal for Large Language Models**
   IST-DASLab, NeurIPS 2025. [arXiv:2505.14669](https://arxiv.org/abs/2505.14669)
   — FP4 training with stochastic rounding and Hadamard transforms.

5. **Quartet II: Accurate LLM Pre-Training in NVFP4**
   IST-DASLab, 2026. [arXiv:2601.22813](https://arxiv.org/abs/2601.22813)
   — MS-EDEN unbiased quantization for NVFP4 training.

6. **HALO: Hadamard-Assisted Low-Precision Optimization**
   IST-DASLab, 2025. [arXiv:2501.02625](https://arxiv.org/abs/2501.02625)
   — Hadamard transforms for fine-tuning with INT8/FP6.

### Repositories (IST-DASLab / Dan Alistarh)

- **QuTLASS**: [github.com/IST-DASLab/qutlass](https://github.com/IST-DASLab/qutlass)
  — CUTLASS-powered NVFP4/MXFP4 CUDA kernels for Blackwell.
- **FP-Quant**: [github.com/IST-DASLab/FP-Quant](https://github.com/IST-DASLab/FP-Quant)
  — End-to-end quantization pipeline with model export.
- **Quartet**: [github.com/IST-DASLab/Quartet](https://github.com/IST-DASLab/Quartet)
  — Native FP4 training implementation.
- **HALO**: [github.com/IST-DASLab/HALO](https://github.com/IST-DASLab/HALO)
  — Hadamard-assisted low-precision optimization for fine-tuning.
- **WUSH**: [github.com/IST-DASLab/WUSH](https://github.com/IST-DASLab/WUSH)
  — Weight update with stochastic Hadamard transforms.

### Other Repositories

- **Four Over Six**: [github.com/mit-han-lab/fouroversix](https://github.com/mit-han-lab/fouroversix)
  — MIT HAN Lab adaptive 4/6 quantization.
- **RaZeR**: [github.com/abdelfattah-lab/NVFP4-RaZeR](https://github.com/abdelfattah-lab/NVFP4-RaZeR)
  — Redundant zero remapping for NVFP4.

### NVIDIA Resources

- **NVFP4 Inference Blog**: [developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- **NVFP4 Training Blog**: [developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
- **CUTLASS Blackwell Docs**: [docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html](https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html)
- **Transformer Engine FP4 Guide**: [docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- **Triton Block-Scaled Matmul Tutorial**: [triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
