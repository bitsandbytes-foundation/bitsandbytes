# bitsandbytes Architecture Guide

This document provides a comprehensive architecture reference for agents reviewing pull requests
or writing code for the bitsandbytes library. It describes every layer of the codebase, how data
flows through the system, how backends are dispatched, and how the build system produces native
libraries. Read this before reviewing any PR — it replaces the need to read the whole codebase.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Layout](#2-directory-layout)
3. [Layer Architecture](#3-layer-architecture)
4. [The Op Registry (`_ops.py`)](#4-the-op-registry-_opspy)
5. [Backend Dispatch System](#5-backend-dispatch-system)
6. [Native Library Loading (`cextension.py`)](#6-native-library-loading-cextensionpy)
7. [The Functional Layer (`functional.py`)](#7-the-functional-layer-functionalpy)
8. [Quantization Data Types and QuantState](#8-quantization-data-types-and-quantstate)
9. [Autograd Functions (`autograd/_functions.py`)](#9-autograd-functions-autograd_functionspy)
10. [Neural Network Modules (`nn/modules.py`)](#10-neural-network-modules-nnmodulespy)
11. [Optimizer System (`optim/`)](#11-optimizer-system-optim)
12. [CUDA/C++ Native Code (`csrc/`)](#12-cudac-native-code-csrc)
13. [Build System (`CMakeLists.txt`)](#13-build-system-cmakeliststxt)
14. [Data Flow: End-to-End Traces](#14-data-flow-end-to-end-traces)
15. [Key Design Patterns](#15-key-design-patterns)
16. [Cross-Cutting Concerns](#16-cross-cutting-concerns)
17. [Test Structure](#17-test-structure)

---

## 1. Project Overview

bitsandbytes is a library for quantized operations on neural network models. It provides:

- **8-bit matrix multiplication** (LLM.int8() algorithm) for inference and training
- **4-bit quantization** (QLoRA / NF4 / FP4) for memory-efficient inference and fine-tuning
- **8-bit optimizers** (Adam, AdamW, SGD, Lion, AdEMAMix, etc.) that compress optimizer state
- **Quantized `nn.Module` replacements** (`Linear8bitLt`, `Linear4bit`, `Embedding4bit`, etc.)

The library supports multiple backends: CUDA (primary), ROCm/HIP, CPU, XPU (Intel), MPS (Apple
Silicon), HPU (Gaudi), and Triton. CUDA is by far the most complete and optimized backend.

---

## 2. Directory Layout

```
bitsandbytes/
├── __init__.py              # Top-level exports, re-exports from functional, autograd, nn
├── _ops.py                  # torch.library.define() op schemas + register_fake + register_kernel helper
├── functional.py            # Stateless Python API: quantize, dequantize, matmul, optimizer updates
├── cextension.py            # Native library loader (ctypes), detects CUDA/ROCm/CPU
├── cuda_specs.py            # CUDA version detection utilities
├── consts.py                # Constants (PACKAGE_DIR, DYNAMIC_LIBRARY_SUFFIX)
├── utils.py                 # OutlierTracer, weight format mappings, sync_gpu
│
├── autograd/
│   ├── __init__.py
│   └── _functions.py        # MatMul8bitLt, MatMul8bitFp, MatMul4Bit autograd functions
│
├── nn/
│   ├── __init__.py           # Re-exports all nn modules
│   ├── modules.py            # Linear8bitLt, Linear4bit, Int8Params, Params4bit, Embeddings
│   └── triton_based_modules.py  # SwitchBackLinear (triton-based)
│
├── optim/
│   ├── __init__.py           # Re-exports all optimizer classes
│   ├── optimizer.py          # Base classes: Optimizer8bit, Optimizer1State, Optimizer2State, GlobalOptimManager
│   ├── adam.py               # Adam, Adam8bit, Adam32bit, PagedAdam, PagedAdam8bit, PagedAdam32bit
│   ├── adamw.py              # Same pattern for AdamW
│   ├── ademamix.py           # AdEMAMix variants
│   ├── lion.py               # Lion variants
│   ├── sgd.py                # SGD variants
│   ├── rmsprop.py            # RMSprop variants
│   ├── adagrad.py            # Adagrad variants
│   ├── lamb.py               # LAMB variants
│   └── lars.py               # LARS variants + PytorchLARS
│
├── backends/
│   ├── __init__.py           # Empty (backends auto-register via imports)
│   ├── utils.py              # Shared: NF4/FP4 lookup tables (CODE dict), triton_available flag, Gaudi version
│   ├── default/
│   │   └── ops.py            # Pure PyTorch fallback implementations (all ops)
│   ├── cuda/
│   │   └── ops.py            # CUDA implementations via ctypes calls to lib.*
│   ├── cpu/
│   │   └── ops.py            # CPU-optimized implementations (AVX512, torch._int_mm)
│   ├── triton/
│   │   ├── ops.py            # Triton kernel registrations
│   │   ├── kernels_4bit.py   # Triton 4-bit dequant kernels
│   │   ├── kernels_8bit_quant.py  # Triton 8-bit quant kernels
│   │   └── kernels_optim.py  # Triton optimizer kernels
│   ├── xpu/                  # Intel XPU backend
│   └── hpu/                  # Habana Gaudi backend
│
csrc/
├── pythonInterface.cpp       # C++ wrapper: unmangled functions callable via ctypes
├── ops.cu                    # CUDA op dispatch: launches kernels with grid/block configs
├── kernels.cu                # CUDA kernel implementations (__global__ functions)
├── ops.cuh                   # CUDA op declarations + error checking macros + context classes
├── kernels.cuh               # CUDA kernel declarations
├── common.cuh                # Compute capability macros (BNB_CC_VOLTA, etc.)
├── common.h                  # Shared C header
├── cpu_ops.cpp               # CPU-native C++ kernels (blockwise quant, etc.)
├── cpu_ops.h                 # CPU op declarations
├── ops.hip / kernels.hip     # ROCm/HIP equivalents
├── ops_hip.cuh / kernels_hip.cuh / common_hip.cuh
├── mps_ops.mm                # Apple MPS Objective-C++ ops
├── mps_kernels.metal         # Apple Metal shader kernels
├── xpu_ops.cpp / xpu_kernels.cpp  # Intel XPU ops
└── xpu_ops.h / xpu_kernels.h

CMakeLists.txt                # Build system: compiles csrc/ into libbitsandbytes_*.so
pyproject.toml                # Package metadata, build config

tests/
├── conftest.py               # Shared fixtures (device parametrize, etc.)
├── helpers.py                # Test utility functions
├── test_functional.py        # Tests for functional.py ops
├── test_ops.py               # Tests for torch.ops.bitsandbytes.* dispatch
├── test_linear4bit.py        # Tests for Linear4bit / Params4bit
├── test_linear8bitlt.py      # Tests for Linear8bitLt / Int8Params
├── test_modules.py           # Tests for nn modules
├── test_autograd.py          # Tests for autograd correctness
├── test_optim.py             # Tests for all optimizers
├── test_triton.py            # Tests for triton kernels
├── test_deprecated.py        # Tests that deprecated APIs warn/error properly
├── test_parametrize.py       # Tests for weight parametrization
├── test_generation.py        # Integration: text generation with quantized models
└── test_cuda_setup_evaluator.py  # Tests for CUDA detection/setup
```

---

## 3. Layer Architecture

The codebase is organized into **five distinct layers**, from lowest to highest:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 5: nn.Modules (Linear4bit, Linear8bitLt, Embedding4bit)     │
│  → User-facing PyTorch modules that wrap everything below           │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 4: Autograd Functions (MatMul4Bit, MatMul8bitLt)            │
│  → Custom backward passes for quantized matmul                     │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 3: Functional API (functional.py)                           │
│  → Stateless Python functions: quantize_4bit, dequantize_4bit,     │
│    optimizer_update_32bit, etc. Calls torch.ops.bitsandbytes.*     │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 2: Op Registry (_ops.py) + Backend Dispatch                 │
│  → torch.library.define() schemas, register_fake(),                │
│    register_kernel() per device (cuda, cpu, default, triton, etc.) │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 1: Native Kernels (csrc/)                                   │
│  → CUDA kernels, ctypes interface, cuBLAS calls                    │
│  → Loaded via cextension.py → ct.cdll.LoadLibrary()               │
└──────────────────────────────────────────────────────────────────────┘
```

**Important**: Not all paths go through all layers. For example:
- Optimizers: `optim/*.py` → `functional.py` → `torch.ops.bitsandbytes.*` → backend kernel
- Direct quantization: User calls `bnb.functional.quantize_4bit()` → same path but no nn.Module

---

## 4. The Op Registry (`_ops.py`)

This is the central contract layer. Every operation in bitsandbytes is defined here as a
`torch.library` op, which enables:
- **torch.compile** compatibility (via `register_fake` providing shape/dtype metadata)
- **Multi-backend dispatch** (each backend registers its kernel for the same op name)
- **Consistent API** across CUDA, CPU, Triton, etc.

### How it works

```python
# _ops.py defines ops and their schemas:
torch.library.define("bitsandbytes::quantize_4bit", "(Tensor A, int blocksize, str quant_type, ScalarType quant_storage) -> (Tensor, Tensor)")

# register_fake provides shape inference for torch.compile:
@torch.library.register_fake("bitsandbytes::quantize_4bit")
def _(A, blocksize, quant_type, quant_storage):
    # Returns tensors with correct shapes but no real data
    ...

# Each backend registers its implementation:
# In backends/cuda/ops.py:
@register_kernel("bitsandbytes::quantize_4bit", "cuda")
def _(A, blocksize, quant_type, quant_storage):
    # Actual CUDA implementation via ctypes
    ...

# In backends/default/ops.py:
@register_kernel("bitsandbytes::quantize_4bit", "default")
def _(A, blocksize, quant_type, quant_storage):
    # Pure PyTorch fallback
    ...
```

### `register_kernel` helper

The `register_kernel` function in `_ops.py` is a wrapper around
`torch.library.register_kernel`. It handles the `"default"` dispatch key specially — for
`"default"`, it uses `torch.library.impl` with `"default"` which serves as a fallback when no
device-specific kernel is registered for the given device type.

### Current op catalog

All ops are defined with the namespace `bitsandbytes::`:

**Quantization ops:**
- `quantize_blockwise` — 8-bit blockwise quantization (codebook-based)
- `dequantize_blockwise` / `dequantize_blockwise.out` — inverse
- `quantize_4bit` — 4-bit quantization (NF4 or FP4)
- `dequantize_4bit` / `dequantize_4bit.out` — inverse

**Int8 matmul ops:**
- `int8_linear_matmul` / `int8_linear_matmul.out` — int8 x int8 → int32 via cuBLASLt
- `int8_mm_dequant` — dequantize int32 matmul result to fp16/bf16
- `int8_scaled_mm` — fused int8 matmul + dequant (composes the above two)
- `int8_vectorwise_quant` — row-wise int8 quantization with optional outlier detection
- `int8_vectorwise_dequant` — inverse
- `int8_double_quant` — both row-wise and column-wise quantization (for LLM.int8())
- `int8_mixed_scaled_mm` — int8 matmul with outlier decomposition (mixed-precision)

**4-bit inference ops:**
- `gemv_4bit` / `gemv_4bit.out` — fused 4-bit dequant + matmul (single-batch inference)

**Optimizer ops:**
- `optimizer_update_32bit` — 32-bit optimizer step (Adam, Lion, SGD, etc.)
- `optimizer_update_8bit_blockwise` — 8-bit blockwise optimizer step
- `optimizer_update_8bit` — 8-bit non-blockwise optimizer step (legacy)

**Utility ops:**
- `percentile_clipping` — adaptive gradient clipping by percentile

---

## 5. Backend Dispatch System

### How backends are loaded

When Python imports `bitsandbytes`, the following happens:

1. `__init__.py` imports `functional.py`
2. `functional.py` imports from `_ops.py` (registers op schemas and fake kernels)
3. `functional.py` imports the backends module
4. Each backend module (`backends/cuda/ops.py`, etc.) calls `@register_kernel(op_name, device)`
   at module level, registering implementations for their device type

The import chain in `functional.py`:
```python
import bitsandbytes.backends.default.ops      # Always loaded — pure PyTorch fallback
import bitsandbytes.backends.cuda.ops         # Loaded only if CUDA available
import bitsandbytes.backends.cpu.ops          # Always loaded (some ops conditional)
import bitsandbytes.backends.triton.ops       # Loaded only if triton installed
# etc.
```

### Dispatch precedence

When you call `torch.ops.bitsandbytes.quantize_4bit(tensor_on_cuda, ...)`:

1. PyTorch dispatches to the kernel registered for the tensor's device type
2. If `"cuda"` kernel exists → use it
3. If not → fall back to `"default"` kernel (pure PyTorch implementation)

This means:
- CUDA tensors use CUDA kernels (fast, ctypes → native CUDA)
- CPU tensors use CPU kernels if registered, otherwise default (pure PyTorch)
- Any new device automatically gets the `default` fallback

### Backend capabilities matrix

| Op Category | CUDA | CPU | Default | Triton | XPU | HPU | MPS |
|---|---|---|---|---|---|---|---|
| 8-bit quantize/dequant | ctypes | C++/partial | PyTorch | Triton kernels | SYCL | partial | partial |
| 4-bit quantize/dequant | ctypes | partial | PyTorch | Triton kernels | SYCL | partial | — |
| int8 matmul (cuBLASLt) | ctypes | torch._int_mm | PyTorch fp32 fallback | — | — | — | — |
| gemv_4bit (fused) | ctypes | — | PyTorch | — | — | — | — |
| Optimizer 32-bit | ctypes | — | torch.compile | Triton | — | — | — |
| Optimizer 8-bit blockwise | ctypes | — | — | Triton | — | — | — |

---

## 6. Native Library Loading (`cextension.py`)

This module handles discovering and loading the compiled C/CUDA shared library via ctypes.

### Loading process

1. `get_cuda_specs()` detects the CUDA version from PyTorch
2. `get_cuda_bnb_library_path()` constructs the expected library filename:
   - CUDA: `libbitsandbytes_cuda{VERSION}.so` (e.g., `libbitsandbytes_cuda124.so`)
   - ROCm: `libbitsandbytes_rocm{VERSION}.so`
   - CPU-only: `libbitsandbytes_cpu.so`
   - XPU: `libbitsandbytes_xpu.so`
   - MPS: `libbitsandbytes_mps.dylib`
3. `ct.cdll.LoadLibrary(path)` loads the shared library
4. The loaded library is wrapped in either:
   - `CudaBNBNativeLibrary` — if `get_context` symbol exists (CUDA/ROCm build)
   - `BNBNativeLibrary` — for CPU-only builds
   - `ErrorHandlerMockBNBNativeLibrary` — if loading fails (defers errors to call time)

### The `lib` global

```python
# cextension.py — at module level:
lib = get_native_library()  # This is the global used everywhere
```

All CUDA backend ops access native code through this `lib` object:
```python
from ...cextension import lib

# In backends/cuda/ops.py:
lib.cquantize_blockwise_fp16(code_ptr, A_ptr, absmax_ptr, out_ptr, blocksize, n)
```

### `BNBNativeLibrary.__getattr__`

The library wrapper uses `__getattr__` with caching. If a function is not found in the loaded
library, it returns a stub that raises `RuntimeError` when called (rather than at attribute
access time). This allows CPU-only installations to import successfully and only error when
GPU-specific functions are actually invoked.

### Environment variables

- `BNB_CUDA_VERSION` — Override the auto-detected CUDA version for library selection
- Standard CUDA env vars (`CUDA_HOME`, `LD_LIBRARY_PATH`) affect library discovery

---

## 7. The Functional Layer (`functional.py`)

This is the stateless Python API layer. It contains:

### Quantization codebook infrastructure

```python
# Pre-computed quantization maps:
create_dynamic_map(signed=True, total_bits=8)  # Creates 256-entry dynamic quantization codebook
create_normal_map(offset=0.9677083, symmetric=False)  # NF4 codebook from normal distribution
create_fp4_map()  # FP4 codebook

# These are stored as:
# - torch.Tensor of shape (256,) for 8-bit
# - torch.Tensor of shape (16,) for 4-bit
```

### QuantState class

```python
@dataclass
class QuantState:
    absmax: torch.Tensor          # Per-block absolute maximum values
    shape: torch.Size             # Original tensor shape before quantization
    dtype: torch.dtype            # Original tensor dtype
    blocksize: int                # Block size used for quantization (default 64)
    quant_type: str               # "nf4" or "fp4"
    code: torch.Tensor            # 16-element quantization codebook
    nested: bool = False          # Whether double quantization is used
    # If nested=True, the absmax values are themselves quantized:
    state2: Optional[QuantState]  # Nested quantization state for absmax
    offset: Optional[torch.Tensor]  # Offset for nested quantization
```

The `QuantState` is the metadata container that travels with every quantized tensor. It stores
everything needed to dequantize: the scaling factors (absmax), the codebook, the original shape,
and optionally a nested quantization state for the absmax values themselves ("double quantization").

### Key functions

**4-bit quantization (the QLoRA path):**
```python
def quantize_4bit(A, blocksize=64, compress_statistics=True, quant_type="fp4", quant_storage=torch.uint8):
    """Quantizes tensor A to 4-bit. Returns (packed_4bit_tensor, QuantState)."""
    # 1. Calls torch.ops.bitsandbytes.quantize_4bit → dispatched to backend
    # 2. If compress_statistics=True, also quantizes the absmax values (double quant)
    # 3. Returns QuantState with all metadata

def dequantize_4bit(A, quant_state, absmax=None, out=None, blocksize=64, quant_type="fp4"):
    """Dequantizes 4-bit tensor back to float. Uses QuantState for metadata."""
    # 1. If double quantization, first dequantize the absmax
    # 2. Calls torch.ops.bitsandbytes.dequantize_4bit → dispatched to backend
```

**8-bit quantization:**
```python
def int8_vectorwise_quant(A, threshold=0.0):
    """Row-wise int8 quantization. Returns (quantized, row_stats, outlier_cols)."""
    # If threshold > 0: identifies outlier columns (for LLM.int8())
    # Calls torch.ops.bitsandbytes.int8_vectorwise_quant

def int8_double_quant(A, threshold=0.0):
    """Both row-wise and column-wise int8 quantization."""
    # Used by the backward pass of LLM.int8()
    # Returns (quant_row, quant_col, row_stats, col_stats, outlier_cols)
```

**Blockwise 8-bit quantization (for optimizers):**
```python
def quantize_blockwise(A, code=None, absmax=None, out=None, blocksize=4096):
    """Blockwise quantization using a 256-entry codebook."""
    # Used for optimizer state compression
    # Default blocksize=4096 for optimizers (larger blocks = less memory overhead)

def dequantize_blockwise(A, quant_state=None, absmax=None, code=None, out=None, blocksize=4096, ...):
    """Inverse of quantize_blockwise."""
```

**Optimizers:**
```python
def optimizer_update_32bit(optimizer_name, grad, param, state1, beta1, eps, step, lr, state2=None, ...):
    """Dispatches 32-bit optimizer update to the appropriate backend kernel."""
    # Calls torch.ops.bitsandbytes.optimizer_update_32bit

def optimizer_update_8bit_blockwise(optimizer_name, grad, param, state1, state2, ...):
    """Dispatches 8-bit blockwise optimizer update."""
    # Calls torch.ops.bitsandbytes.optimizer_update_8bit_blockwise
```

**Inference (4-bit GEMV):**
```python
def gemv_4bit(A, B, out=None, transposed_A=False, transposed_B=False, state=None):
    """Fused 4-bit dequantize + matrix-vector multiply."""
    # Used when: single batch (A.numel() == A.shape[-1]) and inference mode
    # Much faster than separate dequant+matmul for single-token generation
    # Calls torch.ops.bitsandbytes.gemv_4bit
```

### CUBLAS_Context and utility classes

```python
class CUBLAS_Context:
    """Singleton managing cuBLAS handles per CUDA device."""
    # Used by int8 matmul to get cuBLASLt handle
    # get_instance().get_context(device) → cublasLtHandle_t

class GlobalPageManager:
    """Manages CUDA unified memory for paged optimizers."""
    # Paged optimizers use cudaMallocManaged for state tensors
    # Allows automatic CPU↔GPU migration
```

### Helper functions

```python
def get_ptr(tensor):
    """Gets raw pointer for ctypes calls. Returns None for None tensors."""

def _cuda_device_of(tensor):
    """Context manager that sets the correct CUDA device for the tensor."""

def _get_tensor_stream(tensor):
    """Gets the current CUDA stream for a tensor's device."""
```

---

## 8. Quantization Data Types and QuantState

### NF4 (Normal Float 4-bit)

NF4 is a 4-bit data type where each of the 16 quantization bins has equal probability under a
standard normal distribution N(0,1). This makes it optimal for normally-distributed weights
(which neural network weights approximately are).

The 16 NF4 values (normalized to [-1, 1]):
```
-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
 0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0
```

Note the asymmetry: there are 8 negative values and 8 non-negative values, with 0.0 as one of
the representable values.

### FP4 (Float Point 4-bit)

FP4 uses a 1-bit sign + 3-bit magnitude with a custom encoding:
```
Sign bit + 3-bit value:
0b000 = 0.0
0b001 = 0.005208 (subnormal)
0b010 = 0.6667
0b011 = 1.0
0b100 = 0.3333
0b101 = 0.5
0b110 = 0.1667
0b111 = 0.25
```

### 4-bit packing

Two 4-bit values are packed per byte:
```
packed_byte = (high_nibble << 4) | low_nibble
```

The packed tensor has shape `((n + 1) // 2, 1)` with `quant_storage` dtype (default `uint8`).
When `quant_storage` is not `uint8`, the packed bytes are viewed as the storage dtype.

### QuantState serialization

QuantState can serialize/deserialize for checkpointing via `as_dict(packed=True)` and
`from_dict()`. When saved to a state dict (e.g., in `Linear4bit._save_to_state_dict`), the
quant state components are stored alongside the weight with keys like:
```
weight.quant_state.bitsandbytes__nf4
weight.absmax
weight.quant_map
weight.nested_absmax
weight.nested_quant_map
weight.quant_state.nested_blocksize
weight.quant_state.nested_dtype
weight.quant_state.nested_offset
```

### Double quantization (compress_statistics)

When `compress_statistics=True` (default for 4-bit), the `absmax` values themselves are quantized
using 8-bit blockwise quantization. This reduces the memory overhead of storing scaling factors.
The nested quant state is stored inside `QuantState.state2`.

---

## 9. Autograd Functions (`autograd/_functions.py`)

### MatMul8bitLt (LLM.int8())

The core 8-bit matmul with custom forward and backward.

**Forward path:**
1. Quantize activations A to int8 (row-wise) via `int8_vectorwise_quant` or `int8_double_quant`
2. Quantize weights B to int8 (row-wise) if not already cached
3. If `threshold > 0`: identify outlier columns, use mixed-precision decomposition
   - Non-outlier part: int8 matmul via `int8_scaled_mm`
   - Outlier part: fp16 matmul on outlier columns only, added back to result
4. If `threshold == 0`: pure int8 matmul via `int8_scaled_mm`
5. Save quantized states for backward

**Backward path:**
- `grad_B`: Uses int8 matmul of grad_output^T × A^T (both quantized) + outlier correction
- `grad_A`: Dequantizes weights and does fp16 matmul: grad_output × W_dequant

**Key state object — `MatmulLtState`:**
```python
@dataclass
class MatmulLtState:
    CB: Optional[torch.Tensor] = None      # Quantized weight (int8)
    SCB: Optional[torch.Tensor] = None     # Weight row statistics (float32)
    threshold: float = 0.0                  # Outlier threshold for mixed-precision
    has_fp16_weights: bool = True           # Whether to keep fp16 weights
    is_training: bool = True
    # ... more fields for backward state
```

### MatMul8bitFp

A simpler 8-bit matmul for CPU/XPU that avoids the expensive int8 backward path:
- Forward: Dequantize weights to float, then `torch.nn.functional.linear`
- Backward: Standard fp16/fp32 matmul (no int8 in backward)
- ~3x faster on CPU/XPU because int8 quant/dequant kernels are slow on those platforms

### MatMul4Bit (QLoRA)

The 4-bit matmul autograd function.

**Forward path:**
1. Dequantize 4-bit weights B using `dequantize_4bit(B, quant_state)`
2. Cast to activation dtype
3. Standard `torch.nn.functional.linear(A, B_dequant, bias)`

**Backward path:**
- `grad_A`: Dequantize weights again, matmul with grad_output
- `grad_B`: **Not supported** (4-bit weights are frozen; this is by design for QLoRA)

### Dispatch logic

The top-level `matmul()` and `matmul_4bit()` functions choose which autograd class to use:

```python
def matmul(A, B, ...):
    if training and device in ("cpu", "xpu"):
        return MatMul8bitFp.apply(...)  # Faster on CPU/XPU
    return MatMul8bitLt.apply(...)      # Full LLM.int8()

def matmul_4bit(A, B, quant_state, ...):
    if A.numel() == A.shape[-1] and not requires_grad:
        return gemv_4bit(...)  # Fast path: fused kernel for single-token inference
    return MatMul4Bit.apply(...)  # General path: dequant + matmul
```

### GlobalOutlierPooler

A singleton that tracks outlier dimensions across layers:
```python
class GlobalOutlierPooler:
    """Pools outlier dimensions across layers for small models."""
    # Important for small models where outlier features are less systematic
    # Used when MatmulLtState.use_pool = True
```

---

## 10. Neural Network Modules (`nn/modules.py`)

### Linear4bit

The QLoRA module. This is the most widely used component via HuggingFace transformers integration.

```python
class Linear4bit(nn.Linear):
    def __init__(self, input_features, output_features, bias=True,
                 compute_dtype=None, compress_statistics=True,
                 quant_type="fp4", quant_storage=torch.uint8, device=None):
        # Weight is wrapped in Params4bit (quantizes on .to(device))
        self.weight = Params4bit(self.weight.data, ...)
```

**Quantization trigger:** Weights are quantized lazily — when you call `.to("cuda")` or `.cuda()`,
`Params4bit.to()` detects the device move and calls `_quantize()`.

**Forward pass:**
1. Fix quant state if lost (FSDP compatibility)
2. Auto-detect compute dtype from input if not set
3. Cast input to compute_dtype
4. Call `bnb.matmul_4bit(x, weight.t(), quant_state=...)`

**CPU inference path:** When `has_avx512bf16` and not training, weights are converted to a special
packed format optimized for CPU AVX512 inference.

### Params4bit

Custom `torch.nn.Parameter` subclass that carries quantization metadata:

```python
class Params4bit(torch.nn.Parameter):
    blocksize: int
    compress_statistics: bool
    quant_type: str          # "nf4" or "fp4"
    quant_state: QuantState
    quant_storage: torch.dtype
    bnb_quantized: bool
    module: Optional[Linear4bit]  # Back-reference to parent module
```

Key behaviors:
- `to(device)`: If not yet quantized and moving to a non-meta device → quantize
- `__torch_function__`: Handles `torch.chunk` and `torch.split` to preserve quant metadata
- `from_prequantized()`: Class method for loading already-quantized weights
- Supports `__getstate__`/`__setstate__` for pickling and `__deepcopy__`/`__copy__`

### Linear8bitLt

The LLM.int8() module.

```python
class Linear8bitLt(nn.Linear):
    def __init__(self, input_features, output_features, bias=True,
                 has_fp16_weights=True, threshold=0.0, ...):
        self.state = bnb.MatmulLtState()
        self.weight = Int8Params(self.weight.data, has_fp16_weights=...)
```

**`has_fp16_weights` modes:**
- `True` (default): Keeps fp16 weights, quantizes on every forward pass (training mode)
- `False`: Quantizes weights once on `.to(device)`, stores int8 permanently (inference mode)

**`threshold` parameter:**
- `0.0`: No outlier decomposition, pure int8 matmul
- `> 0.0` (e.g., 6.0): Mixed-precision decomposition — columns with activations exceeding
  threshold are computed in fp16

**State dict handling:**
- Saves `weight` (int8 data) + `SCB` (row statistics) + `weight_format` (always "row")
- Custom `_load_from_state_dict` to handle SCB restoration
- `_register_load_state_dict_pre_hook(maybe_rearrange_weight)` for format migration

### Int8Params

```python
class Int8Params(torch.nn.Parameter):
    CB: Optional[torch.Tensor]   # Quantized weight (same as .data when quantized)
    SCB: Optional[torch.Tensor]  # Row-wise scale factors
    has_fp16_weights: bool
```

Quantization trigger: Like Params4bit, quantizes on `to(device)` when moving from CPU to GPU.

### Embedding variants

- `StableEmbedding` — Adds LayerNorm + forces 32-bit optimizer states
- `Embedding` — Standard with 32-bit optimizer override
- `Embedding8bit` — Int8 quantized embeddings (dequant on lookup)
- `Embedding4bit` — 4-bit quantized with partial dequantization optimization
- `EmbeddingFP4`, `EmbeddingNF4` — Convenience subclasses

### Convenience aliases

```python
LinearFP4 = Linear4bit(quant_type="fp4")
LinearNF4 = Linear4bit(quant_type="nf4")
```

---

## 11. Optimizer System (`optim/`)

### Class hierarchy

```
torch.optim.Optimizer
└── Optimizer8bit
    ├── Optimizer1State    # SGD, Adagrad, RMSprop (1 moment)
    │   ├── SGD / SGD8bit / SGD32bit
    │   ├── Adagrad / Adagrad8bit / Adagrad32bit
    │   └── RMSprop / RMSprop8bit / RMSprop32bit
    └── Optimizer2State    # Adam, Lion, LAMB, LARS, AdEMAMix (2 moments)
        ├── Adam / Adam8bit / Adam32bit / PagedAdam / PagedAdam8bit / PagedAdam32bit
        ├── AdamW / AdamW8bit / AdamW32bit / PagedAdamW / PagedAdamW8bit / PagedAdamW32bit
        ├── Lion / Lion8bit / Lion32bit / PagedLion / PagedLion8bit / PagedLion32bit
        ├── LAMB / LAMB8bit / LAMB32bit
        ├── LARS / LARS8bit / LARS32bit / PytorchLARS
        └── AdEMAMix / AdEMAMix8bit / AdEMAMix32bit / PagedAdEMAMix*
```

### How optimizer dispatch works

Each concrete optimizer class (e.g., `Adam8bit`) is a thin wrapper that calls `super().__init__`
with the optimizer name string and the bit width:

```python
class Adam8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), ...):
        super().__init__("adam", params, lr, betas, ..., optim_bits=8, ...)
```

The base class `Optimizer2State.update_step()` then dispatches based on state dtype:

```python
def update_step(self, group, p, gindex, pindex):
    if state["state1"].dtype == torch.float:
        F.optimizer_update_32bit(self.optimizer_name, grad, p, state1, ...)
    elif state["state1"].dtype == torch.uint8 and config["block_wise"]:
        F.optimizer_update_8bit_blockwise(self.optimizer_name, grad, p, state1, ...)
    elif state["state1"].dtype == torch.uint8 and not config["block_wise"]:
        F.optimizer_update_8bit(self.optimizer_name, grad, p, state1, ...)
```

### Optimizer state initialization

In `init_state()`:
- If parameter numel < `min_8bit_size` (default 4096): always use 32-bit state (too small for
  quantization to help)
- 32-bit state: `state1 = zeros_like(p, dtype=float32)`
- 8-bit state: `state1 = zeros_like(p, dtype=uint8)` + quantization maps + absmax buffers

### 8-bit optimizer state compression

For 8-bit optimizers, the optimizer states (momentum, variance) are stored as uint8 and
dynamically quantized/dequantized each step:

1. Each state tensor is divided into blocks of 256 elements
2. Per-block `absmax` values are maintained (float32)
3. A quantization map (`qmap`) maps 256 uint8 values to float32 values
4. The kernel reads uint8 state → dequantizes → applies update → re-quantizes → writes back

### Paged optimizers

Paged optimizers use CUDA unified memory (`cudaMallocManaged`) for state tensors > 100K elements.
This allows automatic CPU↔GPU page migration, reducing GPU memory pressure when many parameters
have inactive gradients:

```python
def get_state_buffer(self, p, dtype):
    if not self.is_paged or p.numel() < 1e5:
        return torch.zeros_like(p, dtype=dtype, device=p.device)
    else:
        buff = F.get_paged(*p.shape, dtype=dtype, device=p.device)  # cudaMallocManaged
        ...
```

### GlobalOptimManager

Singleton that allows per-parameter optimizer config overrides:

```python
mng = bnb.optim.GlobalOptimManager.get_instance()
mng.register_parameters(model.parameters())
mng.override_config(model.fc1.weight, 'optim_bits', 32)  # Force 32-bit for this param
```

Used by `StableEmbedding` and `Embedding` to force 32-bit optimizer states for embedding layers.

### FSDP compatibility

`Optimizer8bit` overrides `state_dict()` and `load_state_dict()` to wrap quantization-specific
tensors (state1, state2, absmax, qmap, etc.) in a nested dict. This prevents FSDP's
`full_optim_state_dict` from trying to gather these tensors across ranks (they have different
shapes than the parameter tensors, which would cause gather failures).

---

## 12. CUDA/C++ Native Code (`csrc/`)

### File organization

| File | Purpose |
|---|---|
| `kernels.cu` | `__global__` CUDA kernel functions (kQuantizeBlockwise, kOptimizer*, etc.) |
| `ops.cu` | Host-side dispatch functions that launch kernels with grid/block configs |
| `pythonInterface.cpp` | C-linkage wrappers for ctypes: unmangled function names, macro-expanded per dtype |
| `ops.cuh` | Declarations for ops.cu functions + cuBLAS/cuSPARSE context classes |
| `kernels.cuh` | Declarations for kernel functions |
| `common.cuh` | Compute capability macros and constants |
| `cpu_ops.cpp` / `cpu_ops.h` | CPU-native implementations (blockwise quant, etc.) |

### The call chain: Python → C

```
Python: lib.cquantize_blockwise_fp16(code_ptr, A_ptr, absmax_ptr, out_ptr, blocksize, n)
   ↓
pythonInterface.cpp: void cquantize_blockwise_fp16(...)
   calls → quantizeBlockwise<half, 0, 0>(code, A, absmax, out, NULL, 0, blocksize, n)
   ↓
ops.cu: template<T, STOCHASTIC, DATA_TYPE> void quantizeBlockwise(...)
   launches → kQuantizeBlockwise<half, 4096, 4, 0, 0><<<num_blocks, 1024>>>(...)
   ↓
kernels.cu: __global__ void kQuantizeBlockwise<T, BLOCK_SIZE, NUM_PER_TH, STOCHASTIC, DATA_TYPE>(...)
   actual CUDA computation
```

### Naming convention in pythonInterface.cpp

Functions are generated via macros to cover all dtype combinations:

```cpp
#define MAKE_FUNC_BLOCKWISE(fname, optim_name, gtype, gbits)
    void c##fname##_blockwise_##gbits(...)
    { fname##Blockwise<gtype, optim_name>(...); }

// Expands to:
// void cquantize_blockwise_fp16(...)
// void cquantize_blockwise_bf16(...)
// void cquantize_blockwise_fp32(...)
```

Similarly for optimizers:
```cpp
MAKE_FUNC32(cadam, ADAM, float, fp32)
MAKE_FUNC32(cadam, ADAM, half, fp16)
MAKE_FUNC32(cadam, ADAM, __nv_bfloat16, bf16)
// → cadam32bit_grad_fp32, cadam32bit_grad_fp16, cadam32bit_grad_bf16
```

4-bit functions use a separate naming pattern:
```cpp
// void cquantize_blockwise_fp16_nf4(...)  ← 4-bit NF4 with fp16 input
// void cquantize_blockwise_bf16_fp4(...)  ← 4-bit FP4 with bf16 input
```

### Optimizer kernel organization

The CUDA optimizer kernels handle all optimizer types via a single templated kernel, switched on
the `OPTIMIZER` template parameter:

```cpp
enum Optimizer_t {
    ADAM = 0,
    MOMENTUM = 1,
    RMSPROP = 2,
    LARS = 3,
    ADAGRAD = 4,
    LION = 5,
    ADEMAMIX = 6
};

template <typename T, int OPTIMIZER>
__global__ void kOptimizer32bit2State(...) {
    switch (OPTIMIZER) {
        case ADAM: ...
        case ADEMAMIX: ...
    }
}

template <typename T, int OPTIMIZER>
__global__ void kOptimizer32bit1State(...) {
    switch (OPTIMIZER) {
        case MOMENTUM: ...
        case LION: ...
        case RMSPROP: ...
        case ADAGRAD: ...
    }
}
```

### Compute capability handling

From `common.cuh`:
```cpp
#define BNB_CC_VOLTA 700
#define BNB_CC_TURING 750
#define BNB_CC_AMPERE 800
#define BNB_CC_ADA 890
#define BNB_CC_HOPPER 900
#define BNB_CC_BLACKWELL 1000

#define BNB_FP16_MMA_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_VOLTA)      // sm_70+
#define BNB_INT8_MMA_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_VOLTA_XAVIER) // sm_72+
#define BNB_BF16_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_AMPERE)         // sm_80+
#define BNB_FP8_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_ADA)             // sm_89+
```

Thread/block limits per architecture:
```cpp
// Turing (sm_75): 1024 max threads per SM
// Ampere (sm_80): 2048 max threads per SM
// Ada (sm_86-89): 1536 max threads per SM
// Others: 2048 max threads per SM
```

### int8 matmul via cuBLASLt

The `igemmlt` function in `ops.cu` calls cuBLASLt for int8 × int8 → int32 matmul:

```cpp
template <int DTYPE_OUT, int SCALE_ROWS>
int igemmlt(cublasLtHandle_t ltHandle, int m, int n, int k,
            const int8_t *A, const int8_t *B, void *C,
            float *row_scale, int lda, int ldb, int ldc, cudaStream_t stream);
```

This is the performance-critical path for LLM.int8(). When inner dimensions are not divisible
by 4, the CUDA backend falls back to fp32 matmul (cuBLASLt requirement).

### Quantization kernel design

The blockwise quantization kernels process data in blocks (typically 64-4096 elements):

1. Each CUDA block handles one quantization block
2. Shared memory is used for block-level reduction (finding absmax)
3. Each thread processes `NUM_PER_TH` elements (typically 2-8)
4. CUB block-level primitives are used for reductions (`BlockReduce`)

For 4-bit: two values are packed per byte. A specialized kernel `kQuantizeBlockwise32` handles
the smallest blocksize (32) by processing 2 quantization blocks per warp.

### ROCm/HIP support

ROCm uses separate source files (`ops.hip`, `kernels.hip`, etc.) that mirror the CUDA versions
with HIP API translations. Key difference: ROCm uses warp size 64 on some architectures
(vs CUDA's 32), tracked by `ROCM_WARP_SIZE_64`. This affects allowed blocksizes:
- CUDA: blocksizes 32, 64, 128, 256, 512, 1024, 2048, 4096
- ROCm (warp 64): blocksizes 64, 128, 256, 512, 1024, 2048, 4096 (no 32)

---

## 13. Build System (`CMakeLists.txt`)

### Build configurations

The `COMPUTE_BACKEND` CMake variable selects the target:

| Backend | Library name | Languages | Dependencies |
|---|---|---|---|
| `cpu` | `libbitsandbytes_cpu.so` | C++17 | OpenMP (optional) |
| `cuda` | `libbitsandbytes_cuda{VER}.so` | C++17 + CUDA | cudart, cublas, cublasLt, cusparse |
| `hip` | `libbitsandbytes_rocm{VER}.so` | C++17 + HIP | hipblas, hiprand, hipsparse |
| `mps` | `libbitsandbytes_mps.dylib` | C++17 + ObjC++ | Metal framework |
| `xpu` | `libbitsandbytes_xpu.so` | C++20 + SYCL | Intel oneAPI |

### CUDA architecture targeting

By default, the build targets all architectures supported by the detected CUDA toolkit:

```cmake
# CUDA 12.8+: sm_50 through sm_121
# CUDA 13.0+: sm_75 through sm_121 (drops pre-Turing)
```

Users can override with `-DCOMPUTE_CAPABILITY="89;90;100"`.

The build generates native cubin for all selected architectures, plus PTX for the highest
(enabling forward compatibility with future GPUs).

### CPU-specific flags

For x86_64:
```cmake
-mavx512f -mavx512dq -mavx512bw -mavx512vl    # AVX-512 if supported
-mavx512bf16                                     # BF16 instructions if supported
-mprefer-vector-width=256 -mfma -mavx2          # Always
```

### Supported CUDA versions

- Minimum: CUDA 11.8
- Maximum: CUDA 13.x (CUDA 14+ is rejected)
- Key feature thresholds:
  - CUDA 12.8+: Blackwell support (sm_100, sm_120)
  - CUDA 13.0+: sm_110 (Thor Blackwell), drops pre-Turing

---

## 14. Data Flow: End-to-End Traces

### Trace 1: 4-bit inference (single token generation)

```
User: model(input_ids)
  ↓
Linear4bit.forward(x)                              # nn/modules.py
  ├── fix_4bit_weight_quant_state_from_module()     # Recover quant_state if lost (FSDP)
  ├── x = x.to(self.compute_dtype)                  # Cast input
  └── bnb.matmul_4bit(x, weight.t(), quant_state)  # autograd/_functions.py
        ↓
      matmul_4bit():
        ├── A.numel() == A.shape[-1]?               # Single batch check
        │   YES → F.gemv_4bit(A, B.t(), state)      # Fast path!
        │            ↓
        │          torch.ops.bitsandbytes.gemv_4bit  # _ops.py dispatch
        │            ↓
        │          CUDA: lib.cgemm_4bit_inference_naive_fp16(...)  # backends/cuda/ops.py
        │            ↓
        │          gemm_4bit_inference_naive<half, 16>(...)  # csrc/pythonInterface.cpp
        │            ↓
        │          kgemm_4bit_inference_naive<half><<<...>>>  # csrc/kernels.cu
        │
        │   NO → MatMul4Bit.apply(A, B, quant_state)  # General path
        │            ↓
        │          F.dequantize_4bit(B, quant_state)
        │            ↓
        │          torch.nn.functional.linear(A, B_dequant.t(), bias)
        └── + bias
```

### Trace 2: 8-bit linear forward (LLM.int8() with outlier decomposition)

```
Linear8bitLt.forward(x)                               # nn/modules.py
  ├── self.init_8bit_state()                            # Move CB/SCB from weight to state
  └── bnb.matmul(x, self.weight, state=self.state)     # autograd/_functions.py
        ↓
      MatMul8bitLt.forward(A, B, state):
        ├── A_int8, SCA, outlier_cols = F.int8_vectorwise_quant(A.fp16, threshold=6.0)
        │     → torch.ops.bitsandbytes.int8_vectorwise_quant → CUDA kernel
        │
        ├── state.CB, state.SCB = F.int8_vectorwise_quant(B.fp16)  # If not cached
        │
        ├── threshold > 0 and outlier_cols exist:
        │   output, subA = torch.ops.bitsandbytes.int8_mixed_scaled_mm(
        │       A, CA, CB, SCA, SCB, outlier_cols, bias)
        │     ↓
        │   1. Dequantize weight outlier columns: int8_vectorwise_dequant(CB[:, outliers], SCB)
        │   2. Int8 matmul: int8_scaled_mm(CA, CB, SCA, SCB, bias)
        │        ↓
        │      int8_linear_matmul(CA, CB) → cuBLASLt igemmlt
        │        ↓
        │      int8_mm_dequant(result_i32, SCA, SCB) → fp16 via CUDA kernel
        │   3. Outlier contribution: output.addmm(subA, subB_dequant)
        │
        └── Save state for backward (CAt, SCAt, idx)
```

### Trace 3: 8-bit optimizer step (Adam8bit)

```
optimizer.step()                                       # optim/optimizer.py
  ↓
Optimizer8bit.step():
  for p in params:
    if state empty → self.init_state(group, p, ...)
    self.update_step(group, p, ...)
      ↓
    Optimizer2State.update_step():
      ├── p.data = p.data.contiguous()
      ├── config = self.get_config(gindex, pindex, group)
      │
      ├── state["state1"].dtype == uint8 and block_wise:
      │   F.optimizer_update_8bit_blockwise("adam", grad, p, state1, state2,
      │       beta1, beta2, ..., qmap1, qmap2, absmax1, absmax2, ...)
      │     ↓
      │   torch.ops.bitsandbytes.optimizer_update_8bit_blockwise(...)
      │     ↓
      │   CUDA: lib.cadam_8bit_blockwise_grad_fp16(p, g, state1, state2,
      │       beta1, beta2, ..., qmap1, qmap2, absmax1, absmax2, ...)
      │     ↓
      │   pythonInterface.cpp → optimizerStatic8bitBlockwise<half, ADAM>(...)
      │     ↓
      │   ops.cu → kOptimizerStatic8bit2StateBlockwise<half, ADAM><<<...>>>
      │     ↓
      │   kernels.cu:
      │     1. Load uint8 state → dequantize via qmap lookup
      │     2. Apply Adam update: m = β₁m + (1-β₁)g; v = β₂v + (1-β₂)g²
      │     3. p = p - lr * m_hat / (√v_hat + ε)
      │     4. Re-quantize states → write back as uint8
      │     5. Update absmax values
      │
      └── state["state1"].dtype == float32:
          F.optimizer_update_32bit("adam", grad, p, state1, ...)
            ↓
          (Similar path but simpler: no quantize/dequantize)
```

### Trace 4: Quantization on `.to("cuda")`

```
model = model.to("cuda")
  ↓
For each Linear4bit module:
  Linear4bit.to("cuda")
    → Params4bit.to(device="cuda")
      ↓
    Params4bit.to():
      if not bnb_quantized and device.type != "meta":
        self._quantize(device)
          ↓
        w = self.data.contiguous().to(device)
        w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=64, ...)
          ↓
        1. torch.ops.bitsandbytes.quantize_4bit(w, 64, "nf4", uint8) → CUDA kernel
        2. If compress_statistics: quantize absmax with quantize_blockwise
        3. Build QuantState(absmax, shape, dtype, blocksize, code, nested_state)
          ↓
        self.data = w_4bit      # Packed 4-bit tensor
        self.quant_state = quant_state
        self.bnb_quantized = True
```

---

## 15. Key Design Patterns

### Pattern 1: torch.library for multi-backend ops

Every new operation must follow this pattern:
```python
# 1. Define schema in _ops.py
torch.library.define("bitsandbytes::my_op", "(Tensor A, int param) -> Tensor")

# 2. Register fake kernel for torch.compile
@torch.library.register_fake("bitsandbytes::my_op")
def _(A, param):
    return torch.empty_like(A)

# 3. Register CUDA implementation
# In backends/cuda/ops.py:
@register_kernel("bitsandbytes::my_op", "cuda")
def _(A, param):
    # ... actual CUDA implementation ...

# 4. Register default fallback
# In backends/default/ops.py:
@register_kernel("bitsandbytes::my_op", "default")
def _(A, param):
    # ... pure PyTorch implementation ...
```

### Pattern 2: Input validation with `torch._check`

Backend ops use `torch._check()` (not `assert`) for input validation:
```python
torch._check(A.dtype == torch.int8, lambda: f"A must be int8, got {A.dtype}")
torch._check_is_size(blocksize)
```

This ensures validation works correctly under `torch.compile` (assertions are not traced).

### Pattern 3: Lazy quantization on device transfer

Both `Params4bit` and `Int8Params` override `.to()` to trigger quantization:
```python
def to(self, *args, **kwargs):
    device, dtype, ... = torch._C._nn._parse_to(*args, **kwargs)
    if not self.bnb_quantized and device is not None and device.type != "meta":
        return self._quantize(device)
    ...
```

### Pattern 4: ctypes calling convention for CUDA

```python
# Standard pattern in backends/cuda/ops.py:
with _cuda_device_of(A):          # Set correct CUDA device
    lib.c_function_name(
        get_ptr(A),                # Raw pointer via ctypes
        get_ptr(out),
        ct.c_int32(n),             # Scalar args as c_types
        ct.c_float(threshold),
        _get_tensor_stream(A),     # Current CUDA stream
    )
```

### Pattern 5: Optimizer naming convention

Every optimizer follows a strict naming pattern:
```python
class {Name}(Optimizer{1,2}State):     # Default: 32-bit, switches to 8-bit if optim_bits=8
class {Name}8bit(Optimizer{1,2}State): # Always 8-bit (hardcoded optim_bits=8)
class {Name}32bit(Optimizer{1,2}State): # Always 32-bit (hardcoded optim_bits=32)
class Paged{Name}(Optimizer{1,2}State): # Paged variant (is_paged=True)
class Paged{Name}8bit(...):             # Paged + 8-bit
class Paged{Name}32bit(...):            # Paged + 32-bit
```

All pass `optimizer_name` (e.g., `"adam"`, `"lion"`) to the base class, which is used to look
up the correct C function in the `str2optimizer*` dictionaries.

### Pattern 6: `.out` variants for ops

Many ops have both a returning variant and an `.out` variant:
```python
# _ops.py:
torch.library.define("bitsandbytes::dequantize_4bit",     "(...) -> Tensor")
torch.library.define("bitsandbytes::dequantize_4bit.out", "(..., Tensor(a!) out) -> ()")

# backends/cuda/ops.py:
@register_kernel("bitsandbytes::dequantize_4bit", "cuda")
def _(A, absmax, blocksize, quant_type, shape, dtype):
    out = torch.empty(shape, dtype=dtype, device=A.device)
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)
    return out

@register_kernel("bitsandbytes::dequantize_4bit.out", "cuda")
def _(A, absmax, blocksize, quant_type, shape, dtype, out):
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)
```

---

## 16. Cross-Cutting Concerns

### torch.compile compatibility

The codebase has extensive `torch.compile` support:
- All ops registered via `torch.library` with `register_fake` for tracing
- Input validation uses `torch._check` instead of Python `assert`
- The `default` backend implementations use `@_try_torch_compile` decorator for automatic
  compilation with fallback
- `_is_compiling = torch.compiler.is_compiling` is used to skip certain operations during
  compilation (e.g., dtype warnings)

### FSDP / distributed training compatibility

Several components have FSDP-specific handling:
- `Params4bit.module` back-reference enables quant_state recovery after FSDP parameter flattening
- `fix_4bit_weight_quant_state_from_module()` restores lost quant_state
- `Optimizer8bit.state_dict()` wraps quantization tensors to prevent FSDP gather failures
- `Linear4bit._save_to_state_dict()` serializes quant_state components alongside weights

### Thread safety and CUDA streams

- `_cuda_device_of(tensor)` context manager ensures operations run on the correct device
- `_get_tensor_stream(tensor)` passes the current CUDA stream to native kernels
- cuBLAS/cuSPARSE contexts are per-device singletons (`CUBLAS_Context`)
- `sync_gpu(p)` is called after paged optimizer steps to ensure async operations complete

### Tensor mutation safety

Backend ops must NOT mutate user-provided input tensors. This was a historical bug source
(see issue #1587 where `int8_vectorwise_quant` mutated the input's absmax values). The
pattern to follow:
```python
# WRONG: Mutates user tensor
A[outliers] = 0

# RIGHT: Clone or use masked_fill
A = A.masked_fill(outlier_mask, 0.0)
```

### Error handling in native code

CUDA errors are checked via macros:
```cpp
#define CUDA_CHECK_RETURN(value) {
    cudaError_t _m_cudaStat = value;
    if (_m_cudaStat != cudaSuccess) {
        fprintf(stderr, "Error %s at line %d\n", cudaGetErrorString(_m_cudaStat), __LINE__);
        exit(1);  // Note: calls exit(), not throw
    }
}
```

The `exit(1)` behavior means CUDA errors are fatal and crash the process. cuBLAS errors
return error codes that are propagated back to Python as exceptions.

---

## 17. Test Structure

### Test files and what they cover

| File | Tests |
|---|---|
| `test_functional.py` | Quantize/dequantize correctness, codebook generation, percentile clipping, optimizer updates |
| `test_ops.py` | `torch.ops.bitsandbytes.*` dispatch, multi-backend, torch.compile tracing |
| `test_linear4bit.py` | Linear4bit module: forward, serialization, FSDP, compute dtype, quant types |
| `test_linear8bitlt.py` | Linear8bitLt: forward, backward, outlier threshold, state dict |
| `test_modules.py` | Embedding modules, StableEmbedding, general nn.Module behavior |
| `test_autograd.py` | Gradient correctness for quantized matmul |
| `test_optim.py` | All optimizers: convergence, state dict save/load, paged variants, 8-bit vs 32-bit |
| `test_triton.py` | Triton kernel equivalence with CUDA kernels |
| `test_deprecated.py` | Deprecation warnings fire correctly |
| `test_parametrize.py` | Weight parametrization with quantized modules |
| `test_generation.py` | End-to-end text generation with quantized models |
| `test_cuda_setup_evaluator.py` | CUDA detection and library loading |

### Common test patterns

- Device parametrization: Tests run across available devices (cuda, cpu, xpu)
- Dtype parametrization: Tests cover fp16, bf16, fp32 where applicable
- Blocksize parametrization: Multiple blocksizes to catch edge cases
- Error bound checks: `torch.testing.assert_close(actual, expected, atol=..., rtol=...)`
- GPU-only marking: `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`

### conftest.py fixtures

- `requires_cuda` — Skip if no CUDA GPU
- `requires_gpu` — Skip if no GPU of any type
- Device fixtures for parametrized testing
