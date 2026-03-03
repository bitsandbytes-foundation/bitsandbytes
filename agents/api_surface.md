# bitsandbytes Public API Surface

This document catalogs every public symbol in the bitsandbytes library, organized by
subsystem. For each symbol it lists: the module path, what it is, its stability status,
and its signature or key attributes. A reviewer can use this to quickly check whether a
PR is adding, removing, or modifying public API correctly.

**Version at time of writing:** 0.49.2.dev0

---

## Table of Contents

1. [Top-Level Exports (`bitsandbytes`)](#1-top-level-exports)
2. [Neural Network Modules (`bitsandbytes.nn`)](#2-neural-network-modules)
3. [Optimizers (`bitsandbytes.optim`)](#3-optimizers)
4. [Functional API (`bitsandbytes.functional`)](#4-functional-api)
5. [Autograd Functions (`bitsandbytes.autograd._functions`)](#5-autograd-functions)
6. [Torch Custom Ops (`bitsandbytes._ops`)](#6-torch-custom-ops)
7. [Research / Experimental (`bitsandbytes.research`)](#7-research--experimental)
8. [Utilities (`bitsandbytes.utils`)](#8-utilities)
9. [Native Library Interface (`bitsandbytes.cextension`)](#9-native-library-interface)
10. [Backend System (`bitsandbytes.backends`)](#10-backend-system)
11. [Deprecated Symbols](#11-deprecated-symbols)
12. [Downstream Integration Points](#12-downstream-integration-points)
13. [Stability Tiers](#13-stability-tiers)

---

## 1. Top-Level Exports

These are available directly as `import bitsandbytes as bnb; bnb.<symbol>`.

### Re-exported from submodules

| Symbol | Origin | Type | Notes |
|--------|--------|------|-------|
| `bnb.MatmulLtState` | `autograd._functions` | dataclass | State container for 8-bit matmul |
| `bnb.matmul` | `autograd._functions` | function | 8-bit matrix multiplication |
| `bnb.matmul_4bit` | `autograd._functions` | function | 4-bit matrix multiplication |
| `bnb.modules` | `nn.modules` | module | nn module namespace |
| `bnb.adam` | `optim.adam` | module | Adam optimizer namespace |
| `bnb.research` | `research` | module | Research/experimental namespace |
| `bnb.utils` | `utils` | module | Utilities namespace |

### Module-level attributes

| Symbol | Type | Value/Description |
|--------|------|-------------------|
| `bnb.__version__` | `str` | `"0.49.2.dev0"` |
| `bnb.features` | `set` | `{"multi_backend"}` — Integration signal for transformers/diffusers |
| `bnb.supported_torch_devices` | `set` | `{"cpu", "cuda", "xpu", "hpu", "npu", "mps"}` |
| `bnb.__pdoc__` | `dict` | Controls pdoc visibility for internal classes |

### Backend auto-loading

On import, bitsandbytes conditionally imports backend modules based on device availability:

- `backends.cpu.ops` — Always loaded
- `backends.default.ops` — Always loaded
- `backends.cuda.ops` — Loaded if `torch.cuda.is_available()`
- `backends.xpu.ops` — Loaded if `torch.xpu.is_available()`
- `backends.hpu.ops` — Loaded if `habana_frameworks` is importable and `torch.hpu.is_available()`

Additionally, `_import_backends()` discovers external packages with `bitsandbytes.backends`
entry points (pip-installed backend plugins).

---

## 2. Neural Network Modules

**Import path:** `from bitsandbytes.nn import <Class>`

All modules are in `bitsandbytes.nn.modules` and re-exported through `bitsandbytes.nn.__init__`.

### 2.1 Linear Layers

#### `Linear4bit` — 4-bit quantized linear layer (QLoRA)

```
bitsandbytes.nn.Linear4bit(
    input_features: int,
    output_features: int,
    bias: bool = True,
    compute_dtype: Optional[torch.dtype] = None,
    compress_statistics: bool = True,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
    device = None,
)
```

**Parent:** `torch.nn.Linear`
**Stability:** Stable — Core API, used extensively by transformers and PEFT.
**Behavior:**
- Weights are stored as `Params4bit` (quantized on `.to(device)`)
- Forward: dequantizes, computes matmul via `bnb.matmul_4bit`
- `compute_dtype` controls the dtype used for the matmul computation
- `compress_statistics` enables double quantization of absmax values (saves memory)
- `quant_type` selects the 4-bit quantization scheme: `"fp4"` or `"nf4"`
- `quant_storage` controls the packed storage dtype (default: `torch.uint8`)
- State dict serialization includes packed `QuantState` for safetensors compatibility
- CPU inference path supports AVX512BF16 acceleration via packed weight format

#### `LinearFP4` — Convenience wrapper for FP4

```
bitsandbytes.nn.LinearFP4(
    input_features, output_features, bias=True,
    compute_dtype=None, compress_statistics=True,
    quant_storage=torch.uint8, device=None,
)
```

**Parent:** `Linear4bit` with `quant_type="fp4"` hardcoded.
**Stability:** Stable.

#### `LinearNF4` — Convenience wrapper for NF4

```
bitsandbytes.nn.LinearNF4(
    input_features, output_features, bias=True,
    compute_dtype=None, compress_statistics=True,
    quant_storage=torch.uint8, device=None,
)
```

**Parent:** `Linear4bit` with `quant_type="nf4"` hardcoded.
**Stability:** Stable.

#### `Linear8bitLt` — 8-bit linear layer (LLM.int8())

```
bitsandbytes.nn.Linear8bitLt(
    input_features: int,
    output_features: int,
    bias: bool = True,
    has_fp16_weights: bool = True,
    threshold: float = 0.0,
    index = None,
    device = None,
)
```

**Parent:** `torch.nn.Linear`
**Stability:** Stable — Core API for LLM.int8().
**Behavior:**
- Weights stored as `Int8Params` (quantized on `.to(device)` if `has_fp16_weights=False`)
- `has_fp16_weights=True`: weights stay in fp16, quantized on-the-fly each forward pass
- `has_fp16_weights=False`: weights quantized once on `.to(device)`, stored as int8
- `threshold > 0.0`: enables mixed-precision decomposition (outlier columns in fp16, rest in int8)
- `threshold == 0.0`: all columns quantized to int8
- Forward: calls `bnb.matmul(x, self.weight, bias, state)`
- State dict includes SCB (column scaling factors) and weight_format metadata

#### `OutlierAwareLinear` — Base class for outlier-aware quantization

```
bitsandbytes.nn.OutlierAwareLinear(
    input_features, output_features, bias=True, device=None,
)
```

**Parent:** `torch.nn.Linear`
**Stability:** Experimental / semi-public.
**Notes:** Requires `OutlierTracer.initialize(model)` before use. Abstract methods
`forward_with_outliers` and `quantize_weight` must be overridden.

#### `SwitchBackLinearBnb` — SwitchBack linear using bnb backend

```
bitsandbytes.nn.SwitchBackLinearBnb(
    input_features, output_features, bias=True,
    has_fp16_weights=True, memory_efficient_backward=False,
    threshold=0.0, index=None, device=None,
)
```

**Parent:** `torch.nn.Linear`
**Stability:** Experimental.
**Notes:** Uses `Int8Params` + `MatmulLtState`. Calls `bnb.matmul_mixed` for int8 matmul with mixed precision in forward.

### 2.2 Triton-Based Linear Layers

These require triton to be installed. Import from `bitsandbytes.nn`.

#### `SwitchBackLinear` — Triton-based SwitchBack

```
bitsandbytes.nn.SwitchBackLinear(
    in_features: int, out_features: int, bias: bool = True,
    device=None, dtype=None,
    vector_wise_quantization: bool = False,
    mem_efficient: bool = False,
)
```

**Parent:** `torch.nn.Linear`
**Stability:** Experimental — requires triton.
**Notes:** Has a `prepare_for_eval()` method that pre-quantizes weights.

#### `SwitchBackLinearGlobal`

`functools.partial(SwitchBackLinear, vector_wise_quantization=False)`
**Stability:** Experimental.

#### `SwitchBackLinearVectorwise`

`functools.partial(SwitchBackLinear, vector_wise_quantization=True)`
**Stability:** Experimental.

#### `StandardLinear` — Standard linear with explicit autograd

```
bitsandbytes.nn.StandardLinear
```

**Parent:** `torch.nn.Linear`
**Stability:** Experimental — utility/baseline.

### 2.3 Embedding Layers

#### `StableEmbedding` — Embedding with 32-bit optimizer states

```
bitsandbytes.nn.StableEmbedding(
    num_embeddings: int, embedding_dim: int,
    padding_idx=None, max_norm=None, norm_type=2.0,
    scale_grad_by_freq=False, sparse=False,
    _weight=None, device=None, dtype=None,
)
```

**Parent:** `torch.nn.Embedding`
**Stability:** Stable.
**Notes:** Xavier uniform init + LayerNorm applied after embedding lookup. Automatically
registers 32-bit optimizer override via `GlobalOptimManager`.

#### `Embedding` — Embedding with 32-bit optimizer states

```
bitsandbytes.nn.Embedding(
    num_embeddings: int, embedding_dim: int,
    padding_idx=None, max_norm=None, norm_type=2.0,
    scale_grad_by_freq=False, sparse=False,
    _weight=None, device=None,
)
```

**Parent:** `torch.nn.Embedding`
**Stability:** Stable.
**Notes:** Like StableEmbedding but without LayerNorm. Xavier uniform init. Registers
32-bit optimizer override.

#### `Embedding8bit` — Int8 quantized embedding

```
bitsandbytes.nn.Embedding8bit(
    num_embeddings, embedding_dim, device=None, dtype=None,
)
```

**Parent:** `torch.nn.Embedding`
**Stability:** Stable.
**Notes:** Weight stored as `Int8Params`. Saving (`_save_to_state_dict`) is NOT implemented
(raises `NotImplementedError`).

#### `Embedding4bit` — 4-bit quantized embedding

```
bitsandbytes.nn.Embedding4bit(
    num_embeddings, embedding_dim, dtype=None,
    quant_type="fp4", quant_storage=torch.uint8, device=None,
)
```

**Parent:** `torch.nn.Embedding`
**Stability:** Stable.
**Notes:** Weight stored as `Params4bit`. Uses partial dequantization when
`embedding_dim % blocksize == 0`. Saving is NOT implemented.

#### `EmbeddingFP4` — Convenience wrapper

```
bitsandbytes.nn.EmbeddingFP4(num_embeddings, embedding_dim, dtype=None, quant_storage=torch.uint8, device=None)
```

**Parent:** `Embedding4bit` with `quant_type="fp4"`.

#### `EmbeddingNF4` — Convenience wrapper

```
bitsandbytes.nn.EmbeddingNF4(num_embeddings, embedding_dim, dtype=None, quant_storage=torch.uint8, device=None)
```

**Parent:** `Embedding4bit` with `quant_type="nf4"`.

### 2.4 Parameter Types

#### `Params4bit` — 4-bit quantized parameter

```
bitsandbytes.nn.Params4bit(
    data: Optional[torch.Tensor] = None,
    requires_grad: bool = False,
    quant_state: Optional[QuantState] = None,
    blocksize: Optional[int] = None,        # default: 64 (128 on ROCm)
    compress_statistics: bool = True,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
    module: Optional[Linear4bit] = None,
    bnb_quantized: bool = False,
)
```

**Parent:** `torch.nn.Parameter`
**Stability:** Stable — essential for 4-bit workflows.
**Key behaviors:**
- `.to(device)` triggers quantization on first move to non-meta device
- `_quantize(device)` calls `bnb.functional.quantize_4bit`
- Custom `__torch_function__` for `torch.chunk` and `torch.split` to preserve quant state
- `from_prequantized(data, quantized_stats, ...)` class method for loading pre-quantized weights
- Custom `__deepcopy__`, `__copy__`, `__getstate__`, `__setstate__` for serialization
- `.cpu()`, `.cuda()`, `.xpu()` handle CPU packing format conversion

#### `Int8Params` — 8-bit quantized parameter

```
bitsandbytes.nn.Int8Params(
    data: Optional[torch.Tensor] = None,
    requires_grad: bool = True,
    has_fp16_weights: bool = False,
    CB: Optional[torch.Tensor] = None,
    SCB: Optional[torch.Tensor] = None,
)
```

**Parent:** `torch.nn.Parameter`
**Stability:** Stable — essential for 8-bit workflows.
**Key behaviors:**
- `.to(device)` triggers quantization if moving from CPU to non-meta device and not already quantized
- `_quantize(device)` calls `bnb.functional.int8_vectorwise_quant`
- `.CB` stores the int8 quantized data
- `.SCB` stores the per-row scaling factors
- `has_fp16_weights=True` skips quantization entirely

---

## 3. Optimizers

**Import path:** `from bitsandbytes.optim import <Class>`

All optimizers follow the same pattern: a base class that accepts `optim_bits` to control
32-bit vs 8-bit state, and concrete classes that fix the bit width. All support
`is_paged=True` for paged optimizers (offloading state to CPU via managed memory).

### 3.1 Base Classes

#### `GlobalOptimManager` — Singleton for per-parameter optimizer config overrides

```
bitsandbytes.optim.GlobalOptimManager.get_instance()
```

**Methods:**
- `register_parameters(params)` — Register parameters for config lookup
- `override_config(parameters, key=None, value=None, key_value_dict=None)` — Override optimizer hyperparams per parameter
- `register_module_override(module, param_name, config)` — Register module-level overrides

**Stability:** Stable — used by StableEmbedding, Embedding to force 32-bit states.

#### `Optimizer8bit` — Base class for all bnb optimizers

```
bitsandbytes.optim.optimizer.Optimizer8bit(params, defaults, optim_bits=32, is_paged=False)
```

**Parent:** `torch.optim.Optimizer`
**Stability:** Semi-public — users don't instantiate directly.
**Key features:**
- Custom `state_dict()` / `load_state_dict()` for FSDP compatibility
  (wraps quant state tensors in nested dict to prevent FSDP gather failures)
- `non_castable_tensor_keys`: set of state keys that should not be dtype-cast during load
- `is_paged`: enables CUDA managed memory for optimizer states
- `fill_qmap()`: initializes dynamic quantization maps

#### `Optimizer2State` — Base for 2-state optimizers (Adam, AdamW, LAMB, AdEMAMix)

```
bitsandbytes.optim.optimizer.Optimizer2State(
    optimizer_name, params, lr=1e-3, betas=(0.9, 0.999),
    eps=1e-8, weight_decay=0.0, optim_bits=32, args=None,
    min_8bit_size=4096, percentile_clipping=100,
    block_wise=True, max_unorm=0.0, skip_zeros=False,
    is_paged=False, alpha=0.0, t_alpha=None, t_beta3=None,
)
```

**Parent:** `Optimizer8bit`
**Stability:** Semi-public.

#### `Optimizer1State` — Base for 1-state optimizers (SGD, Adagrad, RMSprop, LARS, Lion)

```
bitsandbytes.optim.optimizer.Optimizer1State(
    optimizer_name, params, lr=1e-3, betas=(0.9, 0.0),
    eps=1e-8, weight_decay=0.0, optim_bits=32, args=None,
    min_8bit_size=4096, percentile_clipping=100,
    block_wise=True, max_unorm=0.0, skip_zeros=False,
    is_paged=False,
)
```

**Parent:** `Optimizer8bit`
**Stability:** Semi-public.

### 3.2 Concrete Optimizer Classes

All follow the naming pattern: `Name` (configurable bits), `Name8bit` (fixed 8-bit state),
`Name32bit` (fixed 32-bit state), `PagedName` (paged, configurable), `PagedName8bit`, `PagedName32bit`.

#### Adam Family (2-state, `optimizer_name="adam"`)

| Class | Parent | `optim_bits` | `is_paged` |
|-------|--------|-------------|------------|
| `Adam` | `Optimizer2State` | configurable (default 32) | `False` |
| `Adam8bit` | `Optimizer2State` | 8 (hardcoded) | `False` |
| `Adam32bit` | `Optimizer2State` | 32 (hardcoded) | `False` |
| `PagedAdam` | `Optimizer2State` | configurable (default 32) | `True` |
| `PagedAdam8bit` | `Optimizer2State` | 8 (hardcoded) | `True` |
| `PagedAdam32bit` | `Optimizer2State` | 32 (hardcoded) | `True` |

**Stability:** Stable.

#### AdamW Family (2-state, `optimizer_name="adam"`, decoupled weight decay)

| Class | Parent | `optim_bits` | `is_paged` |
|-------|--------|-------------|------------|
| `AdamW` | `Optimizer2State` | configurable | `False` |
| `AdamW8bit` | `Optimizer2State` | 8 | `False` |
| `AdamW32bit` | `Optimizer2State` | 32 | `False` |
| `PagedAdamW` | `Optimizer2State` | configurable | `True` |
| `PagedAdamW8bit` | `Optimizer2State` | 8 | `True` |
| `PagedAdamW32bit` | `Optimizer2State` | 32 | `True` |

**Stability:** Stable.

#### AdEMAMix Family (2-state, `optimizer_name="ademamix"`)

| Class | Parent | `optim_bits` | `is_paged` |
|-------|--------|-------------|------------|
| `AdEMAMix` | `Optimizer2State` | configurable | `False` |
| `AdEMAMix8bit` | `AdEMAMix` | 8 | `False` |
| `AdEMAMix32bit` | `Optimizer2State` | 32 | `False` |
| `PagedAdEMAMix` | `AdEMAMix` | configurable | `True` |
| `PagedAdEMAMix8bit` | `AdEMAMix8bit` | 8 | `True` |
| `PagedAdEMAMix32bit` | `AdEMAMix32bit` | 32 | `True` |

**Stability:** Stable.
**Notes:** Takes additional `betas=(beta1, beta2, beta3)`, `alpha`, `t_alpha`, `t_beta3` params.

#### LAMB Family (2-state, `optimizer_name="lamb"`)

| Class | Parent | `optim_bits` | `is_paged` |
|-------|--------|-------------|------------|
| `LAMB` | `Optimizer2State` | configurable | `False` |
| `LAMB8bit` | `Optimizer2State` | 8 | `False` |
| `LAMB32bit` | `Optimizer2State` | 32 | `False` |

**Stability:** Stable.

#### SGD Family (1-state, `optimizer_name="momentum"`)

| Class | Parent | `optim_bits` | `is_paged` |
|-------|--------|-------------|------------|
| `SGD` | `Optimizer1State` | configurable | `False` |
| `SGD8bit` | `Optimizer1State` | 8 | `False` |
| `SGD32bit` | `Optimizer1State` | 32 | `False` |

**Stability:** Stable.

#### Adagrad Family (1-state, `optimizer_name="adagrad"`)

| Class | Parent | `optim_bits` | `is_paged` |
|-------|--------|-------------|------------|
| `Adagrad` | `Optimizer1State` | configurable | `False` |
| `Adagrad8bit` | `Optimizer1State` | 8 | `False` |
| `Adagrad32bit` | `Optimizer1State` | 32 | `False` |

**Stability:** Stable.

#### RMSprop Family (1-state, `optimizer_name="rmsprop"`)

| Class | Parent | `optim_bits` | `is_paged` |
|-------|--------|-------------|------------|
| `RMSprop` | `Optimizer1State` | configurable | `False` |
| `RMSprop8bit` | `Optimizer1State` | 8 | `False` |
| `RMSprop32bit` | `Optimizer1State` | 32 | `False` |

**Stability:** Stable.

#### LARS Family (1-state, `optimizer_name="lars"`)

| Class | Parent | `optim_bits` | `is_paged` |
|-------|--------|-------------|------------|
| `LARS` | `Optimizer1State` | configurable | `False` |
| `LARS8bit` | `Optimizer1State` | 8 | `False` |
| `LARS32bit` | `Optimizer1State` | 32 | `False` |
| `PytorchLARS` | `torch.optim.Optimizer` | N/A | N/A |

**Stability:** Stable.
**Notes:** `PytorchLARS` is a pure-PyTorch reference implementation (not quantized).

#### Lion Family (1-state, `optimizer_name="lion"`)

| Class | Parent | `optim_bits` | `is_paged` |
|-------|--------|-------------|------------|
| `Lion` | `Optimizer1State` | configurable | `False` |
| `Lion8bit` | `Optimizer1State` | 8 | `False` |
| `Lion32bit` | `Optimizer1State` | 32 | `False` |
| `PagedLion` | `Optimizer1State` | configurable | `True` |
| `PagedLion8bit` | `Optimizer1State` | 8 | `True` |
| `PagedLion32bit` | `Optimizer1State` | 32 | `True` |

**Stability:** Stable.

### 3.3 Common Optimizer Parameters

All bnb optimizers share these parameters beyond the standard PyTorch ones:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optim_bits` | `int` | 32 | 32 for full precision state, 8 for quantized state |
| `min_8bit_size` | `int` | 4096 | Parameters smaller than this use 32-bit state even in 8-bit mode |
| `percentile_clipping` | `int` | 100 | Gradient clipping at a percentile. 100 = disabled |
| `block_wise` | `bool` | `True` | Block-wise quantization of optimizer states (vs global) |
| `max_unorm` | `float` | 0.0 | Maximum update norm relative to weight norm. 0 = disabled |
| `skip_zeros` | `bool` | `False` | Skip zero gradients in sparse models |
| `is_paged` | `bool` | `False` | Use CUDA managed memory for state offloading |

---

## 4. Functional API

**Import path:** `import bitsandbytes.functional as F` or `from bitsandbytes.functional import <symbol>`

### 4.1 4-Bit Quantization

#### `quantize_4bit`

```python
F.quantize_4bit(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,         # default: 64 (128 on ROCm)
    compress_statistics: bool = False,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, QuantState]
```

**Stability:** Stable.
**Supported dtypes:** float16, bfloat16, float32.
**Valid blocksizes:** 32, 64, 128, 256, 512, 1024, 2048, 4096.
**Quant types:** `"fp4"`, `"nf4"`.

#### `dequantize_4bit`

```python
F.dequantize_4bit(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    quant_type: str = "fp4",
) -> torch.Tensor
```

**Stability:** Stable.

#### `quantize_fp4` / `quantize_nf4`

Convenience wrappers that call `quantize_4bit` with the quant_type fixed.
**Stability:** Stable.

#### `dequantize_fp4` / `dequantize_nf4`

Convenience wrappers that call `dequantize_4bit` with the quant_type fixed.
**Stability:** Stable.

#### `get_4bit_type`

```python
F.get_4bit_type(typename: str, device=None, blocksize=64) -> torch.Tensor
```

Returns a 16-element codebook tensor for the given type name.
**Valid typenames:** `"nf4"`, `"fp4"`, `"int4"`, `"af4"` (af4 only supports blocksize 64).
**Stability:** Stable.

### 4.2 Blockwise (8-bit) Quantization

#### `quantize_blockwise`

```python
F.quantize_blockwise(
    A: torch.Tensor,
    code: Optional[torch.Tensor] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 4096,
    nested: bool = False,
) -> tuple[torch.Tensor, QuantState]
```

**Stability:** Stable.
**Supported dtypes:** float16, bfloat16, float32.
**Valid blocksizes:** 64, 128, 256, 512, 1024, 2048, 4096.

#### `dequantize_blockwise`

```python
F.dequantize_blockwise(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 4096,
    nested: bool = False,
) -> torch.Tensor
```

**Stability:** Stable.

### 4.3 Int8 Operations

#### `int8_vectorwise_quant`

```python
F.int8_vectorwise_quant(
    A: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
```

Returns `(quantized_int8, row_stats, outlier_cols_or_None)`.
**Stability:** Stable.
**Notes:** When `threshold > 0.0`, returns outlier column indices. This is the core of LLM.int8() decomposition.

#### `int8_vectorwise_dequant`

```python
F.int8_vectorwise_dequant(
    A: torch.Tensor,         # int8
    stats: torch.Tensor,     # float32 row stats
) -> torch.Tensor            # float32
```

**Stability:** Stable.

#### `int8_double_quant`

```python
F.int8_double_quant(
    A: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
```

Returns `(out_row, out_col, row_stats, col_stats, outlier_cols)`.
Performs both row-wise and column-wise int8 quantization simultaneously.
**Stability:** Stable.
**Notes:** Used in the backward pass of MatMul8bitLt when weight gradients are needed.

#### `int8_linear_matmul`

```python
F.int8_linear_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor
```

Int8 matrix multiplication: `A @ B.T` where both A and B are int8.
Returns int32 result.
**Stability:** Stable.

#### `int8_mm_dequant`

```python
F.int8_mm_dequant(
    A: torch.Tensor,              # int32 matmul result
    row_stats: torch.Tensor,      # float32
    col_stats: torch.Tensor,      # float32
    dtype: torch.dtype = torch.float16,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

Dequantizes the int32 result of int8 matmul using row and column statistics.
**Stability:** Stable.

### 4.4 QuantState

```python
class F.QuantState:
    valid_quant_types = ("fp4", "nf4")

    def __init__(self, absmax, shape=None, code=None, blocksize=None,
                 quant_type=None, dtype=None, offset=None, state2=None): ...

    @classmethod
    def from_dict(cls, qs_dict: dict, device: torch.device) -> QuantState: ...

    def as_dict(self, packed=False) -> dict: ...

    def to(self, device): ...

    def __eq__(self, other) -> bool: ...

    def __getitem__(self, idx): ...    # backward compatibility with list-based state
```

**Stability:** Stable — essential for serialization of quantized weights.
**Key attributes:**
- `absmax` — Per-block scaling factors
- `shape` — Original tensor shape
- `code` — Quantization codebook (16 values for 4-bit)
- `blocksize` — Block size used for quantization
- `quant_type` — `"fp4"` or `"nf4"`
- `dtype` — Original tensor dtype
- `offset` — Mean of absmax (used in double quantization / `compress_statistics`)
- `state2` — Nested QuantState for doubly-quantized absmax
- `nested` — `True` if `state2` is not None

### 4.5 Quantization Map Constructors

#### `create_dynamic_map`

```python
F.create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8) -> torch.Tensor
```

Creates a 256-element dynamic quantization codebook. This is the default
codebook used by blockwise quantization.
**Stability:** Stable.

#### `create_normal_map`

```python
F.create_normal_map(offset=0.9677083, use_extra_value=True) -> torch.Tensor
```

Creates the NF4 quantization codebook (16 values + padding to 256).
**Stability:** Stable.
**Notes:** Requires scipy for the `norm.ppf` call. The hardcoded NF4 values in
`get_4bit_type("nf4")` avoid this dependency at runtime.

#### `create_fp8_map`

```python
F.create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8) -> torch.Tensor
```

Creates a floating-point quantization codebook. Despite the name, works for
any `total_bits` (including FP4 with `total_bits=4`).
**Stability:** Stable.

#### `create_linear_map`

```python
F.create_linear_map(signed=True, total_bits=8, add_zero=True) -> torch.Tensor
```

Creates a uniform linear quantization codebook.
**Stability:** Stable.

### 4.6 4-Bit GEMV

#### `gemv_4bit`

```python
F.gemv_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A: bool = False,
    transposed_B: bool = False,
    state: QuantState = None,              # required
) -> torch.Tensor
```

Efficient matrix-vector product with 4-bit quantized weight matrix.
Used for single-batch inference in `matmul_4bit`.
**Stability:** Stable.
**Supported dtypes for A:** float16, bfloat16, float32.

### 4.7 Optimizer Update Functions

#### `optimizer_update_32bit`

```python
F.optimizer_update_32bit(
    optimizer_name: str, g: Tensor, p: Tensor, state1: Tensor,
    beta1: float, eps: float, step: int, lr: float,
    state2: Optional[Tensor] = None, beta2: float = 0.0,
    beta3: float = 0.0, alpha: float = 0.0,
    weight_decay: float = 0.0, gnorm_scale: float = 1.0,
    unorm_vec: Optional[Tensor] = None, max_unorm: float = 0.0,
    skip_zeros: bool = False,
) -> None
```

In-place optimizer step with 32-bit state.
**Stability:** Stable.
**Valid optimizer names:** `"adam"`, `"momentum"`, `"rmsprop"`, `"lion"`, `"adagrad"`, `"ademamix"`, `"lamb"`, `"lars"`.

#### `optimizer_update_8bit_blockwise`

```python
F.optimizer_update_8bit_blockwise(
    optimizer_name: str, g: Tensor, p: Tensor,
    state1: Tensor, state2: Optional[Tensor],
    beta1: float, beta2: float, beta3: float, alpha: float,
    eps: float, step: int, lr: float,
    qmap1: Tensor, qmap2: Optional[Tensor],
    absmax1: Tensor, absmax2: Optional[Tensor],
    weight_decay: float = 0.0, gnorm_scale: float = 1.0,
    skip_zeros: bool = False,
) -> None
```

In-place optimizer step with 8-bit blockwise-quantized state.
**Stability:** Stable.

### 4.8 Integer GEMM

#### `igemm`

```python
F.igemm(
    A: Tensor, B: Tensor, out: Optional[Tensor] = None,
    transposed_A: bool = False, transposed_B: bool = False,
) -> torch.Tensor
```

Int8 matrix multiplication via cuBLAS igemm.
**Stability:** Stable (internal, used by the library).

#### `batched_igemm`

```python
F.batched_igemm(
    A: Tensor, B: Tensor, out: Optional[Tensor] = None,
    transposed_A: bool = False, transposed_B: bool = False,
) -> torch.Tensor
```

Batched int8 matrix multiplication.
**Stability:** Stable (internal).

### 4.9 Sparse Operations

#### `COOSparseTensor`

```python
class F.COOSparseTensor:
    def __init__(self, rows, cols, nnz, rowidx, colidx, values): ...
```

**Stability:** Legacy — used internally for sparse decomposition.

#### `CSRSparseTensor` / `CSCSparseTensor`

Similar sparse tensor containers.
**Stability:** Legacy.

#### `coo_zeros`

```python
F.coo_zeros(rows, cols, nnz, device, dtype=torch.half) -> COOSparseTensor
```

#### `coo2csr` / `coo2csc`

```python
F.coo2csr(cooA: COOSparseTensor) -> CSRSparseTensor
F.coo2csc(cooA: COOSparseTensor) -> CSCSparseTensor
```

#### `spmm_coo`

```python
F.spmm_coo(
    cooA: COOSparseTensor, B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

Sparse matrix-dense matrix multiply using cusparse.
**Stability:** Legacy.

#### `spmm_coo_very_sparse`

```python
F.spmm_coo_very_sparse(cooA, B, dequant_stats=None, out=None) -> torch.Tensor
```

Optimized for very sparse matrices with custom kernel.
**Stability:** Legacy.

### 4.10 Paged Memory

#### `get_paged`

```python
F.get_paged(*shape, dtype=torch.float32, device=FIRST_CUDA_DEVICE) -> torch.Tensor
```

Allocates a CUDA managed-memory tensor.
**Stability:** Stable (internal, used by paged optimizers).

#### `prefetch_tensor`

```python
F.prefetch_tensor(A: torch.Tensor, to_cpu: bool = False) -> None
```

Prefetch a paged tensor to GPU or CPU.
**Stability:** Stable (internal).

### 4.11 CPU-Specific Functions

#### `_convert_weight_packed_for_cpu`

```python
F._convert_weight_packed_for_cpu(
    qweight: torch.Tensor, quant_state: QuantState, block_n: int = 32,
) -> tuple[torch.Tensor, QuantState]
```

Converts 4-bit quantized weights to a packed format optimized for CPU AVX512BF16 inference.
**Stability:** Internal (prefixed with `_`).

#### `_convert_weight_packed_for_cpu_inverse`

```python
F._convert_weight_packed_for_cpu_inverse(
    qweight: torch.Tensor, quant_state: QuantState,
) -> tuple[torch.Tensor, QuantState]
```

Reverses the CPU packing format.
**Stability:** Internal (prefixed with `_`).

#### `has_avx512bf16`

```python
F.has_avx512bf16() -> bool
```

Detects AVX512BF16 CPU support.
**Stability:** Internal but may be useful externally.

### 4.12 Utility Functions

#### `is_on_gpu`

```python
F.is_on_gpu(tensors: Iterable[Optional[torch.Tensor]]) -> bool
```

Verifies all tensors are on the same GPU. Raises RuntimeError if not.
**Stability:** Stable (internal validation).

#### `get_ptr`

```python
F.get_ptr(A: Optional[Tensor]) -> Optional[ct.c_void_p]
```

Gets the data pointer of a tensor for ctypes calls.
**Stability:** Internal.

### 4.13 Singleton Managers

#### `GlobalPageManager`

```python
F.GlobalPageManager.get_instance() -> GlobalPageManager
```

Manages paged tensors for prefetching.
**Stability:** Internal.

#### `CUBLAS_Context`

```python
F.CUBLAS_Context.get_instance() -> CUBLAS_Context
```

Manages cuBLAS context handles per device.
**Stability:** Internal.

#### `Cusparse_Context`

```python
F.Cusparse_Context.get_instance() -> Cusparse_Context
```

Manages cusparse context handle.
**Stability:** Internal.

---

## 5. Autograd Functions

**Import path:** `from bitsandbytes.autograd._functions import <symbol>`

Top-level re-exports: `bnb.matmul`, `bnb.matmul_4bit`, `bnb.MatmulLtState`.

### `MatmulLtState` — State container for 8-bit matmul

```python
@dataclass
class MatmulLtState:
    CB: Optional[torch.Tensor] = None
    SCB: Optional[torch.Tensor] = None
    threshold: float = 0.0
    has_fp16_weights: bool = True
    is_training: bool = True
    use_pool: bool = False
    ...
```

**Stability:** Stable.
**Key fields:**
- `CB` / `SCB` — Quantized weight and scale columns
- `threshold` — Outlier threshold for mixed-precision decomposition
- `has_fp16_weights` — Whether weights are stored in fp16 or int8
- `is_training` — Switches between training and inference code paths

### `matmul` — 8-bit matrix multiplication

```python
bnb.matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    state: Optional[MatmulLtState] = None,
    threshold: float = 0.0,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**Stability:** Stable.
**Dispatches to:**
- `MatMul8bitFp` on CPU/XPU during training (faster path, no quantized grad computation)
- `MatMul8bitLt` elsewhere (full quantized matmul with backward support)

### `matmul_4bit` — 4-bit matrix multiplication

```python
bnb.matmul_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    quant_state: F.QuantState,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**Stability:** Stable.
**Dispatches to:**
- `F.gemv_4bit` for single-batch inference (fast path, no autograd)
- `MatMul4Bit.apply` for batched/training (autograd-enabled, dequant + torch.matmul)
- CPU path supports packed weight format for AVX512BF16

### Internal autograd classes

| Class | Description | Stability |
|-------|-------------|-----------|
| `MatMul8bitLt` | Full 8-bit matmul with backward for weight and input grad | Internal |
| `MatMul8bitFp` | Dequant + matmul path for CPU/XPU training | Internal |
| `MatMul4Bit` | Dequant + matmul with backward for 4-bit weights | Internal |
| `GlobalOutlierPooler` | Pools outlier dimensions across layers | Internal |

---

## 6. Torch Custom Ops

**Module:** `bitsandbytes._ops`

These are defined via `torch.library.define` and provide the contract between
the functional API and backend implementations. Each op has a `register_fake`
implementation for `torch.compile` / FX tracing.

### Op Schema Table

| Op Name | Signature | Description |
|---------|-----------|-------------|
| `bitsandbytes::int8_mixed_scaled_mm` | `(A, CA, CB, SCA, SCB, outlier_cols?, bias?) -> (Tensor, Tensor?)` | Int8 matmul with mixed-precision outlier handling |
| `bitsandbytes::int8_scaled_mm` | `(A, B, row_stats, col_stats, bias?, dtype?) -> Tensor` | Int8 matmul + dequant + bias |
| `bitsandbytes::int8_linear_matmul` | `(A, B) -> Tensor` | Raw int8 matmul (A, B are int8, result is int32) |
| `bitsandbytes::int8_linear_matmul.out` | `(A, B, out!) -> ()` | In-place variant |
| `bitsandbytes::int8_vectorwise_quant` | `(A, threshold=0.0) -> (Tensor, Tensor, Tensor?)` | Row-wise int8 quantization with optional outlier extraction |
| `bitsandbytes::int8_vectorwise_dequant` | `(A, stats) -> Tensor` | Row-wise int8 dequantization |
| `bitsandbytes::int8_mm_dequant` | `(A, row_stats, col_stats, dtype?, bias?) -> Tensor` | Dequantize int32 matmul result |
| `bitsandbytes::int8_double_quant` | `(A, threshold=0.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor?)` | Simultaneous row and column quantization |
| `bitsandbytes::quantize_4bit` | `(A, blocksize, quant_type, quant_storage) -> (Tensor, Tensor)` | 4-bit blockwise quantization |
| `bitsandbytes::dequantize_4bit` | `(A, absmax, blocksize, quant_type, shape, dtype) -> Tensor` | 4-bit blockwise dequantization |
| `bitsandbytes::dequantize_4bit.out` | `(A, absmax, blocksize, quant_type, shape, dtype, out!) -> ()` | In-place variant |
| `bitsandbytes::quantize_blockwise` | `(A, code, blocksize) -> (Tensor, Tensor)` | 8-bit blockwise quantization |
| `bitsandbytes::dequantize_blockwise` | `(A, absmax, code, blocksize, dtype) -> Tensor` | 8-bit blockwise dequantization |
| `bitsandbytes::dequantize_blockwise.out` | `(A, absmax, code, blocksize, dtype, out!) -> ()` | In-place variant |
| `bitsandbytes::gemv_4bit` | `(A, B, shapeB, absmax, code, blocksize) -> Tensor` | 4-bit GEMV (matrix-vector product) |
| `bitsandbytes::gemv_4bit.out` | `(A, B, shapeB, absmax, code, blocksize, out!) -> ()` | In-place variant |
| `bitsandbytes::optimizer_update_32bit` | `(name, g!, p!, state1!, state2!?, ...) -> ()` | 32-bit optimizer step |
| `bitsandbytes::optimizer_update_8bit_blockwise` | `(name, g!, p!, state1!, state2!?, ...) -> ()` | 8-bit blockwise optimizer step |

**Stability:** Semi-public. The op schemas are the most important stability contract in
the codebase — changing a schema breaks all backend implementations.

### Default Implementations

`int8_vectorwise_dequant` has a default PyTorch-native implementation registered in `_ops.py`
itself (simple `A * stats * (1/127)`). All other ops must be implemented by backends.

---

## 7. Research / Experimental

**Import path:** `from bitsandbytes.research import <symbol>`

### Research Functions

```python
from bitsandbytes.research import matmul_fp8_global, matmul_fp8_mixed, switchback_bnb
```

#### `matmul_fp8_global`

```python
bitsandbytes.research.matmul_fp8_global(
    A, B, fw_code, bw_code, bsz, bsz2,
) -> torch.Tensor
```

FP8 matmul with global quantization.
**Stability:** Experimental.

#### `matmul_fp8_mixed`

```python
bitsandbytes.research.matmul_fp8_mixed(
    A, B, fw_code, bw_code, bsz, bsz2,
) -> torch.Tensor
```

FP8 matmul with mixed (row-wise) quantization.
**Stability:** Experimental.

#### `switchback_bnb`

```python
bitsandbytes.research.switchback_bnb(
    A, B, out=None, bias=None, state=MatmulLtState,
) -> torch.Tensor
```

SwitchBack-style matmul using bnb backend.
**Stability:** Experimental.

### Research NN Modules

```python
from bitsandbytes.research.nn import LinearFP8Mixed, LinearFP8Global
```

#### `LinearFP8Mixed` / `LinearFP8Global`

```python
bitsandbytes.research.nn.LinearFP8Mixed(input_features, output_features, bias=True)
bitsandbytes.research.nn.LinearFP8Global(input_features, output_features, bias=True)
```

**Parent:** `torch.nn.Linear`
**Stability:** Experimental.
**Notes:** Automatically select block sizes based on feature dimensions. Use FP8
quantization maps created via `create_fp8_map`.

---

## 8. Utilities

**Import path:** `from bitsandbytes.utils import <symbol>`

| Symbol | Type | Description | Stability |
|--------|------|-------------|-----------|
| `replace_linear` | function | Recursively replace `nn.Linear` modules in a model | Stable |
| `OutlierTracer` | class (singleton) | Traces outlier dimensions across linear layers | Experimental |
| `find_outlier_dims` | function | Find outlier dimensions via z-score or top-k | Experimental |
| `outlier_hook` | function | Forward pre-hook for `OutlierTracer` | Internal |
| `pack_dict_to_tensor` | function | Pack a dict into a uint8 tensor (for safetensors) | Stable (internal) |
| `unpack_tensor_to_dict` | function | Unpack uint8 tensor back to dict | Stable (internal) |
| `execute_and_return` | function | Run a shell command and return stdout/stderr | Internal |
| `sync_gpu` | function | Synchronize CUDA/XPU device | Internal |
| `LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING` | dict | Maps format names to int codes | Stable (internal) |
| `INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING` | dict | Reverse mapping | Stable (internal) |

### `replace_linear`

```python
bitsandbytes.utils.replace_linear(
    model: torch.nn.Module,
    linear_replacement: type,
    skip_modules: tuple = ("lm_head",),
    copy_weights: bool = False,
    post_processing_function: Optional[str] = None,
) -> torch.nn.Module
```

**Stability:** Stable — commonly used by integrations.

---

## 9. Native Library Interface

**Module:** `bitsandbytes.cextension`

### Classes

| Class | Description |
|-------|-------------|
| `BNBNativeLibrary` | Base wrapper for the ctypes-loaded native library |
| `CudaBNBNativeLibrary` | CUDA-specific subclass (sets up context/cusparse/managed ptr) |
| `ErrorHandlerMockBNBNativeLibrary` | Fallback mock that defers error messages to call time |

### Module-level symbols

| Symbol | Type | Description |
|--------|------|-------------|
| `lib` | `BNBNativeLibrary` | The loaded native library instance |
| `BNB_BACKEND` | `str` | `"CUDA"`, `"ROCm"`, `"XPU"`, or `"CPU"` |
| `HIP_ENVIRONMENT` | `bool` | `True` if running on ROCm |
| `ROCM_GPU_ARCH` | `str` or `None` | e.g., `"gfx90a"` |
| `ROCM_WARP_SIZE_64` | `bool` | `True` if ROCm warp size is 64 |

**Stability:** Internal — but `lib` is used extensively by `functional.py` for ctypes calls.

---

## 10. Backend System

**Module:** `bitsandbytes.backends`

Backends provide device-specific implementations of the ops defined in `_ops.py`.
Each backend registers kernels via `@register_kernel("bitsandbytes::<op_name>", "<device>")`.

### Backend → Op Coverage Matrix

| Op | `default` | `cuda` | `cpu` | `xpu` | `hpu` | `triton` |
|----|-----------|--------|-------|-------|-------|----------|
| `int8_linear_matmul` | Yes | Yes | Yes | Yes | — | — |
| `int8_linear_matmul.out` | Yes | Yes | — | — | — | — |
| `int8_vectorwise_quant` | Yes | Yes | — | — | — | — |
| `int8_vectorwise_dequant` | (in _ops.py) | — | — | — | — | — |
| `int8_mm_dequant` | Yes | Yes | — | — | — | — |
| `int8_mixed_scaled_mm` | Yes | — | — | — | — | — |
| `int8_scaled_mm` | Yes | — | — | — | — | — |
| `int8_double_quant` | — | Yes | — | — | — | — |
| `quantize_blockwise` | Yes | Yes | Yes | Yes | — | Yes |
| `dequantize_blockwise` | Yes | Yes | Yes | Yes | — | Yes |
| `dequantize_blockwise.out` | — | Yes | — | Yes | — | — |
| `quantize_4bit` | Yes | Yes | — | Yes | — | Yes |
| `dequantize_4bit` | Yes | Yes | Yes | Yes | Yes | Yes |
| `dequantize_4bit.out` | — | Yes | — | Yes | — | Yes |
| `gemv_4bit` | Yes | Yes | Yes | Yes | — | Yes |
| `gemv_4bit.out` | — | Yes | — | Yes | — | — |
| `optimizer_update_32bit` | Yes | Yes | — | Yes | — | Yes |
| `optimizer_update_8bit_blockwise` | — | Yes | — | Yes | — | Yes |

**Notes:**
- `default` backend is pure PyTorch (no native code), registered for any device
- `cuda` backend uses ctypes calls to the native CUDA/HIP library
- `cpu` backend uses ctypes calls to the CPU native library (limited coverage)
- `xpu` backend uses triton kernels when available, ctypes fallback otherwise
- `hpu` backend only covers `dequantize_4bit` (Intel Gaudi)
- `triton` backend is not registered directly; XPU imports its implementations

### External Backend Entry Points

Third-party packages can register backends via the `bitsandbytes.backends` entry point
group in their `pyproject.toml`. This is how the MPS (Apple Silicon) backend is expected
to be distributed.

---

## 11. Deprecated Symbols

These symbols are marked with `@deprecated` and emit `FutureWarning`. They will be
removed in a future release.

| Symbol | Module | Replacement |
|--------|--------|-------------|
| `quantize` | `functional` | `quantize_blockwise` |
| `dequantize` | `functional` | `dequantize_blockwise` |
| `quantize_no_absmax` | `functional` | `quantize_blockwise` |
| `dequantize_no_absmax` | `functional` | `dequantize_blockwise` |
| `optimizer_update_8bit` | `functional` | `optimizer_update_8bit_blockwise` |
| `percentile_clipping` | `functional` | N/A (still used internally by non-blockwise path) |

---

## 12. Downstream Integration Points

These are the specific API surfaces that downstream libraries (transformers, PEFT,
accelerate, etc.) depend on. Changes here have the highest breakage risk.

### Used by HuggingFace `transformers`

- `bnb.nn.Linear4bit` — Instantiated by `BitsAndBytesConfig(load_in_4bit=True)`
- `bnb.nn.Linear8bitLt` — Instantiated by `BitsAndBytesConfig(load_in_8bit=True)`
- `bnb.nn.Params4bit` — Used for weight loading and quantization
- `bnb.nn.Int8Params` — Used for weight loading and quantization
- `bnb.nn.Params4bit.from_prequantized()` — Loading pre-quantized weights
- `bnb.functional.QuantState` — Serialization/deserialization of quant states
- `bnb.functional.QuantState.from_dict()` / `.as_dict()` — State dict handling
- `bnb.features` — Feature detection (`"multi_backend"` in `bnb.features`)
- `bnb.supported_torch_devices` — Device support detection
- `bnb.__version__` — Version checks
- `bnb.utils.replace_linear` — Model conversion

### Used by PEFT / LoRA

- `bnb.nn.Linear4bit` — Base layer for QLoRA adapters
- `bnb.nn.Params4bit` — Parameter type checks
- `bnb.nn.Linear8bitLt` — Base layer for 8-bit LoRA

### Used by `accelerate`

- `bnb.optim.*` — Paged optimizers for DeepSpeed/FSDP
- `Optimizer8bit.state_dict()` / `load_state_dict()` — FSDP compatibility

### Integration Contract Summary

A PR that changes any of these symbols MUST consider downstream impact:

1. **`Linear4bit` constructor signature** — changing defaults breaks `BitsAndBytesConfig`
2. **`Params4bit.__new__` signature** — changing parameter order breaks weight loading
3. **`QuantState` serialization format** — changes break loading saved models
4. **Op schemas in `_ops.py`** — changes break ALL backend implementations
5. **`features` / `supported_torch_devices`** — changes break feature detection in transformers

---

## 13. Stability Tiers

### Tier 1: Stable Public API (breaking changes require deprecation cycle)

- `bnb.nn.Linear4bit`, `LinearFP4`, `LinearNF4`
- `bnb.nn.Linear8bitLt`
- `bnb.nn.Params4bit`, `Int8Params`
- `bnb.nn.Embedding`, `StableEmbedding`, `Embedding4bit`, `Embedding8bit`, `EmbeddingFP4`, `EmbeddingNF4`
- `bnb.functional.quantize_4bit`, `dequantize_4bit`
- `bnb.functional.quantize_blockwise`, `dequantize_blockwise`
- `bnb.functional.QuantState` (including serialization format)
- `bnb.functional.int8_vectorwise_quant`, `int8_double_quant`, `int8_mm_dequant`
- `bnb.matmul`, `bnb.matmul_4bit`, `bnb.MatmulLtState`
- All optimizer classes in `bnb.optim.*`
- `bnb.optim.GlobalOptimManager`
- `bnb.utils.replace_linear`
- `bnb.features`, `bnb.supported_torch_devices`, `bnb.__version__`

### Tier 2: Semi-Public (may change between minor versions)

- Op schemas in `_ops.py` (stable within a minor version, but may evolve)
- `bnb.functional.create_*_map` functions
- `bnb.functional.get_4bit_type`
- `bnb.functional.gemv_4bit`
- `bnb.functional.int8_linear_matmul`
- `bnb.functional.igemm`, `batched_igemm`
- Backend registration system (`register_kernel` pattern)
- `Optimizer8bit`, `Optimizer1State`, `Optimizer2State` base classes

### Tier 3: Experimental (may change or be removed at any time)

- Everything in `bitsandbytes.research.*`
- `bnb.nn.SwitchBackLinear*` (triton-based)
- `bnb.nn.SwitchBackLinearBnb`
- `bnb.nn.OutlierAwareLinear`
- `bnb.nn.StandardLinear`
- `bnb.utils.OutlierTracer`, `find_outlier_dims`

### Tier 4: Internal (not part of public API, may change freely)

- `bitsandbytes.cextension.*` (native library loading)
- `bitsandbytes.functional.get_ptr`, `is_on_gpu`, `_get_tensor_stream`
- `bitsandbytes.functional.GlobalPageManager`, `CUBLAS_Context`, `Cusparse_Context`
- `bitsandbytes.functional._convert_weight_packed_for_cpu*`
- `bitsandbytes.functional.check_matmul`, `elementwise_func`, `fill`, `_mul`
- `bitsandbytes.functional.spmm_coo`, `spmm_coo_very_sparse`
- `bitsandbytes.functional.COOSparseTensor`, `CSRSparseTensor`, `CSCSparseTensor`
- `bitsandbytes.utils.pack_dict_to_tensor`, `unpack_tensor_to_dict`
- `bitsandbytes.utils.execute_and_return`, `sync_gpu`
- `bitsandbytes.optim.optimizer.MockArgs`
- All backend implementation files (`backends/*/ops.py`)
- All CUDA/C++ code (`csrc/*`)
