# bitsandbytes Code Standards

This document defines the coding standards, patterns, and conventions for the bitsandbytes
codebase. It is written for agents reviewing pull requests or writing code — it captures
what an experienced maintainer knows about "how code should look" in this project, beyond
what automated linters check.

For automated linting rules, see `agents/linting_guide.md`. For architecture, see
`agents/architecture_guide.md`. This document covers the _semantic_ standards: what patterns
to follow, what to avoid, how to name things, how to validate inputs, and how to write tests.

---

## Table of Contents

1. [Python Conventions](#1-python-conventions)
2. [The Op Registry Pattern (`_ops.py`)](#2-the-op-registry-pattern-_opspy)
3. [Backend Implementation Pattern](#3-backend-implementation-pattern)
4. [The Functional Layer Pattern (`functional.py`)](#4-the-functional-layer-pattern-functionalpy)
5. [Neural Network Module Conventions (`nn/`)](#5-neural-network-module-conventions-nn)
6. [Optimizer Conventions (`optim/`)](#6-optimizer-conventions-optim)
7. [Input Validation Rules](#7-input-validation-rules)
8. [Error Handling](#8-error-handling)
9. [Tensor Immutability and Side Effects](#9-tensor-immutability-and-side-effects)
10. [ctypes / Native Library Calling Convention](#10-ctypes--native-library-calling-convention)
11. [CUDA Device Management](#11-cuda-device-management)
12. [CUDA/C++ Kernel Conventions (`csrc/`)](#12-cudac-kernel-conventions-csrc)
13. [Test Conventions](#13-test-conventions)
14. [Deprecation Protocol](#14-deprecation-protocol)
15. [API Design Rules](#15-api-design-rules)
16. [Dependency Policy](#16-dependency-policy)
17. [Common Anti-Patterns to Reject](#17-common-anti-patterns-to-reject)
18. [Performance Expectations](#18-performance-expectations)
19. [Documentation Standards](#19-documentation-standards)
20. [Serialization and State Dict Conventions](#20-serialization-and-state-dict-conventions)

---

## 1. Python Conventions

### 1.1 Formatting and Style

All Python code is auto-formatted by `ruff format` and linted by `ruff check`. The
authoritative configuration is in `pyproject.toml`. Key settings:

- **Line length**: 119 characters
- **Target Python version**: 3.10 (minimum supported)
- **Import ordering**: isort via ruff, with `bitsandbytes` as known-first-party

Do not fight the formatter. If ruff wraps a line in a way that looks odd, that is the
project's style. Do not add `# fmt: off` or `# noqa` comments unless there is a genuine
reason the tool is wrong.

### 1.2 Import Conventions

Imports follow a strict ordering enforced by isort:

1. Standard library
2. Third-party packages (`torch`, `numpy`, etc.)
3. First-party (`bitsandbytes`, `bitsandbytes.functional`, etc.)

Within the codebase:

```python
# GOOD: import the module, use qualified names
import bitsandbytes.functional as F
result = F.quantize_4bit(...)

# GOOD: explicit imports from submodules
from bitsandbytes.functional import QuantState, get_ptr

# AVOID: star imports
from bitsandbytes.functional import *  # Never do this
```

The top-level `__init__.py` re-exports key symbols. Backend modules import from their
relative parents:

```python
# In backends/cuda/ops.py:
from ..._ops import register_kernel
from ...cextension import ROCM_WARP_SIZE_64, lib
```

### 1.3 Type Annotations

- Use `Optional[X]` (not `X | None`) — the ruff config explicitly ignores `UP045`
- Use `typing.Optional`, `typing.Any` from the `typing` module
- Use `collections.abc.Sequence` for sequence type hints (not `typing.Sequence`)
- Use built-in generics where possible: `list[int]`, `tuple[str, ...]`, `dict[str, Any]`
- Function signatures in `_ops.py` (op schemas) **must** have full type annotations
- Backend implementations should match the signature of the op schema exactly
- Type annotations on internal helper functions are optional but encouraged

```python
# GOOD: matches the conventions used throughout
def quantize_4bit(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=None,  # no annotation for simple defaults is OK
    compress_statistics=False,
    quant_type="fp4",
    quant_storage=torch.uint8,
) -> tuple[torch.Tensor, QuantState]:
```

### 1.4 Naming Conventions

**Functions**:
- Public API functions in `functional.py`: `snake_case` — `quantize_4bit`, `dequantize_blockwise`
- Internal helpers: prefix with `_` — `_dequantize_4bit_impl`, `_get_col_absmax`
- ctypes C function wrappers start with `c`: `lib.cquantize_blockwise_fp16`

**Variables**:
- Tensor variables use short uppercase names by convention: `A`, `B`, `CB`, `SCB`, `SCA`
- This is a deliberate style choice reflecting the mathematical notation in the papers
- Statistics tensors: `row_stats`, `col_stats`, `absmax`
- Output tensors: `out`, `output`
- Shape-related: `shapeA`, `shapeB`, `shapeC`

**Classes**:
- `PascalCase`: `QuantState`, `MatmulLtState`, `Params4bit`, `Int8Params`
- Singletons use the pattern: private `__init__` that raises, classmethod `get_instance()`
- Module classes: `Linear4bit`, `Linear8bitLt`, `Embedding4bit`, `Embedding8bit`
- Optimizer classes: `Adam`, `Adam8bit`, `Adam32bit`, `PagedAdam`, `PagedAdam8bit`

**Constants**:
- `UPPER_SNAKE_CASE`: `FIRST_CUDA_DEVICE`, `ROCM_WARP_SIZE_64`, `HIP_ENVIRONMENT`
- Compute capability constants in C: `BNB_CC_VOLTA`, `BNB_CC_AMPERE`, etc.

### 1.5 Singleton Pattern

Several manager classes use a singleton pattern. Follow this exact structure:

```python
class GlobalOptimManager:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.some_state = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance
```

This pattern is used by: `GlobalOptimManager`, `GlobalPageManager`, `CUBLAS_Context`,
`Cusparse_Context`, `GlobalOutlierPooler`, `OutlierTracer`.

---

## 2. The Op Registry Pattern (`_ops.py`)

### 2.1 How to Define a New Op

Every operation that crosses the Python-to-native boundary goes through PyTorch's custom
op system. The pattern has three parts:

**Step 1: Define the op schema** in `_ops.py`:

```python
torch.library.define(
    "bitsandbytes::my_new_op",
    "(Tensor A, Tensor B, int blocksize, str quant_type) -> Tensor",
)
```

Schema rules:
- The namespace is always `bitsandbytes::`
- Use PyTorch schema syntax: `Tensor`, `Tensor?` (optional), `int`, `float`, `str`,
  `bool`, `ScalarType`, `int[]`, `Tensor!` (mutated in-place)
- Optional tensor arguments use `Tensor? name=None`
- Mutated tensors (in-place ops) use `Tensor(a0!) name` with aliasing annotations

**Step 2: Define the fake (meta) implementation** in `_ops.py`:

```python
@register_fake("bitsandbytes::my_new_op")
def _(A: torch.Tensor, B: torch.Tensor, blocksize: int, quant_type: str) -> torch.Tensor:
    # Validate inputs using torch._check (NOT assert)
    torch._check_is_size(blocksize)
    torch._check(A.dtype in [torch.float16, torch.bfloat16, torch.float32],
                 lambda: f"A must be float16/bfloat16/float32, got {A.dtype}")

    # Return an empty tensor of the correct shape/dtype/device
    return torch.empty(A.shape, dtype=A.dtype, device=A.device)
```

The fake implementation is critical for `torch.compile` and `torch.export`. It must:
- Validate all input constraints using `torch._check` (see Section 7)
- Return tensors with the **exact** correct shape, dtype, and device
- Never perform actual computation
- Handle dynamic shapes using `torch.library.get_ctx().new_dynamic_size()` when output
  size depends on data (e.g., outlier column detection)

**Step 3: Define the `.out` variant** (when applicable):

```python
torch.library.define(
    "bitsandbytes::my_new_op.out",
    "(Tensor A, Tensor B, int blocksize, str quant_type, Tensor! out) -> ()",
)

@register_fake("bitsandbytes::my_new_op.out")
def _(A: torch.Tensor, B: torch.Tensor, blocksize: int, quant_type: str, out: torch.Tensor):
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == A.dtype, lambda: f"Expected out.dtype == {A.dtype}, got {out.dtype}")
```

### 2.2 Compatibility Shim

The codebase supports PyTorch 2.3+. The API names changed in PyTorch 2.4:

```python
# This shim is at the top of _ops.py:
if hasattr(torch.library, "register_fake"):
    _IS_TORCH_GTE_24 = True
    register_fake = torch.library.register_fake
    register_kernel = torch.library.register_kernel
else:
    register_fake = torch.library.impl_abstract
    register_kernel = torch.library.impl
```

Always use the module-level `register_fake` and `register_kernel` from `_ops.py`, never
the `torch.library` methods directly.

### 2.3 Naming Convention for Anonymous Functions

The `@register_fake` and `@register_kernel` decorated functions are conventionally named
`_` (underscore) because they are not called directly — PyTorch dispatches to them:

```python
@register_fake("bitsandbytes::quantize_4bit")
def _(A: torch.Tensor, blocksize: int, ...) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

This is the established pattern throughout the codebase. Do not give these functions
descriptive names.

---

## 3. Backend Implementation Pattern

### 3.1 Structure

Each backend lives in `bitsandbytes/backends/<name>/ops.py`. A backend registers kernel
implementations for ops defined in `_ops.py`:

```python
# In backends/cuda/ops.py:
from ..._ops import register_kernel

@register_kernel("bitsandbytes::my_new_op", "cuda")
def _(A: torch.Tensor, B: torch.Tensor, blocksize: int, quant_type: str) -> torch.Tensor:
    # Actual CUDA implementation
    ...
```

The dispatch key strings are:
- `"cuda"` — NVIDIA CUDA and AMD ROCm
- `"cpu"` — CPU
- `"default"` — PyTorch-native fallback (works on any device)
- `"xpu"` — Intel GPU
- `"hpu"` — Intel Gaudi
- `"mps"` — Apple Silicon

### 3.2 Implementation Hierarchy

**Three levels of implementation exist for each op:**

1. **`default` backend** (`backends/default/ops.py`): Pure PyTorch implementation. Works
   on any device. Used as fallback. Often uses `@_try_torch_compile` for performance.

2. **`cpu` backend** (`backends/cpu/ops.py`): Uses C++ native library via ctypes when
   available, falls back to default otherwise. Conditional registration based on library
   availability.

3. **`cuda` backend** (`backends/cuda/ops.py`): Uses CUDA kernels via ctypes. Most
   optimized path.

**A new op should always provide at minimum a `default` implementation.** This ensures
the op works on all devices and with `torch.compile`. Device-specific backends are
optimizations.

### 3.3 Shared Implementation Helper Pattern

When both the default op and the `.out` variant share logic, extract to a private helper:

```python
@register_kernel("bitsandbytes::dequantize_4bit", "cuda")
def _(A, absmax, blocksize, quant_type, shape, dtype):
    out = torch.empty(shape, dtype=dtype, device=A.device)
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)
    return out

@register_kernel("bitsandbytes::dequantize_4bit.out", "cuda")
def _(A, absmax, blocksize, quant_type, shape, dtype, out):
    torch._check(out.shape == shape, ...)
    torch._check(out.dtype == dtype, ...)
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)

def _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out):
    # Shared implementation
    ...
```

### 3.4 Conditional Registration

CPU backend ops are conditionally registered based on library availability:

```python
if not isinstance(lib, ErrorHandlerMockBNBNativeLibrary):
    @register_kernel("bitsandbytes::quantize_blockwise", "cpu")
    def _(A, code, blocksize):
        ...
```

Use this pattern for any backend that may not be available at runtime.

### 3.5 ROCm/HIP Considerations

ROCm uses a warp size of 64 (vs NVIDIA's 32). This affects blocksize constraints:

```python
if ROCM_WARP_SIZE_64:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
else:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64, 32])
```

Blocksize 32 is not supported on ROCm because the blocksize must be >= warp size.

---

## 4. The Functional Layer Pattern (`functional.py`)

### 4.1 Role

`functional.py` is the stateless Python API. It wraps `torch.ops.bitsandbytes.*` calls
with user-friendly signatures, handles QuantState management, and provides convenience
wrappers.

### 4.2 Function Signature Convention

Public functions in `functional.py` follow this pattern:

```python
def quantize_blockwise(
    A: torch.Tensor,
    code: Optional[torch.Tensor] = None,     # optional codebook
    absmax: Optional[torch.Tensor] = None,    # optional pre-allocated output
    out: Optional[torch.Tensor] = None,       # optional pre-allocated output
    blocksize=4096,                           # configuration
    nested=False,                             # configuration
) -> tuple[torch.Tensor, QuantState]:         # always return tuple with QuantState
```

Conventions:
- First argument is always the input tensor `A`
- Optional output tensors (`out`, `absmax`) come after required args
- Configuration parameters (`blocksize`, `quant_type`) come last
- Return type includes `QuantState` when quantization state is produced
- `blocksize` defaults are ROCm-aware: `64 if not ROCM_WARP_SIZE_64 else 128`

### 4.3 Dispatching to Ops

Functional layer functions dispatch to the `torch.ops.bitsandbytes` namespace:

```python
# GOOD: use torch.ops for dispatch
_out, _absmax = torch.ops.bitsandbytes.quantize_4bit.default(A, blocksize, quant_type, quant_storage)

# Use .out variant when pre-allocated output is available
torch.ops.bitsandbytes.dequantize_4bit.out(A, absmax, blocksize, quant_type, shape, dtype, out=out)
```

Do **not** call backend functions directly from `functional.py`. Always go through
`torch.ops.bitsandbytes.*` so dispatch works correctly.

### 4.4 QuantState Management

Quantization functions create and return `QuantState` objects that bundle all metadata
needed for dequantization:

```python
state = QuantState(
    absmax=_absmax,
    shape=input_shape,
    dtype=A.dtype,
    blocksize=blocksize,
    code=code,
    quant_type=quant_type,
    offset=offset,       # only for nested quantization
    state2=state2,       # only for nested quantization
)
```

The `QuantState` must contain everything needed to dequantize without any other context.
This is critical for serialization.

### 4.5 Codebook / Quantization Map Management

Quantization maps (codebooks) are cached in the module-level `name2qmap` dict:

```python
if "dynamic" not in name2qmap:
    name2qmap["dynamic"] = create_dynamic_map().to(A.device)
code = name2qmap["dynamic"]
```

When creating a QuantState, always copy the code tensor to avoid cross-device issues:

```python
quant_state = QuantState(
    absmax=_absmax,
    code=code.to(A.device, copy=True),  # copy=True is important
    ...
)
```

---

## 5. Neural Network Module Conventions (`nn/`)

### 5.1 Module Class Structure

Quantized modules follow this pattern:

1. Inherit from the corresponding `torch.nn` class (`nn.Linear`, `nn.Embedding`)
2. Replace `self.weight` with a custom Parameter class (`Params4bit` or `Int8Params`)
3. Override `forward()` to handle quantization
4. Override `_save_to_state_dict()` for serialization of quantization state
5. Register a `_register_load_state_dict_pre_hook` for deserialization

```python
class Linear4bit(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, ...):
        super().__init__(input_features, output_features, bias, device)
        self.weight = Params4bit(
            self.weight.data,
            requires_grad=False,  # quantized weights are frozen
            ...
            module=self,  # back-reference for quant_state sync
        )
```

### 5.2 Custom Parameter Classes

`Params4bit` and `Int8Params` are subclasses of `torch.nn.Parameter` that handle
quantization-on-device-transfer:

```python
class Params4bit(torch.nn.Parameter):
    def __new__(cls, data=None, requires_grad=False, ...):
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        # Store quantization config on the parameter
        self.blocksize = blocksize
        self.quant_type = quant_type
        ...
        return self

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type != "meta" and not self.bnb_quantized:
            return self._quantize(device)  # quantize on first device transfer
        ...
```

Key rules:
- Quantization happens lazily, on first `.to(device)` call
- The `module` back-reference keeps `module.quant_state` in sync
- `__getstate__`/`__setstate__`/`__deepcopy__` must be implemented for pickling
- `__torch_function__` must handle `torch.chunk` and `torch.split` to preserve metadata

### 5.3 Forward Method Pattern

The forward method in quantized modules should:
1. Fix up quant_state if needed (FSDP recovery)
2. Cast bias to match input dtype
3. Dispatch to the appropriate matmul function
4. Return output in the input's original dtype

```python
def forward(self, x: torch.Tensor):
    fix_4bit_weight_quant_state_from_module(self)  # FSDP recovery
    quant_state = self.weight.quant_state

    # Cast bias if needed
    if self.bias is not None and self.bias.dtype != x.dtype:
        self.bias.data = self.bias.data.to(x.dtype)

    # Dispatch
    inp_dtype = x.dtype
    if self.compute_dtype is not None:
        x = x.to(self.compute_dtype)
    bias = None if self.bias is None else self.bias.to(self.compute_dtype)

    return bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=quant_state).to(inp_dtype)
```

---

## 6. Optimizer Conventions (`optim/`)

### 6.1 Class Hierarchy

```
torch.optim.Optimizer
  └── Optimizer8bit          # Base class with 8-bit state management
        ├── Optimizer1State   # For optimizers with 1 state tensor (SGD, Lion)
        └── Optimizer2State   # For optimizers with 2 state tensors (Adam, AdamW)
```

Concrete optimizer classes are thin wrappers:

```python
class Adam(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), ...):
        super().__init__("adam", params, lr, betas, eps, weight_decay, optim_bits, ...)
```

### 6.2 Adding a New Optimizer

To add a new optimizer:

1. Add the optimizer name to `str2optimizer32bit` and `str2optimizer8bit_blockwise`
   dicts in `backends/cuda/ops.py`
2. Add corresponding C function entries in `str2optimizer8bit` in `functional.py`
3. Create the Python class in a new file under `optim/`
4. Inherit from `Optimizer1State` or `Optimizer2State`
5. Add to `optim/__init__.py` exports
6. Add the optimizer to the `default` backend implementation in `backends/default/ops.py`

### 6.3 Optimizer Name String Convention

Optimizer names are lowercase strings matching the dict keys:
`"adam"`, `"momentum"`, `"rmsprop"`, `"lion"`, `"adagrad"`, `"lamb"`, `"lars"`, `"ademamix"`

These strings are passed through the op dispatch system to select the correct C function.

---

## 7. Input Validation Rules

### 7.1 Use `torch._check`, Not `assert`

In op implementations (both fake/meta and kernel implementations), **always** use
`torch._check` for input validation, never `assert`:

```python
# GOOD: works with torch.compile, provides clear error messages
torch._check(A.dtype == torch.int8, lambda: f"A must be int8, got {A.dtype}")
torch._check_is_size(blocksize)
torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64],
             lambda: f"Invalid blocksize: {blocksize}")

# BAD: stripped in optimized mode, breaks torch.compile
assert A.dtype == torch.int8, f"A must be int8, got {A.dtype}"
```

The error message should be a **lambda** (lazy evaluation) to avoid string formatting
overhead in the hot path.

### 7.2 When to Validate

- **In `@register_fake` functions**: Validate all inputs. These run during tracing and
  are the contract definition.
- **In `@register_kernel` functions**: Validate critical constraints. Some checks can be
  skipped for performance (see commented-out checks in `_gemv_4bit_impl` for an example).
- **In `functional.py`**: Use `assert` sparingly for internal invariants. Use `ValueError`
  or `RuntimeError` for user-facing errors.

### 7.3 Standard Validation Patterns

```python
# Validate tensor dtype
torch._check(A.dtype == torch.int8, lambda: f"A must be int8, got {A.dtype}")

# Validate dtype is one of several options
torch._check(
    A.dtype in [torch.float16, torch.bfloat16, torch.float32],
    lambda: f"A must be float16, bfloat16, or float32, got {A.dtype}",
)

# Validate blocksize
torch._check_is_size(blocksize)  # ensures positive integer

# Validate shape match
torch._check(out.shape == expected_shape,
             lambda: f"Expected out.shape == {expected_shape}, got {out.shape}")

# Validate device match
torch._check(out.device == A.device,
             lambda: f"Expected out.device == {A.device}, got {out.device}")

# Validate string enum
torch._check(quant_type in ["fp4", "nf4"],
             lambda: f"quant_type must be fp4 or nf4, got {quant_type}")
```

---

## 8. Error Handling

### 8.1 Error Types

- `RuntimeError` — for runtime failures (CUDA errors, library not loaded, invalid state)
- `ValueError` — for invalid argument values
- `NotImplementedError` — for unimplemented features/paths
- `ImportError` — for missing optional dependencies (e.g., scipy)

### 8.2 Deferred Error Pattern

The native library uses a deferred error pattern to avoid breaking import:

```python
class ErrorHandlerMockBNBNativeLibrary(BNBNativeLibrary):
    """Throws when a method is CALLED, not when it's ACCESSED."""
    def __getattr__(self, name):
        def throw_on_call(*args, **kwargs):
            raise RuntimeError(f"{self.formatted_error}...")
        return throw_on_call
```

This allows `import bitsandbytes` to succeed even without CUDA, deferring the error to
when GPU functionality is actually used.

### 8.3 Warning Conventions

Use `warnings.warn()` for non-fatal issues. The codebase uses this for:
- Performance warnings (wrong dtype for inference speed)
- Deprecation warnings
- Configuration suggestions

```python
warnings.warn(
    "Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 "
    "(default). This will lead to slow inference.",
)
```

After issuing a one-time warning, filter subsequent occurrences:
```python
warnings.filterwarnings("ignore", message=".*inference.")
```

---

## 9. Tensor Immutability and Side Effects

### 9.1 Never Mutate User-Provided Tensors

This is one of the most critical rules. Functions must **never** modify tensors passed
in by the caller unless the API contract explicitly documents in-place behavior.

```python
# BAD: mutates the user's tensor (caused bug #1587)
A[outliers] = 0  # A was passed in by the caller!

# GOOD: clone or mask without modifying the original
A_clean = A.masked_fill(outlier_mask, 0.0)

# GOOD: if mutation is required, restore afterward
outlier_backup = A[outliers].clone()
A[outliers] = 0
# ... use A ...
A[outliers] = outlier_backup  # restore
```

The `default` backend's `int8_vectorwise_quant` shows the correct pattern:
```python
# Backup outliers, zero them, quantize, then restore
outlier_restore = A[outliers].clone()
A[outliers] = 0
# ... quantize ...
A[outliers] = outlier_restore
```

### 9.2 In-Place Op Convention

Ops that mutate tensors in-place use PyTorch's `Tensor!` annotation in the schema and
return `None`:

```python
# Schema for in-place op
"(Tensor(a0!) g, Tensor(a1!) p, ...) -> ()"

# Python implementation modifies g and p in-place, returns None
```

### 9.3 Output Tensor Handling

When an `out` parameter is provided:
```python
# Copy result to pre-allocated output
out = out.copy_(_result) if out is not None else _result
```

---

## 10. ctypes / Native Library Calling Convention

### 10.1 Getting Pointers

Always use the `get_ptr()` utility to get ctypes pointers from tensors:

```python
from bitsandbytes.functional import get_ptr

ptrA = get_ptr(A)       # ct.c_void_p or None if A is None
ptrOut = get_ptr(out)
```

### 10.2 Type Casting for C Functions

Match the C function's parameter types exactly:

```python
lib.cquantize_blockwise_fp16(
    get_ptr(code),                  # void* (pointer to tensor data)
    get_ptr(A),                     # void*
    get_ptr(absmax),                # void*
    get_ptr(out),                   # void*
    ct.c_int32(blocksize),          # int32_t
    ct.c_int(A.numel()),            # int
)
```

Type mapping:
- `ct.c_void_p` — pointers
- `ct.c_int32` — int32_t (use for blocksize, dimensions)
- `ct.c_int` — int (use for element counts)
- `ct.c_int64` / `ct.c_longlong` — int64_t (CPU backend uses longlong)
- `ct.c_float` — float (use for hyperparameters: lr, beta, eps, etc.)
- `ct.c_bool` — bool
- `ct.c_size_t` — size_t (use for byte counts)

### 10.3 Dtype Dispatch Pattern

C functions are named with dtype suffixes. The Python code dispatches:

```python
if A.dtype == torch.float16:
    lib.cquantize_blockwise_fp16(*args)
elif A.dtype == torch.bfloat16:
    lib.cquantize_blockwise_bf16(*args)
elif A.dtype == torch.float32:
    lib.cquantize_blockwise_fp32(*args)
else:
    raise ValueError(f"Unsupported dtype: {A.dtype}")
```

For 4-bit ops, the naming includes both dtype and quant_type:
```python
lib.cquantize_blockwise_bf16_nf4(...)
lib.cdequantize_blockwise_fp16_fp4(...)
```

### 10.4 Optimizer Function Dispatch

Optimizer functions use a dict-based dispatch:

```python
str2optimizer32bit = {
    "adam": (lib.cadam32bit_grad_fp32, lib.cadam32bit_grad_fp16, lib.cadam32bit_grad_bf16),
    "lion": (lib.clion32bit_grad_fp32, lib.clion32bit_grad_fp16, lib.clion32bit_grad_bf16),
    ...
}

# Select by dtype index: [0]=fp32, [1]=fp16, [2]=bf16
if g.dtype == torch.float32:
    optim_func = optim_fns[0]
elif g.dtype == torch.float16:
    optim_func = optim_fns[1]
elif g.dtype == torch.bfloat16 and len(optim_fns) == 3:
    optim_func = optim_fns[2]
```

When adding a new optimizer, add entries to **all** relevant dicts in both
`functional.py` (8-bit variants) and `backends/cuda/ops.py` (32-bit and 8-bit blockwise).

---

## 11. CUDA Device Management

### 11.1 Device Context Manager

All CUDA kernel calls must be wrapped in a device context:

```python
with _cuda_device_of(A):
    lib.some_cuda_function(...)
```

The `_cuda_device_of` function is optimized: on single-GPU systems it returns a no-op
context manager, avoiding the overhead of `cudaGetDevice`/`cudaSetDevice`.

### 11.2 Stream Handling

Get the current CUDA stream for async operations:

```python
stream = _get_tensor_stream(A)
# Pass as last argument to C functions that accept streams
lib.cdequantize_blockwise_fp16(*args, stream)
```

The `_get_tensor_stream` function handles both CUDA and XPU streams.

### 11.3 Multi-Device Safety

When a function takes multiple tensors, they should all be on the same device. The
`is_on_gpu()` function validates this:

```python
is_on_gpu([A, out, absmax])  # raises RuntimeError if on different devices
```

---

## 12. CUDA/C++ Kernel Conventions (`csrc/`)

### 12.1 File Organization

```
csrc/
├── ops.cu              # CUDA op implementations (dispatching, cuBLAS calls)
├── kernels.cu          # CUDA kernel definitions (__global__ functions)
├── ops.cuh             # CUDA op declarations
├── common.cuh          # Compute capability macros, warp size, constants
├── include/ops.cuh     # Public header
├── pythonInterface.cpp  # C-to-Python interface (ctypes entry points)
├── cpu_ops.cpp         # CPU-only native implementations
├── ops.hip / kernels.hip  # ROCm/HIP variants
```

### 12.2 Compute Capability Macros

Use the macros from `common.cuh`:

```cpp
#define BNB_CC_VOLTA 700
#define BNB_CC_AMPERE 800
#define BNB_CC_ADA 890
#define BNB_CC_HOPPER 900
#define BNB_CC_BLACKWELL 1000

// Feature availability
#define BNB_FP16_MMA_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_VOLTA)
#define BNB_INT8_MMA_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_VOLTA_XAVIER)
#define BNB_BF16_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_AMPERE)
```

### 12.3 Error Checking

Use the project's error checking macros:

```cpp
CUDA_CHECK_RETURN(cudaMemcpy(...));
CHECK_CUSPARSE(cusparseCreate(...));
```

The `checkCublasStatus` function returns an error code rather than throwing — the Python
side interprets it:

```python
has_error = lib.cigemmlt_32(ctx, m, n, k, ...)
if has_error == 100:  # ERR_NOT_IMPLEMENTED
    raise NotImplementedError(...)
```

### 12.4 Kernel Launch Conventions

- Warp size is always 32 on NVIDIA (`BNB_WARP_SIZE`)
- The `common.cuh` header defines per-architecture thread/block limits
- Blocksize for quantization ops is always a power of 2, minimum 32 (64 on ROCm)

### 12.5 C-to-Python Interface

Every C function exposed to Python is declared in `pythonInterface.cpp` with `extern "C"`:

```cpp
extern "C" {
    void cquantize_blockwise_fp16(float* code, half* A, float* absmax,
                                   unsigned char* out, int blocksize, int n);
}
```

The naming convention is `c<function_name>_<dtype>` (prefix `c` for "C interface").

### 12.6 clang-format

All C/C++/CUDA files under `csrc/` are formatted by `clang-format`. The configuration
is in `.clang-format` at the repo root. Run `pre-commit run --all-files` to auto-format.

---

## 13. Test Conventions

### 13.1 Test File Organization

Tests are organized by module:
- `test_ops.py` — Tests for `torch.ops.bitsandbytes.*` operations
- `test_functional.py` — Tests for `bitsandbytes.functional` API
- `test_linear4bit.py` — Tests for `nn.Linear4bit` and related modules
- `test_linear8bitlt.py` — Tests for `nn.Linear8bitLt`
- `test_modules.py` — Integration tests for modules
- `test_optim.py` — Optimizer tests
- `test_autograd.py` — Autograd function tests

### 13.2 Parametrization Pattern

Use multi-axis parametrization for thorough coverage:

```python
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("blocksize", [4096, 2048, 1024, 512, 256, 128, 64])
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("nested", TRUE_FALSE, ids=id_formatter("nested"))
def test_quantize_blockwise(device, dtype, blocksize, quant_type, nested):
    ...
```

Conventions:
- Always parametrize by `device` using `get_available_devices()`
- Use `get_available_devices(no_cpu=True)` for GPU-only tests
- Use `TRUE_FALSE` from `tests.helpers` for boolean parameters
- Use `id_formatter("label")` for readable test IDs
- Use `describe_dtype` for dtype test IDs

### 13.3 Device Compatibility

Tests must handle device-specific limitations:

```python
# Skip configurations unsupported on specific hardware
if device == "hpu" and not is_supported_on_hpu(quant_type, dtype, quant_storage):
    pytest.skip("This configuration is not supported on HPU.")

# ROCm blocksize restrictions
blocksizes = [4096, 2048, 1024, 512, 256, 128, 64] if not ROCM_WARP_SIZE_64 else [4096, 2048, 1024, 512, 256, 128]
```

### 13.4 Test Assertions

**Assert specific values, not just "no crash":**

```python
# GOOD: verifies actual correctness
assert out.shape == (10, 30)
assert out.dtype == torch.int32
assert out.device == A.device

# GOOD: numerical accuracy check
torch.testing.assert_close(dequantized, original, rtol=0.1, atol=0.01)

# GOOD: custom tolerance with count
def assert_all_approx_close(a, b, rtol=1e-3, atol=1e-3, count=0):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    sumval = (idx == 0).sum().item()
    if sumval > count:
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

# BAD: only checks it doesn't crash
result = my_function(input)
assert result is not None  # This proves nothing about correctness
```

### 13.5 opcheck for Custom Ops

Use `torch.library.opcheck` to validate op correctness with torch.compile:

```python
opcheck(torch.ops.bitsandbytes.int8_linear_matmul.default, (A, B))
```

This verifies:
- The fake implementation produces correct shapes/dtypes
- The op works with autograd
- The op works with torch.compile tracing

### 13.6 Test Helper Functions

Use helpers from `tests/helpers.py`:

- `get_available_devices()` — returns list of available device strings
- `get_test_dims(min, max, n=N)` — random dimensions for fuzz testing
- `torch_save_to_buffer(obj)` / `torch_load_from_buffer(buf)` — in-memory serialization
- `id_formatter("label")` — creates readable pytest parameter IDs
- `describe_dtype(dtype)` — short dtype name for test IDs

### 13.7 Seed Management

The conftest automatically sets seeds before each test:

```python
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
```

Do not set seeds inside individual tests unless testing randomness-sensitive behavior.

### 13.8 Memory Management

The conftest runs `gc.collect()` every 50 tests and `torch.cuda.empty_cache()` after
each test. If your test allocates large tensors, consider explicit cleanup:

```python
del large_tensor
torch.cuda.empty_cache()
```

### 13.9 Test Markers

```python
@pytest.mark.slow          # excluded from default run
@pytest.mark.benchmark     # excluded from default run
@pytest.mark.deprecated    # excluded from default run
```

Default pytest config: `-m 'not slow and not benchmark and not deprecated'`

---

## 14. Deprecation Protocol

### 14.1 How to Deprecate

Use the `@deprecated` decorator from `typing_extensions`:

```python
from typing_extensions import deprecated

@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def quantize(A, code=None, out=None):
    ...
```

### 14.2 Deprecation Timeline

- Add `@deprecated` decorator with `category=FutureWarning`
- Keep the deprecated function working for at least one minor version
- Move tests for deprecated functions to `test_deprecated.py` and mark with
  `@pytest.mark.deprecated`
- Remove the function in the next minor or major version
- When removing, also remove any compatibility shims that existed only to support the
  deprecated path

### 14.3 Parameter Deprecation

When deprecating a parameter (not removing it yet):

```python
def some_function(A, old_param=None, new_param=None):
    if old_param is not None:
        warnings.warn(
            "old_param is deprecated, use new_param instead",
            FutureWarning,
            stacklevel=2,
        )
        if new_param is None:
            new_param = old_param
```

---

## 15. API Design Rules

### 15.1 Public API Surface

Public API consists of:
- Functions in `bitsandbytes.functional` — `quantize_4bit`, `dequantize_4bit`, etc.
- Classes in `bitsandbytes.nn` — `Linear4bit`, `Linear8bitLt`, `Params4bit`, etc.
- Classes in `bitsandbytes.optim` — `Adam`, `Adam8bit`, etc.
- Top-level re-exports in `bitsandbytes.__init__` — `matmul`, `matmul_4bit`, `MatmulLtState`

The `torch.ops.bitsandbytes.*` namespace is also public (for advanced users and
torch.compile integration) but changes to it affect the fake implementations.

### 15.2 New Public Functions

When adding a new public function:
1. Add the op schema to `_ops.py`
2. Add fake implementation with full validation
3. Add at least a `default` backend implementation
4. Add the Python-facing wrapper to `functional.py`
5. Add comprehensive tests covering all parametrization axes
6. Export from the appropriate `__init__.py`

### 15.3 Breaking Changes

Any change that modifies the behavior of existing public API is a breaking change.
Breaking changes require:
- A deprecation period (see Section 14)
- Mention in the changelog
- Consideration of downstream impact (transformers, PEFT, accelerate)

---

## 16. Dependency Policy

### 16.1 Core Dependencies

The only runtime dependencies are (from `pyproject.toml`):
- `torch>=2.3,<3`
- `numpy>=1.17`
- `packaging>=20.9`

### 16.2 Optional Dependencies

- `scipy` — only for `create_normal_map()` which is rarely called at runtime (the NF4
  codebook values are hardcoded)
- Test dependencies: `einops`, `lion-pytorch`, `pytest`, `scipy`, `transformers`

### 16.3 Adding New Dependencies

**Do not add new runtime dependencies without explicit maintainer approval.** This is a
widely-used library and every dependency adds installation burden, version conflict risk,
and supply chain surface.

For optional functionality:
```python
try:
    from scipy.stats import norm
except ImportError as ie:
    raise ImportError(
        "Scipy is required for `create_normal_map`. Install `bitsandbytes` with the `[test]` extra.",
    ) from ie
```

---

## 17. Common Anti-Patterns to Reject

### 17.1 Mutating User Tensors

```python
# REJECT: modifies caller's tensor
A[:, outlier_cols] = 0  # where A came from the caller
```

See Section 9 for the correct pattern.

### 17.2 Using `assert` in Op Implementations

```python
# REJECT: stripped in optimized mode, breaks torch.compile
assert A.dtype == torch.int8

# USE INSTEAD:
torch._check(A.dtype == torch.int8, lambda: "A must be int8")
```

### 17.3 Direct Backend Calls from functional.py

```python
# REJECT: bypasses dispatch, breaks torch.compile
from bitsandbytes.backends.cuda.ops import _dequantize_4bit_impl
result = _dequantize_4bit_impl(A, ...)

# USE INSTEAD:
result = torch.ops.bitsandbytes.dequantize_4bit.default(A, ...)
```

### 17.4 Adding pip Dependencies Without Discussion

```python
# REJECT in a PR without explicit approval:
import some_external_package  # adds new runtime dependency
```

### 17.5 Hardcoded CUDA Assumptions

```python
# REJECT: assumes CUDA, breaks CPU/XPU/MPS
torch.cuda.synchronize()

# USE INSTEAD: check device type
if A.device.type == "cuda":
    torch.cuda.synchronize()

# Or use the sync utility:
from bitsandbytes.utils import sync_gpu
sync_gpu(tensor)
```

### 17.6 Ignoring ROCm/HIP Differences

```python
# REJECT: doesn't account for warp size 64
torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64, 32])

# USE INSTEAD:
if ROCM_WARP_SIZE_64:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
else:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64, 32])
```

### 17.7 Tests That Only Check "No Crash"

```python
# REJECT: proves nothing
def test_quantize():
    result = bnb.functional.quantize_4bit(torch.randn(100))
    assert result is not None

# REQUIRE: verify shapes, dtypes, numerical accuracy
def test_quantize():
    A = torch.randn(256, dtype=torch.float16, device="cuda")
    out, state = bnb.functional.quantize_4bit(A, blocksize=64, quant_type="nf4")
    assert out.dtype == torch.uint8
    assert state.blocksize == 64
    assert state.quant_type == "nf4"
    assert state.shape == A.shape

    # Round-trip accuracy
    A_deq = bnb.functional.dequantize_4bit(out, state)
    torch.testing.assert_close(A_deq, A, rtol=0.1, atol=0.02)
```

### 17.8 Unscoped Imports in Backend Code

```python
# REJECT in backends/cuda/ops.py:
import bitsandbytes  # circular import risk

# USE INSTEAD:
from bitsandbytes.functional import get_ptr, _cuda_device_of
```

### 17.9 Missing `.out` Variant

If you add a new op that allocates an output tensor, also provide an `.out` variant.
This allows callers to pre-allocate and reuse memory, which is important for performance
in training loops.

### 17.10 Forgetting `_cuda_device_of` Wrapper

```python
# REJECT: may call kernel on wrong GPU in multi-GPU setup
lib.csome_kernel(get_ptr(A), ...)

# REQUIRE: always wrap in device context
with _cuda_device_of(A):
    lib.csome_kernel(get_ptr(A), ...)
```

---

## 18. Performance Expectations

### 18.1 Kernel Performance

- **4-bit GEMV** (`gemv_4bit`): The CUDA path should be within 2x of cuBLAS fp16 GEMV
  for typical shapes (batch=1, hidden_dim >= 1024)
- **8-bit matmul** (`int8_linear_matmul`): Uses cuBLASLt int8 GEMM. Falls back to fp32
  when inner dim is not divisible by 4.
- **Blockwise quantize/dequantize**: These are memory-bandwidth-bound operations

### 18.2 Python Overhead

- Avoid Python loops over tensor elements
- Use `torch.ops.bitsandbytes.*` dispatch rather than manual if/else chains when possible
- The `_cuda_device_of` optimization (no-op on single GPU) is important — do not remove it

### 18.3 Memory

- 4-bit quantization: ~4x memory reduction vs fp16
- 8-bit optimizers: ~4x memory reduction for optimizer state vs fp32
- Nested quantization (compress_statistics=True): additional ~0.5 bits per parameter for absmax

---

## 19. Documentation Standards

### 19.1 Docstring Style

Public functions in `functional.py` use a hybrid format with Google/numpy style:

```python
def quantize_4bit(
    A: torch.Tensor,
    ...
) -> tuple[torch.Tensor, QuantState]:
    """Quantize tensor A in blocks of 4-bit values.

    Quantizes tensor A by dividing it into blocks which are independently quantized.

    Args:
        A (`torch.Tensor`): The input tensor. Supports `float16`, `bfloat16`, or `float32` datatypes.
        blocksize (`int`, *optional*):
            The size of the blocks. Defaults to 128 on ROCm and 64 otherwise.
            Valid values are 32, 64, 128, 256, 512, 1024, 2048, and 4096.

    Raises:
        ValueError: Raised when the input data type is not supported.

    Returns:
        Tuple[`torch.Tensor`, `QuantState`]: A tuple containing the quantization results.
    """
```

Conventions:
- Type annotations use backtick format in docstrings: `` `torch.Tensor` ``
- Optional parameters are marked: `*optional*`
- Default values are documented in the description
- Link to papers when relevant: `[QLoRA](https://arxiv.org/abs/2305.14314)`

### 19.2 Code Comments

- Comments explain **why**, not **what**
- Mathematical operations should reference the algorithm or paper
- TODO comments use the format: `# TODO(username): description` or `# TODO: description`
- Deprecated/removable code is marked: `# TODO: Deprecate/remove`

### 19.3 Module-Level Documentation

Module classes (`Linear4bit`, `Linear8bitLt`) should have class docstrings with:
1. Brief description
2. Link to the relevant paper
3. Usage example

```python
class Linear4bit(nn.Linear):
    """
    This class is the base module for the 4-bit quantization algorithm presented in
    [QLoRA](https://arxiv.org/abs/2305.14314).

    Example:

    ```python
    import bitsandbytes as bnb
    linear_q = bnb.nn.Linear4bit(64, 64)
    linear_q = linear_q.to("cuda")  # Quantization happens here
    ```
    """
```

---

## 20. Serialization and State Dict Conventions

### 20.1 Module State Dict

4-bit modules serialize quantization state alongside weights:

```python
def _save_to_state_dict(self, destination, prefix, keep_vars):
    super()._save_to_state_dict(destination, prefix, keep_vars)
    if getattr(self.weight, "quant_state", None) is not None:
        for k, v in self.weight.quant_state.as_dict(packed=True).items():
            destination[prefix + "weight." + k] = v if keep_vars else v.detach()
```

The packed format uses `pack_dict_to_tensor()` to store non-tensor metadata (blocksize,
quant_type, dtype string) as a JSON-encoded uint8 tensor. This is required for
safetensors compatibility.

### 20.2 Optimizer State Dict

The `Optimizer8bit` class wraps quantization state tensors in a nested dict to hide them
from FSDP's gather operations:

```python
# Keys that get wrapped: qmap1, qmap2, max1, max2, state1, state2, ...
param_state[self._FSDP_WRAPPED_QUANT_STATE_KEY] = quant_state_dict
```

This is unwrapped on `load_state_dict`.

### 20.3 Backward Compatibility

- The `QuantState.__getitem__` method provides backward compatibility with the old
  list-based quant state format
- The `maybe_rearrange_weight` hook handles legacy weight formats (col32, col_turing,
  col_ampere → now only "row" is supported)
- Weight format mapping is maintained in `utils.py`:
  `LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING = {"row": 0, "col32": 1, "col_turing": 2, "col_ampere": 3}`

---

## Summary: PR Review Checklist

When reviewing a PR, check these standards in order of priority:

1. **Tensor immutability**: Does any code mutate user-provided tensors? (Section 9)
2. **Input validation**: Are `torch._check` (not `assert`) used in ops? (Section 7)
3. **Backend dispatch**: Does new code go through `torch.ops.bitsandbytes.*`? (Section 4.3)
4. **Device context**: Are CUDA calls wrapped in `_cuda_device_of`? (Section 11)
5. **ROCm compatibility**: Are blocksize constraints ROCm-aware? (Section 3.5)
6. **Test quality**: Do tests verify actual values, not just "no crash"? (Section 13.4)
7. **Op pattern**: Does a new op have schema + fake + default backend? (Section 2)
8. **Dependencies**: Are any new runtime dependencies added? (Section 16)
9. **Breaking changes**: Does it change public API without deprecation? (Section 15)
10. **Memory safety**: In CUDA code, are bounds checked? (Section 12)
