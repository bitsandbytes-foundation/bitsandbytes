# Downstream Integrations Guide

This document catalogs every major downstream consumer of bitsandbytes, the specific APIs each
consumer calls, the assumptions each makes, and the breaking-change risks a PR reviewer must
evaluate. It is written for agent-reviewers who need to assess whether a bitsandbytes change is
safe to merge without reading each downstream codebase from scratch.

---

## Table of Contents

1. [HuggingFace Transformers](#1-huggingface-transformers)
2. [PEFT (Parameter-Efficient Fine-Tuning)](#2-peft)
3. [Accelerate](#3-accelerate)
4. [Text Generation Inference (TGI)](#4-text-generation-inference-tgi)
5. [vLLM](#5-vllm)
6. [Consolidated API Surface](#6-consolidated-api-surface)
7. [General Breaking-Change Checklist](#7-general-breaking-change-checklist)

---

## 1. HuggingFace Transformers

**Repository**: https://github.com/huggingface/transformers
**Integration depth**: Deep — transformers is the primary user-facing entry point for bnb quantization.
**Minimum bnb version enforced**: `0.46.1` (constant `BITSANDBYTES_MIN_VERSION` in `utils/import_utils.py:97`)

### 1.1 Architecture of the Integration

Transformers implements bnb support through a layered quantizer architecture:

```
User code
  │
  ├── BitsAndBytesConfig (quantization_config.py)
  │     Maps user-facing params → bnb constructor args
  │
  ├── Bnb4BitHfQuantizer / Bnb8BitHfQuantizer (quantizers/)
  │     Orchestrates model surgery: replace nn.Linear → bnb.nn.Linear4bit / Linear8bitLt
  │
  ├── integrations/bitsandbytes.py
  │     Core logic: replace_with_bnb_linear(), dequantize_and_replace(),
  │     dequantize_bnb_weight(), Bnb4bitQuantize, Bnb4bitDeserialize,
  │     Bnb8bitQuantize, Bnb8bitDeserialize, validate_bnb_backend_availability()
  │
  └── modeling_utils.py / trainer.py / trainer_optimizer.py
        Use bnb types for param counting, device movement, optimizer setup
```

### 1.2 BitsAndBytesConfig — Parameter Mapping

The `BitsAndBytesConfig` dataclass in `utils/quantization_config.py` is the user-facing
entry point. It maps to bnb constructor parameters as follows:

| BitsAndBytesConfig field | bnb constructor arg | Used by |
|---|---|---|
| `load_in_4bit` | (selects `bnb.nn.Linear4bit`) | `replace_with_bnb_linear()` |
| `load_in_8bit` | (selects `bnb.nn.Linear8bitLt`) | `replace_with_bnb_linear()` |
| `llm_int8_threshold` | `threshold` kwarg to `Linear8bitLt()` | 8-bit quantizer |
| `llm_int8_has_fp16_weight` | `has_fp16_weights` kwarg to `Linear8bitLt()` | 8-bit quantizer |
| `llm_int8_skip_modules` | modules excluded from conversion | Both quantizers |
| `llm_int8_enable_fp32_cpu_offload` | controls device_map filtering | Both quantizers |
| `bnb_4bit_compute_dtype` | positional arg to `Linear4bit()` | 4-bit quantizer |
| `bnb_4bit_use_double_quant` | `compress_statistics` kwarg to `Linear4bit()` | 4-bit quantizer |
| `bnb_4bit_quant_type` | `quant_type` kwarg to `Linear4bit()` | 4-bit quantizer |
| `bnb_4bit_quant_storage` | `quant_storage` kwarg to `Linear4bit()` | 4-bit quantizer |

**Breaking-change risk**: If any of these `bnb.nn.Linear4bit` or `bnb.nn.Linear8bitLt`
constructor signatures change, transformers will break. The config field names are public API
for thousands of user scripts and HuggingFace model cards.

### 1.3 bnb APIs Called Directly

#### 1.3.1 Module types (isinstance checks and construction)

- **`bnb.nn.Linear4bit`** — Constructed in `replace_with_bnb_linear()`, isinstance-checked in
  `Bnb4BitHfQuantizer.param_needs_quantization()` and `dequantize_and_replace()`.
  Constructor args used: `in_features, out_features, bias, compute_dtype, compress_statistics,
  quant_type, quant_storage`.

- **`bnb.nn.Linear8bitLt`** — Constructed in `replace_with_bnb_linear()`, isinstance-checked in
  `Bnb8BitHfQuantizer.param_needs_quantization()` and `dequantize_and_replace()`.
  Constructor args used: `in_features, out_features, bias, has_fp16_weights, threshold`.

- **`bnb.nn.Params4bit`** — Constructed in `Bnb4bitQuantize.convert()` via
  `bnb.nn.Params4bit(value, requires_grad=False, **old_value.__dict__)`.
  This is the same fragile `__dict__` round-trip pattern used by PEFT (§2.3) and
  Accelerate (§3.2.3). Also accessed via `isinstance(param, bnb.nn.Params4bit)` in
  `modeling_utils.py:987` for parameter counting.

- **`bnb.nn.Params4bit.from_prequantized()`** — Called in `Bnb4bitDeserialize.convert()` with
  args: `data, quantized_stats, requires_grad, device, module`. This is the deserialization path
  for loading pre-quantized 4-bit checkpoints.

- **`bnb.nn.Int8Params`** — Constructed in `Bnb8bitQuantize.convert()` and
  `Bnb8bitDeserialize.convert()`. Constructor: `Int8Params(value, requires_grad=False, **kwargs)`.
  The `SCB` attribute is both popped from kwargs (during quantization) and set (during
  deserialization).

#### 1.3.2 Functional API

- **`bnb.functional.dequantize_4bit(weight.data, weight.quant_state)`** — Called in
  `dequantize_bnb_weight()` for 4-bit dequantization.

- **`bnb.functional.int8_vectorwise_dequant(weight.data, state.SCB)`** — Called in
  `dequantize_bnb_weight()` for 8-bit dequantization (requires bnb v0.45.0+). Falls back to
  manual `weight.data * state.SCB.view(-1, 1) * 7.874015718698502e-3` if not available.

#### 1.3.3 Optimizer API

- **`bitsandbytes.optim.AdamW`** — Used by trainer for 8-bit and paged AdamW variants.
- **`bitsandbytes.optim.Lion`** — Used by trainer for Lion optimizer variants.
- **`bitsandbytes.optim.RMSprop`** — Used by trainer for RMSprop variants.
- **`bitsandbytes.optim.AdEMAMix`** — Used by trainer for AdEMAMix optimizer variants.
- **`bitsandbytes.optim.GlobalOptimManager.get_instance()`** — Called in `trainer.py:1183` to
  register embedding layers for fp32 optimization when using 8-bit optimizers.
- **`manager.register_module_override(module, "weight", {"optim_bits": 32})`** — Sets embedding
  weights to be optimized in fp32 even when the optimizer is 8-bit.

The full list of bnb optimizer names registered in the trainer:
`adamw_bnb`, `adamw_8bit`, `paged_adamw`, `paged_adamw_8bit`, `ademamix`, `ademamix_8bit`,
`paged_ademamix`, `paged_ademamix_8bit`, `lion`, `lion_8bit`, `paged_lion`, `paged_lion_8bit`,
`rmsprop_bnb`, `rmsprop_8bit`, `rmsprop_32bit`.

Optimizer kwargs passed through: `optim_bits` (8 or 32), `is_paged` (bool, except for RMSprop).

#### 1.3.4 Module-level attributes accessed

- **`bnb.supported_torch_devices`** — Accessed via `getattr(bnb, "supported_torch_devices", set())`
  in `validate_bnb_backend_availability()`. Used to check whether the user's available devices
  are supported by the installed bnb version.

- **`module.state`** — Accessed on `Linear8bitLt` instances during dequantization
  (`dequantize_and_replace()` line 298: `state = module.state`).

- **`weight.quant_state`** — Accessed on `Params4bit` instances for dequantization.

- **`weight.SCB`** — Accessed on `Int8Params` instances (the scale/column-wise absmax).

- **`param.element_size()`** and **`param.quant_storage`** — Accessed on `Params4bit` for
  parameter counting in `modeling_utils.py`.

### 1.4 Weight Serialization Format

Transformers defines `WeightConverter` patterns for deserializing pre-quantized bnb checkpoints:

**4-bit checkpoint keys** (per weight tensor):
- `weight` — The packed quantized data
- `weight.absmax` — Absmax scales
- `weight.quant_map` — Quantization code lookup table
- `weight.nested_absmax` — Double-quantization absmax (if `use_double_quant=True`)
- `weight.nested_quant_map` — Double-quantization code lookup
- `weight.quant_state.bitsandbytes__nf4` or `weight.quant_state.bitsandbytes__fp4` — Quant state metadata

These are deserialized via `Params4bit.from_prequantized()`.

**8-bit checkpoint keys** (per weight tensor):
- `weight` — The int8 quantized data
- `SCB` — The scale column-wise absmax
- `weight_format` — Format metadata

These are deserialized via `Int8Params()` with `SCB` set in kwargs.

**Breaking-change risk**: Changing the serialization format for `Params4bit` or `Int8Params`
would break all existing pre-quantized checkpoints on the HuggingFace Hub.

### 1.5 Device Movement and dtype Restrictions

- `modeling_utils.py:3512-3522`: If the model was loaded with bnb, calling `.to(dtype=...)` is
  **blocked** — raises `ValueError("You cannot cast a bitsandbytes model in a new dtype")`.
- Moving 8-bit models across devices requires bnb >= 0.48.0.
- Device map auto-assignment defaults to current CUDA device, NPU, HPU, XPU, or CPU (in that
  priority order).

### 1.6 Conv1D Handling

Transformers includes special handling for OpenAI-style `Conv1D` layers (used by GPT-2):
- Before quantization, the weight matrix is transposed: `value = value.T`
- This is done in both `Bnb4bitQuantize.convert()` and `Bnb8bitQuantize.convert()`
- The `source_cls` attribute is stored on the new bnb module to track this

### 1.7 Test Coverage

Transformers maintains two dedicated test files:
- `tests/quantization/bnb/test_4bit.py` — Tests 4-bit quantization with bloom-1b7
- `tests/quantization/bnb/test_mixed_int8.py` — Tests 8-bit quantization with bloom-1b7

Both test suites require `@slow` (large model downloads) and test:
- Basic quantization and inference
- Serialization / deserialization round-trips
- LoRA-style adapter compatibility
- Multi-GPU scenarios
- Parameter counting with quantized weights

### 1.8 Summary of Breaking-Change Surfaces

| bnb API | Risk if changed | Impact |
|---|---|---|
| `Linear4bit` constructor signature | HIGH | All 4-bit model loading breaks |
| `Linear8bitLt` constructor signature | HIGH | All 8-bit model loading breaks |
| `Params4bit` constructor, `from_prequantized()` | HIGH | Checkpoint deserialization breaks |
| `Int8Params` constructor, `SCB` attribute | HIGH | 8-bit checkpoint deserialization breaks |
| `functional.dequantize_4bit()` signature | HIGH | Dequantization/merging breaks |
| `functional.int8_vectorwise_dequant()` | MEDIUM | Falls back to manual math |
| `Params4bit.quant_state` attribute | HIGH | Dequantization breaks |
| `Linear8bitLt.state` attribute | HIGH | 8-bit dequantization breaks |
| `supported_torch_devices` module attr | LOW | Falls back to empty set via getattr |
| `optim.AdamW/Lion/RMSprop/AdEMAMix` | MEDIUM | Trainer optimizer creation breaks |
| `optim.GlobalOptimManager` | MEDIUM | Embedding fp32 override breaks |
| Serialization key names (`absmax`, `quant_map`, etc.) | CRITICAL | All Hub checkpoints break |

---

## 2. PEFT (Parameter-Efficient Fine-Tuning)

**Repository**: https://github.com/huggingface/peft
**Integration depth**: Very deep — PEFT wraps every bnb linear layer type with adapter-specific subclasses.
**Minimum bnb version**: Checks `is_bnb_available()` (any version) and `is_bnb_4bit_available()` (checks for `bnb.nn.Linear4bit`).

### 2.1 Architecture of the Integration

PEFT has a per-tuner bnb integration pattern. Each tuner method (LoRA, AdaLoRA, IA3, OFT, VeRA,
RandLoRA, ROAD) has a dedicated `bnb.py` file containing specialized wrapper classes:

```
peft/tuners/
  lora/bnb.py      → Linear8bitLt, Linear4bit, dispatch_bnb_8bit, dispatch_bnb_4bit
  adalora/bnb.py   → SVDLinear8bitLt, SVDLinear4bit
  ia3/bnb.py       → Linear8bitLt, Linear4bit
  oft/bnb.py       → Linear8bitLt, Linear4bit, dispatch_bnb_8bit, dispatch_bnb_4bit
  vera/bnb.py      → Linear8bitLt, Linear4bit
  randlora/bnb.py  → Linear8bitLt, Linear4bit
  road/bnb.py      → Linear8bitLt, Linear4bit, dispatch_bnb_8bit, dispatch_bnb_4bit
```

Each tuner's `model.py` uses a dispatcher pattern:
1. `isinstance(target_base_layer, bnb.nn.Linear8bitLt)` → dispatch to 8-bit wrapper
2. `isinstance(target_base_layer, bnb.nn.Linear4bit)` → dispatch to 4-bit wrapper
3. Otherwise → use standard linear wrapper

### 2.2 bnb APIs Called Directly

#### 2.2.1 Module types (isinstance checks)

- **`bnb.nn.Linear8bitLt`** — isinstance-checked in every tuner's dispatch function.
- **`bnb.nn.Linear4bit`** — isinstance-checked in every tuner's dispatch function.
- **`bnb.nn.Int8Params`** — Constructed during merge/unmerge on 8-bit layers. Constructor:
  `bnb.nn.Int8Params(w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights)`.
- **`bnb.nn.Params4bit`** — Constructed during merge/unmerge on 4-bit layers. Constructor:
  `bnb.nn.Params4bit(w_data.to("cpu"), **kwargs)` where kwargs come from `weight.__dict__`.

#### 2.2.2 Functional API

- **`bnb.functional.dequantize_4bit(weight.data, weight.quant_state)`** — Used in:
  - `peft/utils/integrations.py:dequantize_bnb_weight()` (central dequantization utility)
  - `peft/utils/loftq_utils.py` (LoftQ quantization workflow)
  - `tuners/randlora/bnb.py` (direct calls during merge)
  - `tuners/vera/bnb.py` (direct calls during merge)

- **`bnb.functional.int8_vectorwise_dequant(weight.data, state.SCB)`** — Used in
  `dequantize_bnb_weight()` with fallback to manual math for older bnb versions.

#### 2.2.3 Module attributes accessed

Across all tuners, PEFT accesses these bnb-internal attributes:

**On `Linear8bitLt` instances:**
- `target.state` — The `MatmulLtState` object
- `target.state.has_fp16_weights` — Whether weights are stored in fp16
- `target.state.threshold` — The outlier threshold value
- `target.state.SCB` — Scale column-wise absmax (also via `weight.SCB`)
- `target.state.reset_grads()` — Called after merge/unmerge
- `target.index` — The index attribute

**On `Params4bit` instances (via `weight = self.get_base_layer().weight`):**
- `weight.quant_state` — The QuantState for dequantization
- `weight.compress_statistics` — Whether double quantization is used
- `weight.quant_type` — The quantization type (fp4/nf4)
- `weight.__dict__` — The entire attribute dictionary (used to reconstruct after merge)
- `weight.bnb_quantized` — Set to `False` before re-quantization during merge

**On `Linear4bit` instances:**
- `target_base_layer.compute_dtype` — The compute dtype

**On `Params4bit` for parameter counting (`peft_model.py:866`):**
- `param.element_size()` — Element size method
- `param.quant_storage` — The quant storage dtype

### 2.3 Merge/Unmerge Pattern (Critical Path)

The merge/unmerge workflow is the most sensitive integration point. It follows this pattern
consistently across all 7 tuner types:

**4-bit merge:**
```python
weight = self.get_base_layer().weight
kwargs = weight.__dict__
output = dequantize_bnb_weight(weight, state=weight.quant_state)  # → bnb.functional.dequantize_4bit()
w_data = output + lora_delta  # (or matrix multiply for OFT)
if "bnb_quantized" in kwargs:
    kwargs["bnb_quantized"] = False
kwargs["requires_grad"] = False
kwargs.pop("data", None)
kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}  # torch.compile compat
self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), **kwargs).to(weight.device)
```

**8-bit merge:**
```python
weight = self.get_base_layer().weight
state = self.get_base_layer().state
if state.SCB is None:
    state.SCB = weight.SCB
output = dequantize_bnb_weight(weight, state=state)
w_data = output + lora_delta
self.get_base_layer().weight = bnb.nn.Int8Params(
    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
).to(weight.device)
state.reset_grads()
```

**Breaking-change risk**: This pattern depends on:
1. `Params4bit.__dict__` being serializable and re-passable to the constructor
2. `bnb_quantized` being a recognized attribute that can be set to `False`
3. `Int8Params` accepting `has_fp16_weights` as a constructor kwarg
4. `state.reset_grads()` existing and working
5. The dequantize → modify → re-quantize round-trip preserving the weight semantics

### 2.4 4-bit Forward Pass: Defensive Clone

All 4-bit PEFT wrappers include `result = result.clone()` after the base layer forward pass.
This is documented as a workaround for a backprop issue with manipulated views on 4-bit linear
output. The comment attributes this to Tim Dettmers. If the underlying 4-bit forward behavior
changes (e.g., returning a view vs a copy), this defensive clone may become unnecessary or
insufficient.

### 2.5 LoftQ Integration

PEFT includes a LoftQ utility (`utils/loftq_utils.py`) that implements an iterative
quantization-aware initialization. It:
- Creates its own `NFQuantizer` class (reimplements NF4 codebook generation)
- Calls `bnb.functional.dequantize_4bit(qweight.data, qweight.quant_state)` to dequantize
  during iterative refinement
- This is independent of the tuner-level bnb integration

### 2.6 Tuner Coverage Matrix

| Tuner | 8-bit support | 4-bit support | Merge support (8bit) | Merge support (4bit) |
|---|---|---|---|---|
| LoRA | Yes | Yes | Yes | Yes |
| AdaLoRA | Yes | Yes | No | No |
| IA3 | Yes | Yes | No | No |
| OFT | Yes | Yes | Yes | Yes |
| VeRA | Yes | Yes | Yes | Yes |
| RandLoRA | Yes | Yes | Yes | Yes |
| ROAD | Yes | Yes | Yes | Yes |

### 2.7 Summary of Breaking-Change Surfaces

| bnb API | Risk if changed | Impact |
|---|---|---|
| `bnb.nn.Linear4bit` (isinstance check) | HIGH | All 4-bit PEFT adapters fail to dispatch |
| `bnb.nn.Linear8bitLt` (isinstance check) | HIGH | All 8-bit PEFT adapters fail to dispatch |
| `Linear4bit.compute_dtype` attribute | HIGH | 4-bit dispatch fails for all tuners |
| `Params4bit.compress_statistics` attribute | HIGH | 4-bit dispatch fails for all tuners |
| `Params4bit.quant_type` attribute | HIGH | 4-bit dispatch fails for all tuners |
| `Params4bit.quant_state` attribute | HIGH | All 4-bit merge/dequantize operations break |
| `Params4bit.__dict__` round-trip | HIGH | All 4-bit merge operations break |
| `Params4bit.bnb_quantized` attribute | MEDIUM | Merge may fail or re-quantize incorrectly |
| `Int8Params(has_fp16_weights=...)` constructor | HIGH | All 8-bit merge operations break |
| `Linear8bitLt.state` (MatmulLtState) | HIGH | All 8-bit dispatch and merge breaks |
| `MatmulLtState.SCB` | HIGH | 8-bit dequantization breaks |
| `MatmulLtState.has_fp16_weights` | HIGH | 8-bit dispatch breaks |
| `MatmulLtState.threshold` | MEDIUM | 8-bit dispatch passes wrong config |
| `MatmulLtState.reset_grads()` | MEDIUM | 8-bit merge leaves stale state |
| `functional.dequantize_4bit()` signature | HIGH | All 4-bit operations break |
| `functional.int8_vectorwise_dequant()` | MEDIUM | Falls back to manual math |
| `bnb.nn.Linear4bit` forward output semantics | MEDIUM | 4-bit clone() workaround may break |

---

## 3. Accelerate

**Repository**: https://github.com/huggingface/accelerate
**Integration depth**: Medium — accelerate provides model loading, device placement, and offloading for bnb-quantized models.
**Minimum bnb version enforced**: `0.39.0` for 4-bit, `0.37.2` for 8-bit (in `utils/imports.py`).

### 3.1 Architecture of the Integration

Accelerate's bnb integration lives primarily in two files:

```
accelerate/utils/
  bnb.py     → load_and_quantize_model(), replace_with_bnb_layers(), quantize_and_offload_8bit(),
               has_4bit_bnb_layers(), get_keys_to_not_convert()
  modeling.py → set_module_tensor_to_device() (handles bnb param types during weight loading)
```

Plus a `BnbQuantizationConfig` dataclass in `utils/dataclasses.py` that mirrors the same
config fields as transformers' `BitsAndBytesConfig`.

### 3.2 bnb APIs Called Directly

#### 3.2.1 Module construction

- **`bnb.nn.Linear8bitLt(in_features, out_features, bias, has_fp16_weights=False, threshold=...)`**
  — Constructed in `_replace_with_bnb_layers()` to replace `nn.Linear` modules.

- **`bnb.nn.Linear4bit(in_features, out_features, bias, compute_dtype, compress_statistics=..., quant_type=...)`**
  — Constructed in `_replace_with_bnb_layers()` to replace `nn.Linear` modules.

#### 3.2.2 Type checks (by class name, not isinstance)

Accelerate uses **string-based class name checks** rather than isinstance checks in
`set_module_tensor_to_device()`:

```python
param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
param_cls.__name__ == "Int8Params"
module.__class__.__name__ == "Linear8bitLt"
module.__class__.__name__ == "Linear4bit"
```

This is less fragile than isinstance checks (doesn't require importing bnb) but is sensitive
to class **renaming**. If `Int8Params` were renamed to `Int8Parameter`, accelerate would break.

Note: The check also includes `"FP4Params"`, a legacy bnb class that predates `Params4bit`.
Accelerate still guards against it for backward compatibility with older bnb versions.

Also in FSDP utils (`fsdp_utils.py`):
```python
param.__class__.__name__ == "Params4bit"
```

#### 3.2.3 Parameter type construction during weight loading

In `set_module_tensor_to_device()`, accelerate reconstructs bnb parameter types:

```python
kwargs = module._parameters[tensor_name].__dict__
new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(device)
```

This is the same `__dict__` round-trip pattern as PEFT. It depends on:
- `Int8Params.__dict__` and `Params4bit.__dict__` being passable to the constructor
- The constructors accepting the same kwargs they store

Special handling for `Int8Params`:
- Downcasts `float32` → `float16` before constructing `Int8Params`
- For CPU offloading: constructs on GPU (device 0), then moves back to CPU, also moving `.CB`
  and `.SCB` attributes to CPU

#### 3.2.4 Attributes accessed on bnb types

**On `Int8Params`:**
- `.SCB` — Scale column-wise absmax (read during offloading, set during weight loading)
- `.CB` — Accessed during CPU offloading (`new_value.CB.to("cpu")`)
- `.__dict__` — Full attribute dictionary for reconstruction

**On `Params4bit`:**
- `.quant_state` — Checked via `getattr(module.weight, "quant_state", None)` to determine if
  quantization has occurred
- `.__dict__` — Full attribute dictionary for reconstruction

**On `Linear8bitLt`:**
- `.weight.SCB` — Checked to determine if quantization has occurred

**On `Linear4bit`:**
- `.weight.quant_state` — Checked to determine if quantization has occurred

#### 3.2.5 isinstance checks

- `isinstance(m, bnb.nn.Linear4bit)` — Used in `has_4bit_bnb_layers()` to detect 4-bit models.

### 3.3 The `set_module_tensor_to_device()` Function (Critical Path)

This function is the core weight-loading mechanism for all HuggingFace model loading. It handles
bnb parameters specially:

1. **Shape mismatch tolerance**: Allows shape mismatches for `Params4bit` (since packing changes shape)
2. **CPU-first strategy**: Moves quantized params to CPU first, then to GPU (required for bnb quantization)
3. **Auto-quantization**: After setting weight, checks if `Linear8bitLt.weight.SCB` or
   `Linear4bit.weight.quant_state` is `None` — if so, calls `.to(device_index)` to trigger
   quantization
4. **8-bit CPU offloading**: Special path that quantizes on GPU, then offloads the int8 weights
   and SCB stats to disk

### 3.4 BnbQuantizationConfig

The `BnbQuantizationConfig` dataclass in `utils/dataclasses.py` has these bnb-relevant fields:

| Field | Maps to |
|---|---|
| `load_in_8bit` | Use `bnb.nn.Linear8bitLt` |
| `load_in_4bit` | Use `bnb.nn.Linear4bit` |
| `llm_int8_threshold` | `threshold` kwarg to `Linear8bitLt` |
| `bnb_4bit_quant_type` | `quant_type` kwarg to `Linear4bit` |
| `bnb_4bit_use_double_quant` | `compress_statistics` kwarg to `Linear4bit` |
| `bnb_4bit_compute_dtype` | `compute_dtype` kwarg to `Linear4bit` |
| `torch_dtype` | dtype for non-quantized layers |
| `skip_modules` | modules to not convert |
| `keep_in_fp32_modules` | modules to keep in fp32 |

### 3.5 FSDP2 Compatibility

In `fsdp_utils.py`, accelerate checks for `Params4bit` by class name to disable
`cpu_ram_efficient_loading` when 4-bit parameters are present, since FSDP2 cannot handle
bnb parameter types during CPU-efficient loading.

### 3.6 Summary of Breaking-Change Surfaces

| bnb API | Risk if changed | Impact |
|---|---|---|
| `Linear8bitLt` constructor signature | HIGH | Model loading/quantization breaks |
| `Linear4bit` constructor signature | HIGH | Model loading/quantization breaks |
| Class name `Int8Params` | HIGH | Weight loading fails (string-based check) |
| Class name `Params4bit` | HIGH | Weight loading fails, FSDP compat breaks |
| Class name `Linear8bitLt` | HIGH | Auto-quantization trigger fails |
| Class name `Linear4bit` | HIGH | Auto-quantization trigger fails |
| `Int8Params.__dict__` round-trip | HIGH | Weight loading breaks |
| `Params4bit.__dict__` round-trip | HIGH | Weight loading breaks |
| `Int8Params.SCB` attribute | HIGH | Offloading and quantization detection breaks |
| `Int8Params.CB` attribute | MEDIUM | CPU offloading path breaks |
| `Params4bit.quant_state` attribute | MEDIUM | Auto-quantization detection breaks |
| `.to(device)` triggering quantization | HIGH | The entire load pipeline depends on this |

---

## 4. Text Generation Inference (TGI)

**Repository**: https://github.com/huggingface/text-generation-inference
**Integration depth**: Medium — TGI reimplements its own linear wrappers around bnb primitives.
**Notable**: TGI does NOT use `bnb.nn.Linear8bitLt` or `bnb.nn.Linear4bit`. It builds its own.

### 4.1 Architecture of the Integration

TGI creates custom wrapper modules in `server/text_generation_server/layers/bnb.py` that
bypass bnb's high-level `nn.Module` classes and call bnb's lower-level APIs directly:

```
TGI layers/bnb.py:
  BNBWeight    → wraps weight for 8-bit, calls own Linear8bitLt
  BNBFP4Weight → wraps weight for fp4, calls own Linear4bit(quant_type="fp4")
  BNBNF4Weight → wraps weight for nf4, calls own Linear4bit(quant_type="nf4")
  Linear8bitLt → custom 8-bit linear using bnb.MatmulLtState + bnb.matmul()
  Linear4bit   → custom 4-bit linear using bnb.nn.Params4bit + bnb.matmul_4bit()
```

The Rust launcher (`launcher/src/main.rs`) maps quantization strings `"bitsandbytes"`,
`"bitsandbytes-nf4"`, and `"bitsandbytes-fp4"` to the Python weight loaders.

### 4.2 bnb APIs Called Directly

#### 4.2.1 Low-level matmul APIs

- **`bnb.matmul(x, self.weight, bias=self.bias, state=self.state)`** — Called in the custom
  `Linear8bitLt.forward()`. This is the core 8-bit matmul function.

- **`bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)`** —
  Called in the custom `Linear4bit.forward()`. This is the core 4-bit matmul function.

#### 4.2.2 State and parameter types

- **`bnb.MatmulLtState()`** — Constructed directly in `Linear8bitLt.__init__()` to manage
  8-bit matmul state.

- **`bnb.nn.Int8Params(weight.data, has_fp16_weights=..., requires_grad=...)`** — Constructed
  in `Linear8bitLt.__init__()` for weight storage.

- **`bnb.nn.Params4bit(weight.data, requires_grad=False, compress_statistics=True, quant_type=...)`**
  — Constructed in `Linear4bit.__init__()`.

#### 4.2.3 State attributes accessed

**On `MatmulLtState` (directly constructed):**
- `.threshold` — Set to the outlier threshold
- `.has_fp16_weights` — Set to control weight format
- `.memory_efficient_backward` — Set (though deprecated)
- `.use_pool` — Set to `True` when threshold > 0 and not fp16 weights
- `.is_training` — Set to `self.training` each forward pass
- `.CB` — Accessed to check if initialization needed, deleted after first pass
- `.CxB` — Accessed to get the turing/ampere format weights after first pass
- `.SCB` — Accessed during `init_8bit_state()`

**On `Int8Params`:**
- `.CB` — Column-major quantized weights, moved to state during `init_8bit_state()`
- `.SCB` — Scale column-wise absmax, moved to state during `init_8bit_state()`
- `.cuda(weight.device)` — Called to trigger quantization on GPU
- `.data` — Replaced with `self.state.CxB` after first forward pass

**On `Params4bit`:**
- `.quant_state` — Accessed for matmul_4bit call
- `.t()` — Transposed for matmul_4bit
- `.cuda(weight.device)` — Called to trigger quantization on GPU

### 4.3 Key Differences from Other Integrations

1. **No use of bnb.nn.Linear8bitLt or Linear4bit modules** — TGI builds its own forward pass
   from the lower-level `bnb.matmul()` and `bnb.matmul_4bit()` functions.

2. **Direct MatmulLtState management** — TGI constructs and manages the state object itself,
   giving it control over the quantization lifecycle but coupling it to the state's internal
   attributes.

3. **Weight format optimization in forward** — After the first forward pass, TGI replaces the
   weight data with the turing/ampere format (`self.state.CxB`) and deletes the column-major
   format (`self.state.CB`) for better performance.

4. **Hardcoded settings** — `compress_statistics=True` always, `threshold=6.0` for 8-bit,
   no support for user-configurable compute_dtype on 4-bit.

### 4.4 Summary of Breaking-Change Surfaces

| bnb API | Risk if changed | Impact |
|---|---|---|
| `bnb.matmul()` signature | CRITICAL | All TGI 8-bit inference breaks |
| `bnb.matmul_4bit()` signature | CRITICAL | All TGI 4-bit inference breaks |
| `bnb.MatmulLtState` class | CRITICAL | All TGI 8-bit inference breaks |
| `MatmulLtState.CB`, `.SCB`, `.CxB` | HIGH | 8-bit weight management breaks |
| `MatmulLtState.threshold`, `.has_fp16_weights` | HIGH | 8-bit behavior changes |
| `MatmulLtState.is_training` | MEDIUM | Forward pass state management breaks |
| `MatmulLtState.use_pool` | MEDIUM | Pooling behavior changes |
| `Int8Params` constructor | HIGH | 8-bit weight creation breaks |
| `Int8Params.CB`, `.SCB` attributes | HIGH | Weight initialization breaks |
| `Int8Params.cuda()` triggering quantization | HIGH | Weight loading breaks |
| `Params4bit` constructor | HIGH | 4-bit weight creation breaks |
| `Params4bit.quant_state` attribute | HIGH | 4-bit matmul breaks |
| `Params4bit.t()` (transpose) | MEDIUM | 4-bit matmul input format breaks |
| `Params4bit.cuda()` triggering quantization | HIGH | Weight loading breaks |

---

## 5. vLLM

**Repository**: https://github.com/vllm-project/vllm
**Integration depth**: Deep — vLLM has a full custom model loader and quantization method for bnb.
**Minimum bnb version enforced**: `0.46.1` (checked in both `BitsAndBytesLinearMethod` and `BitsAndBytesMoEMethod`).

### 5.1 Architecture of the Integration

vLLM's bnb integration is split across two main files:

```
vllm/model_executor/layers/quantization/bitsandbytes.py
  → BitsAndBytesConfig (vLLM's own config class)
  → BitsAndBytesLinearMethod (handles weight creation and apply for linear layers)
  → BitsAndBytesMoEMethod (handles weight creation and apply for MoE layers)
  → _apply_bnb_4bit() registered as torch.ops.vllm.apply_bnb_4bit custom op

vllm/model_executor/model_loader/bitsandbytes_loader.py
  → BitsAndBytesModelLoader (handles weight loading, sharding, pre-quantized checkpoints)
```

### 5.2 bnb APIs Called Directly

#### 5.2.1 Low-level matmul APIs

- **`bnb.matmul(x, weight, state=matmul_state)`** — Called in `_apply_8bit_weight()` for each
  weight shard. Same API as TGI uses.

- **`bnb.matmul_4bit(x, weight[offsets[i]:offsets[i+1]].t(), quant_states[i])`** — Called in
  `_apply_bnb_4bit()` for each weight shard. Registered as a custom PyTorch op
  (`torch.ops.vllm.apply_bnb_4bit`) for torch.compile compatibility.

#### 5.2.2 Functional API

- **`bitsandbytes.functional.quantize_4bit(weight, quant_type=..., compress_statistics=..., quant_storage=..., blocksize=...)`**
  — Called in the unquantized weight loading path to quantize weights on-the-fly during model
  loading. Returns `(processed_weight, quant_state)`.

- **`bitsandbytes.functional.dequantize_4bit(weight, quant_state)`** — Called in
  `_apply_4bit_dequnt()` for MoE experts to dequantize before fused expert execution.

- **`bitsandbytes.functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)`** —
  Called in `_dequantize_dq()` to dequantize double-quantized absmax values during weight
  loading (optimization: dequantize at load time, not inference time).

#### 5.2.3 QuantState API

vLLM makes extensive use of `bitsandbytes.functional.QuantState`:

- **`QuantState.from_dict(quant_state_dict, device=...)`** — Called to reconstruct QuantState
  from pre-quantized checkpoint keys (e.g., `weight.quant_state.bitsandbytes__nf4`).

- **`QuantState(absmax=..., shape=..., code=..., blocksize=..., quant_type=..., dtype=...)`** —
  Constructed directly in `_fuse_moe_quant_states()` to create fused quantization states for
  MoE expert weights.

- **QuantState attributes accessed:**
  - `.absmax` — Absmax scales (read, modified in dequantize_dq)
  - `.shape` — Shape of the original weight
  - `.code` — Quantization codebook
  - `.blocksize` — Block size for quantization
  - `.dtype` — Original weight dtype
  - `.nested` — Whether double quantization is used
  - `.state2` — Nested quantization state (second level)
  - `.offset` — Offset for nested quantization

#### 5.2.4 Parameter types

- **`bitsandbytes.nn.Int8Params(data=..., has_fp16_weights=..., requires_grad=...)`** —
  Constructed in `create_qweight_for_8bit()` to create 8-bit quantized weight parameters.

#### 5.2.5 MatmulLtState

- **`bitsandbytes.MatmulLtState()`** — Constructed per shard in `_apply_8bit_weight()`.
  Same state management pattern as TGI: set `.CB`, `.SCB`, `.threshold`, `.has_fp16_weights`,
  `.is_training`, `.use_pool`, then delete `.CB` and replace with `.CxB` after first pass.

### 5.3 Weight Shard Management

vLLM implements tensor-parallel weight sharding for bnb quantized models. This involves:

1. **Shard offsets** (`bnb_shard_offsets`) — Stored as parameter attributes to track where each
   shard's data begins/ends in the packed weight tensor.
2. **Per-shard quant states** (`bnb_quant_state`) — A dict mapping shard index → QuantState,
   stored as a parameter attribute.
3. **Per-shard matmul states** (`matmul_state`) — A list of MatmulLtState objects for 8-bit,
   stored as a parameter attribute.
4. **Generation counter** (`generation`) — Tracks first vs subsequent forward passes to manage
   the CB → CxB format conversion.

The custom op `torch.ops.vllm.apply_bnb_4bit` wraps the per-shard matmul loop and is
registered with a fake implementation for torch.compile support.

### 5.4 Pre-quantized Checkpoint Loading

vLLM supports loading pre-quantized bnb checkpoints. The loader:
1. Scans for keys matching `weight.quant_state.bitsandbytes__nf4` or `__fp4`
2. Reconstructs `QuantState` via `QuantState.from_dict()`
3. Binds the reconstructed states to model parameters as `bnb_quant_state` attributes

For unquantized checkpoints, vLLM quantizes on-the-fly using `bitsandbytes.functional.quantize_4bit()`.

### 5.5 MoE Expert Fusion

vLLM fuses individual expert weights into combined w13 (gate+up) and w2 (down) tensors. During
this process, it:
1. Collects per-expert QuantState objects
2. Concatenates their absmax tensors
3. Constructs new fused QuantState objects with combined shapes
4. Dequantizes during inference via `dequantize_4bit()` before `fused_experts()`

### 5.6 Double Quantization Optimization

vLLM dequantizes double-quantized (nested) absmax values at weight-loading time rather than
inference time. It does this by:
1. Calling `dequantize_blockwise(quant_state.absmax, quant_state.state2)`
2. Adding `quant_state.offset`
3. Setting `quant_state.nested = False` and clearing `.state2`/`.offset`

This modifies the QuantState objects in-place and depends on the specific nested quantization
internal structure.

### 5.7 Known Bug Reference

The code comments reference bitsandbytes issue #1235 (out kwarg not working for matmul_4bit)
and #1342 (quantize_4bit requiring specific device handling). These indicate active coupling
to specific bnb behavior details.

### 5.8 Summary of Breaking-Change Surfaces

| bnb API | Risk if changed | Impact |
|---|---|---|
| `bnb.matmul()` signature | CRITICAL | All vLLM 8-bit inference breaks |
| `bnb.matmul_4bit()` signature | CRITICAL | All vLLM 4-bit inference breaks |
| `functional.quantize_4bit()` signature | HIGH | On-the-fly quantization loading breaks |
| `functional.dequantize_4bit()` signature | HIGH | MoE dequantization breaks |
| `functional.dequantize_blockwise()` | HIGH | Double quant optimization breaks |
| `functional.QuantState` class | CRITICAL | All checkpoint loading breaks |
| `QuantState.from_dict()` | HIGH | Pre-quantized checkpoint loading breaks |
| `QuantState` constructor args | HIGH | MoE state fusion breaks |
| `QuantState.absmax/shape/code/blocksize/dtype/nested/state2/offset` | HIGH | Multiple paths break |
| `MatmulLtState` class and attributes | HIGH | 8-bit inference breaks |
| `Int8Params` constructor | HIGH | 8-bit weight creation breaks |
| `Params4bit` / weight `.t()` semantics | HIGH | 4-bit matmul input format breaks |
| Checkpoint key format (`quant_state.bitsandbytes__nf4`) | CRITICAL | All pre-quantized model loading breaks |

---

## 6. Consolidated API Surface

This section cross-references which bnb APIs are used by which downstream projects. An API
used by all 5 projects is maximally dangerous to change.

### 6.1 Module Types

| bnb type | Transformers | PEFT | Accelerate | TGI | vLLM |
|---|---|---|---|---|---|
| `bnb.nn.Linear4bit` | construct + isinstance | isinstance | construct + name check | — | — |
| `bnb.nn.Linear8bitLt` | construct + isinstance | isinstance | construct + name check | — | — |
| `bnb.nn.Params4bit` | construct + `from_prequantized()` | construct (via `__dict__`) | construct (via `__dict__`) | construct | — |
| `bnb.nn.Int8Params` | construct | construct | construct (via `__dict__`) | construct | construct |

### 6.2 Functional API

| bnb function | Transformers | PEFT | Accelerate | TGI | vLLM |
|---|---|---|---|---|---|
| `functional.dequantize_4bit()` | Yes | Yes | — | — | Yes |
| `functional.int8_vectorwise_dequant()` | Yes | Yes | — | — | — |
| `functional.quantize_4bit()` | — | — | — | — | Yes |
| `functional.dequantize_blockwise()` | — | — | — | — | Yes |
| `functional.QuantState` | — | — | — | — | Yes |
| `functional.QuantState.from_dict()` | — | — | — | — | Yes |
| `bnb.matmul()` | — | — | — | Yes | Yes |
| `bnb.matmul_4bit()` | — | — | — | Yes | Yes |
| `bnb.MatmulLtState` | — | — | — | Yes | Yes |

### 6.3 Module Attributes (Deep Coupling)

| Attribute | Transformers | PEFT | Accelerate | TGI | vLLM |
|---|---|---|---|---|---|
| `Params4bit.quant_state` | Yes | Yes | Yes | Yes | — (uses QuantState directly) |
| `Params4bit.compress_statistics` | Yes | Yes | — | — | — |
| `Params4bit.quant_type` | Yes | Yes | — | — | — |
| `Params4bit.__dict__` round-trip | — | Yes | Yes | — | — |
| `Params4bit.bnb_quantized` | — | Yes | — | — | — |
| `Params4bit.quant_storage` | Yes | Yes | — | — | — |
| `Params4bit.element_size()` | Yes | Yes | — | — | — |
| `Linear4bit.compute_dtype` | Yes | Yes | — | — | — |
| `Int8Params.SCB` | Yes | Yes | Yes | Yes | — |
| `Int8Params.CB` | — | — | Yes | Yes | — |
| `Int8Params.has_fp16_weights` | — | Yes | — | Yes | — |
| `Linear8bitLt.state` | Yes | Yes | — | — | — |
| `MatmulLtState.SCB` | — | — | — | Yes | Yes |
| `MatmulLtState.CB` | — | — | — | Yes | Yes |
| `MatmulLtState.CxB` | — | — | — | Yes | Yes |
| `MatmulLtState.threshold` | — | Yes | — | Yes | Yes |
| `MatmulLtState.has_fp16_weights` | — | Yes | — | Yes | Yes |
| `MatmulLtState.is_training` | — | — | — | Yes | Yes |
| `MatmulLtState.use_pool` | — | — | — | Yes | Yes |
| `MatmulLtState.reset_grads()` | — | Yes | — | — | — |
| `supported_torch_devices` | Yes | — | — | — | — |

### 6.4 Optimizer API

| bnb optimizer API | Transformers | PEFT | Accelerate | TGI | vLLM |
|---|---|---|---|---|---|
| `optim.AdamW` | Yes | — | — | — | — |
| `optim.Lion` | Yes | — | — | — | — |
| `optim.RMSprop` | Yes | — | — | — | — |
| `optim.AdEMAMix` | Yes | — | — | — | — |
| `optim.GlobalOptimManager` | Yes | — | — | — | — |

### 6.5 Serialization Format

| Checkpoint key pattern | Transformers | PEFT | Accelerate | TGI | vLLM |
|---|---|---|---|---|---|
| `weight.absmax` | Yes | — | — | — | Yes |
| `weight.quant_map` | Yes | — | — | — | Yes |
| `weight.nested_absmax` | Yes | — | — | — | Yes |
| `weight.nested_quant_map` | Yes | — | — | — | Yes |
| `weight.quant_state.bitsandbytes__nf4` | Yes | — | — | — | Yes |
| `weight.quant_state.bitsandbytes__fp4` | Yes | — | — | — | Yes |
| `weight.SCB` (8-bit) | Yes | — | Yes | — | — |

---

## 7. General Breaking-Change Checklist

When reviewing a bitsandbytes PR, use this checklist to assess downstream impact:

### 7.1 CRITICAL (will break multiple downstream projects immediately)

- [ ] **Constructor signature changes to `Linear4bit` or `Linear8bitLt`**
  — Used by: Transformers, Accelerate (construction), PEFT (isinstance)
  — Check: Do the kwargs `in_features, out_features, bias, compute_dtype, compress_statistics,
  quant_type, quant_storage` still work? Do `has_fp16_weights, threshold` still work for 8-bit?

- [ ] **Constructor signature changes to `Params4bit` or `Int8Params`**
  — Used by: All 5 projects
  — Check: Does `Params4bit(data, requires_grad=..., **old.__dict__)` still work?
  Does `Int8Params(data, has_fp16_weights=..., requires_grad=...)` still work?

- [ ] **`bnb.matmul()` or `bnb.matmul_4bit()` signature changes**
  — Used by: TGI, vLLM (directly), Transformers/PEFT/Accelerate (indirectly via nn modules)
  — Check: Do the `state=`, `bias=`, `quant_state=` kwargs still work?

- [ ] **`functional.dequantize_4bit()` signature changes**
  — Used by: Transformers, PEFT, vLLM
  — Check: Does `dequantize_4bit(weight.data, weight.quant_state)` still work?

- [ ] **`QuantState` constructor or `from_dict()` changes**
  — Used by: vLLM for checkpoint loading and MoE fusion
  — Check: Do `absmax, shape, code, blocksize, quant_type, dtype` constructor args still work?

- [ ] **Serialization key format changes**
  — Affects: All pre-quantized checkpoints on HuggingFace Hub
  — Check: Are keys like `weight.quant_state.bitsandbytes__nf4`, `weight.absmax`, etc. still valid?

### 7.2 HIGH (will break specific functionality in multiple projects)

- [ ] **`Params4bit.quant_state` attribute changes**
  — Used by: Transformers, PEFT, Accelerate, TGI (all for dequantization)

- [ ] **`Int8Params.SCB` attribute changes**
  — Used by: Transformers, PEFT, Accelerate, TGI (all for 8-bit dequantization)

- [ ] **`MatmulLtState` attribute changes (`.CB`, `.SCB`, `.CxB`)**
  — Used by: TGI, vLLM (for 8-bit forward pass management)

- [ ] **Class renaming** (e.g., `Int8Params` → `Int8Parameter`)
  — Accelerate uses string-based class name checks, not isinstance
  — PEFT's `peft_model.py` uses `param.__class__.__name__ == "Params4bit"`

- [ ] **`Params4bit.__dict__` round-trip behavior changes**
  — PEFT and Accelerate reconstruct params via `Params4bit(data, **old_params.__dict__)`
  — Adding new required constructor args that aren't in `__dict__` will break this

- [ ] **`.to(device)` / `.cuda()` triggering quantization**
  — Accelerate and TGI depend on this behavior for weight loading

### 7.3 MEDIUM (will break specific features or have fallback paths)

- [ ] **`functional.int8_vectorwise_dequant()` changes**
  — Transformers and PEFT have manual math fallback

- [ ] **`MatmulLtState.reset_grads()` removal**
  — Only PEFT uses this (during merge/unmerge)

- [ ] **Optimizer class changes** (`optim.AdamW`, `optim.Lion`, etc.)
  — Only Transformers trainer uses these

- [ ] **`supported_torch_devices` module attribute changes**
  — Only Transformers uses this, with `getattr()` fallback

### 7.4 Integration-Specific Concerns

| Project | Specific concern |
|---|---|
| Transformers | Conv1D transpose before quantization — depends on weight shape semantics |
| PEFT | 4-bit `result.clone()` workaround — depends on forward output being a view |
| Accelerate | String-based class name checks — sensitive to renaming, not subclassing |
| TGI | Reimplements forward pass — sensitive to low-level matmul semantics |
| vLLM | Custom op registration — sensitive to matmul_4bit signature and QuantState internals |
| vLLM | MoE expert fusion — constructs QuantState manually from component parts |
| vLLM | Double quant dequant at load time — modifies QuantState.nested internals |

### 7.5 Safe Changes (unlikely to break downstream)

- Adding new optional parameters to constructors (with defaults)
- Adding new functional API functions
- Adding new module types (new quantization methods)
- Performance improvements that don't change API behavior
- Adding new optimizer variants
- Internal refactoring that preserves all public interfaces
- Bug fixes that make behavior match documented semantics
