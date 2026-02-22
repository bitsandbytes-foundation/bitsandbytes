# Spec: Add `out` parameter to kbit dequantize for CUDA graph compatibility

## Problem

`dequantize_kbit` allocates a fresh output tensor on every call. This breaks
CUDA graph capture, which requires kernels to write to the same memory address
on every replay. The dequant is on the inference hot path and needs graph support.

## Changes

### 1. CUDA backend (`bitsandbytes/backends/cuda/ops.py`)

Factor the kernel call into `_dequantize_kbit_impl(packed, codebook, absmax, k, n, dtype, out)`:
- Accepts a pre-allocated `out` tensor
- Validates `out` shape, dtype, device
- Calls the C kernel writing into `out`

The existing `dequantize_kbit` registered kernel allocates `out` then calls `_impl`.

### 2. torch op definition (`bitsandbytes/_ops.py`)

Add a second op `bitsandbytes::dequantize_kbit_` (in-place variant with trailing
underscore, matching existing pattern for `dequantize_4bit`):
- Signature: `(Tensor packed, Tensor codebook, Tensor absmax, int k, int n, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)`
- Fake implementation validates shapes, returns `out`

### 3. Public API (`bitsandbytes/functional.py`)

Add optional `out` parameter to `dequantize_kbit()`:
- `out: Optional[Tensor] = None`
- If provided, validate shape/dtype/device, pass to impl
- If None, allocate as before

### 4. Tests

Add test cases in `tests/test_kbit_quantization.py`:
- Dequant with pre-allocated `out` tensor matches normal dequant
- `out` tensor with wrong shape raises error
- `out` tensor with wrong dtype raises error

## Files touched

- `bitsandbytes/backends/cuda/ops.py`
- `bitsandbytes/_ops.py`
- `bitsandbytes/functional.py`
- `tests/test_kbit_quantization.py`

## Not in scope

- `quantize_kbit` out parameter (runs once at model load, not on hot path)
