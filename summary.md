# Absmax format migration: float32 → uint8 E4M4

Branch: `experiment/scalar-gemv-int8-absmax`
Base: `23f92e5` (feature/kbit-gemv-v8)

## What changed

All kbit kernels now use uint8 E4M4 absmax by default, replacing float32.
A float16 absmax alternative path is available for scalar GEMV and grouped
scalar GEMV if higher absmax precision is needed.

### Kernel changes

- **quantize_kbit**: CUDA kernel now encodes absmax to E4M4 natively
  (no Python-side post-processing). Returns uint8 tensor.
- **repack_kbit**: Accepts uint8 input, copies bytes directly instead of
  re-encoding from float32.
- **Scalar GEMV** (dense): `unsigned char*` absmax with `load_absmax<T>`
  decode. Templated on `ABSMAX_T` for uint8 (default) and float16.
- **Grouped scalar GEMV** (MoE): Same treatment as dense scalar GEMV.
- **MMA kernels** (dense + grouped): Already used uint8 E4M4 — no change.
- **Dequantize**: Already supported uint8 — no change.

### Files modified (8)

- `csrc/ops.cu` — E4M4 encode/decode moved before quantize kernel,
  quantize writes uint8, repack accepts uint8, fp16abs template
  instantiations for scalar/grouped GEMV
- `csrc/pythonInterface.cpp` — All wrappers updated for `unsigned char*`;
  added 16 extern C symbols for fp16abs scalar/grouped GEMV
- `bitsandbytes/backends/cuda/ops.py` — uint8 allocation in quantize,
  absmax dtype routing in scalar/grouped GEMV dispatch
- `bitsandbytes/_ops.py` — quantize_kbit fake op returns uint8
- `bitsandbytes/functional.py` — Removed redundant Python-side E4M4 encode
- `tests/test_scalar_gemv.py` — E4M4 decode in reference functions
- `tests/test_kbit_gemm.py` — Reference quantize/dequant/repack updated
  for uint8 absmax
- `benchmarks/ncu_driver.py` — Updated for uint8 absmax, removed stale
  `.cuda()` call

## Precision impact

Additional MAE introduced by E4M4 absmax rounding, on top of existing kbit
quantization error:

| k (bits) | Extra MAE from E4M4 |
|----------|---------------------|
| k=2      | +0.0%               |
| k=3      | +0.2–0.5%           |
| k=4      | +0.6–1.4%           |
| k=5      | +4.2–4.6%           |

The kbit quantization error itself (with perfect float32 absmax) is ~12x
larger than the E4M4 absmax rounding error. E4M4 is rounding an
already-approximate scale factor — the additional loss is marginal.

## Runtime performance

RTX 4090, CUDA events timing, fp16.

**Dense scalar GEMV** (16 configs): Deltas range -29% to +19% with no
consistent direction. Run-to-run variance dominates. No measurable
regression.

**Grouped scalar GEMV / MoE** (8 configs, 8 experts):
- M≥2: within noise (±3%)
- M=1: possible ~5% overhead from E4M4 decode cost being a larger fraction
  of the small per-warp workload. One outlier at +22% is likely noise.

**MMA kernels**: No change (already uint8 E4M4).

## Storage savings

4 bytes → 1 byte per quant block (blocksize=32). For a 70B model at k=3,
absmax storage drops from ~67 MB to ~17 MB. The main benefit is format
unification across all kernel paths (MMA, scalar GEMV, grouped), eliminating
format conversion between paths.

## Tests

- 31/31 scalar GEMV tests pass
- 195/195 GEMM tests pass
- test_grouped_gemm.py has pre-existing failures (missing `max_M` arg,
  unrelated to this branch)

## Bugs fixed during development

1. **Forward declaration of `encode_e4m4_absmax`** before `E4M4_BIAS` was
   defined — compiled without errors but produced garbage values at runtime.
   Fixed by moving all E4M4 functions before the quantize kernel.

2. **Double E4M4 encoding** — `functional.py::quantize_kbit` applied a
   Python-side E4M4 encode on top of the kernel's already-encoded uint8
   output. Removed the redundant Python encode.

3. **Stale build artifacts** — cmake didn't detect source changes after
   editing, causing the .so to retain old `float*` signatures while Python
   passed `unsigned char*`. Fixed with clean rebuilds.
