# K-Bit Quantization Implementation Progress

**Branch**: `feature/kbit-quantization` (worktree at `~/git/bitsandbytes-kbit`)
**Spec files**: `cuda-spec.md`, `cuda-spec-additions.md` (in main repo root, gitignored)

## Status: Stages 0-5 COMPLETE, 157/157 tests passing

All CUDA kernels are working. The full quantize/dequantize pipeline runs on GPU, validated against the Python reference.

## What's Done

### Stage 0: Pure Python Reference
- File: `tests/test_kbit_quantization.py` (top half)
- `create_normal_float_codebook(k)` -- generates 2^k NF codebook from N(0,1) quantiles
- `quantize_kbit_ref(A, codebook)` -- pure PyTorch blockwise quantize (blocksize=32)
- `dequantize_kbit_ref(indices, absmax, codebook)` -- pure PyTorch dequantize
- `pack_kbit_ref(indices, k)` / `unpack_kbit_ref(packed, k, n)` -- bit-plane packing reference
- Tests: `TestCodebook`, `TestQuantizeRef`, `TestPackUnpackRef`

### Stages 1-3: CUDA Test Kernels (temporary scaffolding)
- `kTestPackUnpack_kbit<K>` -- in-warp __ballot_sync pack / bit-extract unpack round-trip
- `kTestPackWrite_kbit<K>` / `kTestReadUnpack_kbit<K>` -- persistent memory format
- `kTestCodebookLookup_kbit<K>` -- __shfl_sync codebook lookup
- Tests: `TestStage1PackUnpackCUDA`, `TestStage2PackMemoryCUDA`, `TestStage3CodebookLookupCUDA`

### Stage 4: Full Quantize Kernel
- `kQuantizeBlockwise_kbit<T, K>` -- warp-level absmax reduction, branchless codebook search, ballot_sync bit-plane packing
- CUDA indices match Python reference exactly
- Tests: `TestStage4QuantizeCUDA` (absmax correctness, indices match ref, all dtypes, various sizes)

### Stage 5: Full Dequantize Kernel
- `kDequantizeBlockwise_kbit<T, K>` -- bit-plane unpacking, shfl_sync codebook lookup, absmax scaling
- Round-trip error within analytical bounds for all K
- Tests: `TestStage5DequantizeCUDA` (matches ref, all dtypes, various sizes, error bounds)

## Files Modified (relative to main branch)

| File | What changed |
|------|-------------|
| `csrc/ops.cu` | Kernel definitions + device helpers + launch wrappers (~280 lines appended) |
| `csrc/kernels.cu` | Removed: just a comment pointing to ops.cu |
| `csrc/kernels.cuh` | Removed stale forward declarations (was causing "invalid device function") |
| `csrc/pythonInterface.cpp` | Unmangled wrappers + extern "C" exports for all kbit functions |
| `CMakeLists.txt` | Added `CUDA_RESOLVE_DEVICE_SYMBOLS ON` |
| `tests/test_kbit_quantization.py` | Full test file: Python ref + CUDA tests + ctypes wrappers |

### Key Architecture Decision During Implementation

Kernel definitions MUST live in `ops.cu` (same file as launch wrappers), not in `kernels.cu`. The project uses CUDA separable compilation (`-rdc=true`), and having forward declarations in `kernels.cuh` (without `__restrict__`) alongside definitions in a different TU (with `__restrict__`) caused mismatched CUDA function registration. Keeping everything in one compilation unit avoids this entirely.

## C Interface (exported symbols)

Test kernels (prefix `ctest_`):
- `ctest_pack_unpack_k{2,3,4,5}(indices, recovered, n)`
- `ctest_pack_write_k{2,3,4,5}(indices, packed_out, n)`
- `ctest_read_unpack_k{2,3,4,5}(packed_in, indices_out, n)`
- `ctest_codebook_lookup_k{2,3,4,5}(indices, codebook, out, n)`

Production kernels:
- `cquantize_kbit_{fp16,bf16,fp32}_k{2,3,4,5}(codebook, A, absmax, packed_out, n)`
- `cdequantize_kbit_{fp16,bf16,fp32}_k{2,3,4,5}(packed_in, codebook, absmax, out, n, stream)`

## Build & Test

```bash
cd ~/git/bitsandbytes-kbit
cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="89;90" -S . -B build
make -C build -j$(nproc)
ln -sf libbitsandbytes_cuda124.so bitsandbytes/libbitsandbytes_cuda128.so
python -m pytest tests/test_kbit_quantization.py -p no:randomly -v   # 157 pass
```

## Not Yet Implemented

### Stages 6-8 (test scripts only, no new kernels needed)
- **Stage 6**: Round-trip error analysis (analytical bounds, empirical MSE on large tensors)
- **Stage 7**: Cross-validate K=4 against existing NF4 dequant
- **Stage 8**: Performance benchmarking (measure HBM bandwidth utilization, target 60-80%)

### Python API
- `bitsandbytes/functional.py`: `quantize_kbit()` and `dequantize_kbit()` public functions
- `bitsandbytes/_ops.py`: `torch.library` registration
- Codebook caching/registration system (precomputed NF codebooks for K=2..5)

### Cleanup
- Remove temporary test kernels (Stages 1-3) after confirming Stages 4+5 are solid
- Remove `ctest_*` exports from pythonInterface.cpp
- Update KBIT_PROGRESS.md or remove it
