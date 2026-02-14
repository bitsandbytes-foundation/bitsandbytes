# K-Bit Quantization Implementation Progress

**Branch**: `feature/kbit-quantization` (worktree at `~/git/bitsandbytes-kbit`)
**Spec files**: `cuda-spec.md`, `cuda-spec-additions.md` (in main repo root, gitignored)

## Status: ALL STAGES COMPLETE (0-8 + Python API), 218/218 tests passing

Full k-bit quantization pipeline is working end-to-end: CUDA kernels, error validation, NF4 cross-validation, performance benchmarks, and public Python API.

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

### Stage 6: Round-Trip Error Analysis
- Analytical error bound verified on 1M+ elements (zero violations)
- MSE monotonically decreases with increasing K
- SQNR thresholds: K=2 >5dB, K=3 >10dB, K=4 >15dB, K=5 >20dB (all pass)
- All dtypes produce finite, reasonable MSE
- Tests: `TestStage6ErrorAnalysis`

### Stage 7: NF4 Cross-Validation
- K=4 kbit MSE within 2x of existing NF4 MSE (different blocksizes: 32 vs 64)
- Our K=4 NF codebook similar to existing NF4 codebook (max diff <0.15)
- Using exact same NF4 codebook, CUDA output matches Python reference within 1e-4
- All dtypes work with NF4 codebook
- Tests: `TestStage7NF4CrossValidation`

### Stage 8: Performance Benchmarking
- Dequant bandwidth utilization >10% of peak for all K (L40 GPU)
- Throughput scales roughly linearly with tensor size
- K=4 kbit dequant within 10x of existing NF4 dequant throughput
- Tests: `TestStage8PerformanceBenchmark`

### Python API
- `bitsandbytes/functional.py`: `quantize_kbit()`, `dequantize_kbit()`, `create_normal_float_codebook()`
- `bitsandbytes/_ops.py`: `torch.library` definitions with fake/abstract implementations
- `bitsandbytes/backends/cuda/ops.py`: CUDA kernel registration via `register_kernel`
- Codebook caching: precomputed NF codebooks cached per (k, device) pair
- Tests: `TestPythonAPI` (round-trip, all dtypes, custom codebook, various sizes, matches ctypes path)

## Files Modified (relative to main branch)

| File | What changed |
|------|-------------|
| `csrc/ops.cu` | Kernel definitions + device helpers + launch wrappers (~280 lines appended) |
| `csrc/kernels.cu` | Removed: just a comment pointing to ops.cu |
| `csrc/kernels.cuh` | Removed stale forward declarations (was causing "invalid device function") |
| `csrc/pythonInterface.cpp` | Unmangled wrappers + extern "C" exports for all kbit functions |
| `CMakeLists.txt` | Added `CUDA_RESOLVE_DEVICE_SYMBOLS ON` |
| `bitsandbytes/functional.py` | Public API: `quantize_kbit`, `dequantize_kbit`, `create_normal_float_codebook` |
| `bitsandbytes/_ops.py` | `torch.library` definitions for `quantize_kbit` and `dequantize_kbit` |
| `bitsandbytes/backends/cuda/ops.py` | CUDA kernel registrations for kbit ops |
| `tests/test_kbit_quantization.py` | Full test file: 218 tests across all stages + API |

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

## Python API

```python
from bitsandbytes.functional import quantize_kbit, dequantize_kbit

# Quantize (auto-generates NF codebook)
packed, absmax, codebook = quantize_kbit(A, k=4)

# Dequantize
recovered = dequantize_kbit(packed, absmax, codebook, k=4, n=A.numel(), dtype=A.dtype)

# Custom codebook
my_cb = torch.linspace(-1, 1, 8).cuda()
packed, absmax, _ = quantize_kbit(A, k=3, codebook=my_cb)
```

## Build & Test

```bash
cd ~/git/bitsandbytes-kbit
cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="89;90" -S . -B build
make -C build -j$(nproc)
ln -sf libbitsandbytes_cuda124.so bitsandbytes/libbitsandbytes_cuda128.so
python -m pytest tests/test_kbit_quantization.py -p no:randomly -v   # 218 pass
```

## Remaining Cleanup (optional)

- Remove temporary test kernels (Stages 1-3) and `ctest_*` exports from pythonInterface.cpp
- Remove this progress report once merged
