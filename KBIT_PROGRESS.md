# K-Bit Quantization Implementation Progress

**Branch**: `feature/kbit-quantization` (worktree at `~/git/bitsandbytes-kbit`)
**Spec files**: `cuda-spec.md`, `cuda-spec-additions.md` (in main repo, gitignored)

## Completed

### Stage 0: Pure Python Reference -- DONE
- File: `tests/test_kbit_quantization.py`
- Functions: `create_normal_float_codebook()`, `quantize_kbit_ref()`, `dequantize_kbit_ref()`, `pack_kbit_ref()`, `unpack_kbit_ref()`
- 57 tests pass (codebook generation, round-trip, MSE ordering, error bounds, pack/unpack)
- Serves as permanent ground truth for all CUDA validation

### Stages 1-5: CUDA Kernels -- CODE WRITTEN, BUILD ISSUE

All CUDA kernel code is written and compiles, but there's a **device linker issue** preventing the kernels from appearing in the final `.so`.

#### Files modified:

1. **`csrc/kernels.cu`** (appended at end, ~200 lines):
   - `warp_reduce_absmax()` -- device helper for warp-level max reduction
   - `pack_kbit_warp<K>()` -- device helper, __ballot_sync bit-plane packing
   - `unpack_kbit_warp<K>()` -- device helper, bit extraction unpacking
   - `kTestPackUnpack_kbit<K>` -- Stage 1 test kernel (in-warp round-trip)
   - `kTestPackWrite_kbit<K>` -- Stage 2 test kernel (pack to global memory)
   - `kTestReadUnpack_kbit<K>` -- Stage 2 test kernel (read from global memory)
   - `kTestCodebookLookup_kbit<K>` -- Stage 3 test kernel (shfl_sync codebook)
   - `kQuantizeBlockwise_kbit<T, K>` -- Stage 4 production quantize kernel
   - `kDequantizeBlockwise_kbit<T, K>` -- Stage 5 production dequantize kernel
   - Template instantiation macros for K=2,3,4,5 x T=half,bf16,float

2. **`csrc/kernels.cuh`** (appended before `#endif`):
   - Forward declarations of all kernel templates

3. **`csrc/ops.cu`** (appended at end, ~100 lines):
   - Launch wrappers: `test_pack_unpack_kbit<K>()`, `test_pack_write_kbit<K>()`, etc.
   - Launch wrappers: `quantizeBlockwise_kbit<T,K>()`, `dequantizeBlockwise_kbit<T,K>()`
   - Grid calculation: `ceil(n/32)/8` CUDA blocks, 256 threads per block
   - Template instantiation macros

4. **`csrc/pythonInterface.cpp`** (two sections added):
   - Unmangled wrappers (inside `#if BUILD_CUDA || BUILD_HIP`): `test_pack_unpack_k{K}()`, `quantize_kbit_{fp16,bf16,fp32}_k{K}()`, etc.
   - extern "C" wrappers: `ctest_pack_unpack_k{K}()`, `cquantize_kbit_{tname}_k{K}()`, `cdequantize_kbit_{tname}_k{K}()`, etc.

5. **`tests/test_kbit_quantization.py`** (comprehensive test file):
   - Python reference tests (Stage 0): `TestCodebook`, `TestQuantizeRef`, `TestPackUnpackRef`
   - CUDA ctypes wrappers: `_cuda_test_pack_unpack()`, `_cuda_quantize_kbit()`, `_cuda_dequantize_kbit()`, etc.
   - CUDA tests (Stages 1-5): `TestStage1PackUnpackCUDA`, `TestStage2PackMemoryCUDA`, `TestStage3CodebookLookupCUDA`, `TestStage4QuantizeCUDA`, `TestStage5DequantizeCUDA`

## Current Blocker: RDC Device Linking

### Problem
The compiled kernels exist in the `.o` object files (verified via `nm`), and the C-level symbols are exported in the final `.so` (verified via `nm -D`), but the **CUDA device code** (fatbinary) does not contain the new kernel functions. Running any kernel gives "invalid device function".

### Root Cause
The project uses `-rdc=true` (relocatable device code) for separate compilation. The device link step (`cmake_device_link.o`) needs to resolve all device-side references. The template instantiations in `kernels.cu` produce weak symbols in the object file, but the device linker may not be pulling them in because they're not referenced from the device link compilation unit.

### How to Fix (options)

1. **Add `__global__` function declarations to the device link file**: Check how CMake generates the device link step and ensure it sees all `.cu` object files.

2. **Use `--relocatable-device-code=false` for the kbit kernels**: If the kbit kernels don't need cross-file device calls, they could be compiled without RDC. But this requires CMake changes.

3. **Move kernel definitions to the same file as the launch wrappers**: Instead of splitting between `kernels.cu` (kernel definitions) and `ops.cu` (launch wrappers), put everything in a single `.cu` file. This is the simplest fix -- add the kernel bodies directly to `ops.cu` or create a new `kbit_kernels.cu` that contains both kernels and launch wrappers.

4. **Check CMakeLists.txt for device link configuration**: The CMake `CUDA_SEPARABLE_COMPILATION` property or `CUDA_RESOLVE_DEVICE_SYMBOLS` might need adjustment.

**Recommended fix**: Option 3 -- move all kbit kernel code from `kernels.cu` into `ops.cu` (or a new self-contained file). This sidesteps the RDC linking issue entirely since the kernel and its launch site would be in the same compilation unit.

## Build Instructions

```bash
cd ~/git/bitsandbytes-kbit
cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="89;90" -S . -B build
make -C build -j$(nproc)
ln -sf libbitsandbytes_cuda124.so bitsandbytes/libbitsandbytes_cuda128.so
```

## Test Instructions

```bash
# Python-only tests (all pass)
python -m pytest tests/test_kbit_quantization.py -k "not CUDA" -v

# CUDA tests (currently fail due to device link issue)
python -m pytest tests/test_kbit_quantization.py -k "CUDA" -v
```

## Not Yet Implemented

- Stages 6-8: Error analysis, NF4 cross-validation, performance benchmarking (test code not written)
- Python API in `bitsandbytes/functional.py` (quantize_kbit, dequantize_kbit)
- `torch.library` registration in `bitsandbytes/_ops.py`
- Codebook caching/registration system
