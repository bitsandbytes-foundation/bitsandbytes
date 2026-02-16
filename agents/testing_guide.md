# Testing Guide for bitsandbytes

## Quick Start

Run the full test suite with optimal parallelization:

```bash
pytest tests/ -v --tb=short -n 4
```

`-n 4` (4 pytest-xdist workers) is the recommended default for any machine.

## Why 4 Workers?

Benchmarks across two machines with very different hardware show that `-n 4` is consistently the fastest configuration. Going higher provides no benefit and often makes things worse.

### Benchmark Data

**Machine A:** AMD Threadripper 1900X (8 cores / 16 threads), RTX 4090 (24 GB), CUDA 12.4

| Workers | Wall Time | Speedup vs n=1 | Avg CPU | Avg GPU | Failures |
|---------|-----------|-----------------|---------|---------|----------|
| 1       | 1319s     | 1.00x           | 32.5%   | 3.4%    | 0        |
| **4**   | **565s**  | **2.33x**       | 70.5%   | 12.9%   | 0        |
| 6       | 588s      | 2.24x           | 74.8%   | 10.9%   | 7 (OOM)  |
| 8       | 570s      | 2.31x           | 87.9%   | 12.5%   | 7 (OOM)  |

**Machine B:** AMD Threadripper PRO 9975WX (32 cores / 64 threads), RTX PRO 6000 Blackwell (98 GB), CUDA 13.0

| Workers | Wall Time | Speedup vs n=1 | Avg CPU | Avg GPU | Failures |
|---------|-----------|-----------------|---------|---------|----------|
| 1       | 428s      | 1.00x           | 13.4%   | 3.1%    | 25*      |
| **4**   | **322s**  | **1.33x**       | 75.3%   | 5.7%    | 25*      |
| 8       | 578s      | 0.74x (slower)  | 91.9%   | 3.5%    | 25*      |
| 16      | 566s      | 0.76x (slower)  | 97.0%   | 6.2%    | 25*      |
| 24      | 560s      | 0.76x (slower)  | 97.2%   | 6.2%    | 40       |

\* Blackwell-specific failures unrelated to worker count (see Known Issues below).

### Analysis

- **GPU utilization stays very low** (3-13%) regardless of worker count. The tests are primarily CPU-bound: short GPU kernel bursts interleaved with Python/numpy work for test setup, tensor creation, and result validation.
- **4 workers is the sweet spot** because it balances overlapping CPU prep with GPU execution across workers. Each worker can prepare data while another waits on a GPU kernel.
- **Beyond 4 workers, overhead dominates.** Additional workers add pytest-xdist coordination costs and per-worker CUDA context overhead without meaningful GPU throughput gain. On Machine B, `-n 8` was nearly 2x slower than `-n 4` despite 75% idle CPU at `-n 4`.
- **Per-core CPU speed matters more than core count.** Machine B is 3.1x faster single-threaded (Zen 5 vs Zen 1). Having 4x more cores provided no additional benefit at the optimal worker count.
- **GPU memory affects reliability, not speed.** More free VRAM avoids OOM failures at higher worker counts but does not improve throughput.

### What About More/Fewer Workers?

| Situation | Recommendation |
|-----------|---------------|
| Default | `-n 4` |
| Low GPU memory (<8 GB free) | `-n 2` to avoid OOM |
| Running a subset of tests | `-n 4` still fine |
| Single specific test | No `-n` flag needed |
| CI environment | `-n 4` |

## Useful pytest Options

```bash
# Full suite, optimal speed
pytest tests/ -v --tb=short -n 4

# With timing breakdown of slowest tests
pytest tests/ -v --tb=short -n 4 --durations=20

# Run a specific test file
pytest tests/test_functional.py -v --tb=short -n 4

# Run tests matching a keyword
pytest tests/ -v --tb=short -n 4 -k "4bit"

# Stop on first failure
pytest tests/ -v --tb=short -n 4 -x

# Single worker (debugging, deterministic output)
pytest tests/ -v --tb=long
```

## Test Suite Characteristics

The full suite has ~7500 parametrized tests. Most of the wall-clock time is consumed by a small number of test functions with many parametrizations:

- **`test_gemv_4bit`** dominates (~70% of total time) with 1500+ combinations. CPU variants at dim=1024 take 16-20s each; CUDA variants finish in ~0.05s.
- **`test_functional.py`** alone accounts for ~80% of total test time.
- **CPU tests are the bottleneck**: 81% of total time despite being only 37% of test count.
- **87% of individual tests finish under 1 second**, but the remaining 13% consume 80% of wall-clock time.

## Known Issues by Architecture

### Blackwell (sm_120, e.g. RTX PRO 6000)

25 tests fail on Blackwell as of the `main` branch (Feb 2026):

1. **Int8 batched matmul (`test_ibmm`) - 16 failures**: cuBLAS returns `CUBLAS_STATUS_NOT_SUPPORTED` (status 15) for the int8 batched GEMM path on Blackwell. The legacy cuBLAS int8 API is not supported on sm_120. These tests produce garbage output (100% element mismatch). A fix would require migrating to cublasLt or a different int8 GEMM implementation.

2. **FP4 quantization at blocksize=256 - 9 failures**: Relative error is marginally above the threshold (e.g., 0.29091 vs limit of 0.2908). Only affects `fp4` at `blocksize=256` on CUDA across all dtypes (fp32, fp16, bf16). The `nf4` quant type and other blocksizes pass. This is a minor numerical difference in fp4 dequantization likely caused by different FP rounding behavior on Blackwell.

### Ada Lovelace (sm_89, e.g. RTX 4090)

No architecture-specific failures. All tests pass with `-n 4`.

## Build Before Testing

Tests require a compiled native library matching your GPU and CUDA toolkit. See `COMPILE_H100_L40.md` for build instructions. Quick version:

```bash
# Find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Build (replace 89 with your compute capability, e.g. 120 for Blackwell)
cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="89" -S . -B build
cmake --build build -j$(nproc)

# If your CUDA toolkit version differs from PyTorch's CUDA version, create a symlink:
# e.g., toolkit is 12.4 but PyTorch expects 12.8:
ln -sf bitsandbytes/libbitsandbytes_cuda124.so bitsandbytes/libbitsandbytes_cuda128.so

# Install in editable mode
pip install -e .
```

## Test Dependencies

```bash
pip install einops lion-pytorch pytest pytest-xdist scipy transformers
```
