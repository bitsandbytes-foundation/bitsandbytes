# NVFP4 GEMM Benchmark Results

## Hardware
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM_120, 96GB GDDR7, 84 SMs)
- CUDA: 13.1 (nvcc), PyTorch 2.9.1+cu130
- Driver: 580.95.05

## Kernel Implementation
- **NVFP4**: Optimized shared-memory GEMM kernel (`kGemmNVFP4_smem`) with:
  - Cooperative shared memory tiling (32x128 block tile, 8 warps)
  - Register-based pipelining (load/compute overlap)
  - Auto split-K for small-batch shapes (fills GPU when M is small)
  - Vectorized uint32/uint4 loads for FP4 data
  - `mma.sync.aligned.block_scale` PTX instruction (m16n8k64)
- **FP16**: cuBLAS via `torch.matmul` (highly optimized baseline)

## Results (Optimized Kernel)

| Shape | NVFP4 (ms) | FP16 (ms) | Speedup | NVFP4 TFLOPS | FP16 TFLOPS |
|-------|-----------|----------|---------|-------------|------------|
| 128x128x128 | 0.006 | 0.005 | 0.84x | 0.7T | 0.9T |
| 256x256x256 | 0.006 | 0.005 | 0.81x | 5.6T | 6.9T |
| 1024x1024x1024 | 0.012 | 0.010 | 0.84x | 174.6T | 209.1T |
| 2048x2048x2048 | 0.084 | 0.053 | 0.63x | 204.3T | 322.6T |
| 4096x4096x4096 | 0.573 | 0.382 | 0.67x | 239.7T | 359.5T |
| **1x4096x4096** | **0.008** | **0.010** | **1.25x** | 4.1T | 3.3T |
| **8x4096x4096** | **0.010** | **0.011** | **1.05x** | 26.2T | 24.9T |
| **32x4096x4096** | **0.012** | **0.012** | **1.01x** | 87.8T | 86.9T |
| 128x4096x4096 | 0.025 | 0.019 | 0.75x | 174.6T | 232.7T |
| **32x4096x11008** | **0.018** | **0.023** | **1.23x** | 156.5T | 127.7T |
| 128x4096x11008 | 0.051 | 0.041 | 0.80x | 225.7T | 281.5T |

**Bold** rows indicate shapes where NVFP4 meets or exceeds cuBLAS FP16 performance.

## Key Findings

### LLM Inference Performance (bs=1-32)
For typical LLM inference shapes (small batch, large hidden dimensions), the NVFP4
kernel achieves **1.0-1.25x speedup** over FP16 cuBLAS. This is the target use case.

Split-K parallelization is critical for small-batch shapes: with M=1-32 and N=4096,
there are only 32 thread blocks for 84 SMs. Split-K divides the K dimension across
multiple blocks, improving GPU occupancy from ~0.4 to ~4 blocks/SM.

### Large Matrix Performance
For large square matrices (2K-4K), the kernel reaches 63-67% of cuBLAS FP16 performance.
The bottleneck is L1 cache throughput (74% utilization per NCU profiling). Further
optimization with cp.async double buffering could close this gap.

### Memory Savings
| Weight Shape | FP16 | NVFP4 | Compression |
|-------------|------|-------|-------------|
| 4096x4096 | 32.0 MB | 9.0 MB | 3.6x |
| 4096x11008 | 86.0 MB | 24.1 MB | 3.6x |

## Optimization History

| Version | 4Kx4K TFLOPS | 1x4Kx4K Speedup | Description |
|---------|-------------|-----------------|-------------|
| v1 (simple) | 18 | 0.11x | Correctness-first, per-nibble loads |
| v2 (vectorized) | 111 | 0.43x | uint32/uint4 bulk loads |
| v3 (smem) | 225 | 0.43x | Shared memory tiling |
| v4 (pipeline) | 239 | 0.43x | Register-based load/compute pipeline |
| v5 (split-K) | 240 | **1.25x** | Auto split-K for small M |

## NCU Profiling (4096x4096x4096)

| Metric | v1 (simple) | v3 (smem) | v5 (final) |
|--------|------------|-----------|------------|
| L1 Throughput | 40% | 74% | ~74% |
| SM Throughput | 10% | 39% | ~39% |
| Active Warps | 8.0 | 30.3 | ~30 |
| DRAM Throughput | 3.6% | 2.9% | ~3% |

The L1 cache is the primary bottleneck for large matrices. The kernel achieves
good SM occupancy (30 active warps, near-maximum for 4 blocks/SM × 8 warps/block).

## Correctness
All GEMM outputs match the dequantize→torch.matmul reference with 0.000000 relative
error (identical quantized data, same FP32 accumulation). 31 tests pass including
non-aligned shapes, tall/skinny LLM shapes, and NVFP4 output epilogue tests.
