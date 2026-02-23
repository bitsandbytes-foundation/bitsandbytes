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

## CUTLASS GEMM Results (via QuTLASS)

After replacing the hand-written kernel with CUTLASS (QuTLASS-derived), compiled into
bitsandbytes with zero runtime dependency:

### Standalone QuTLASS GEMM (GEMM-only, no Python overhead)

| Shape | cuBLAS BF16 (ms) | cuBLAS TFLOPS | CUTLASS (ms) | CUTLASS TFLOPS | Speedup |
|-------|-------------------|---------------|--------------|----------------|---------|
| 1×4096×4096 | 0.017 | 2.0 | 0.026 | 1.3 | 0.64x |
| 8×4096×4096 | 0.018 | 15.1 | 0.025 | 10.8 | 0.72x |
| 32×4096×4096 | 0.018 | 60.8 | 0.025 | 43.3 | 0.71x |
| 128×4096×4096 | 0.024 | 177.5 | 0.025 | 174.5 | 0.98x |
| **4096×4096×4096** | **0.342** | **402.0** | **0.108** | **1276.0** | **3.17x** |
| 32×4096×11008 | 0.028 | 103.5 | 0.045 | 64.0 | 0.62x |
| 128×4096×11008 | 0.046 | 249.1 | 0.045 | 254.9 | 1.02x |

### Integrated bitsandbytes GEMM (includes Python dispatch overhead)

| Shape | cuBLAS BF16 (ms) | cuBLAS TFLOPS | BNB NVFP4 (ms) | BNB TFLOPS | Speedup |
|-------|-------------------|---------------|----------------|------------|---------|
| 1×4096×4096 | 0.016 | 2.1 | 0.038 | 0.9 | 0.43x |
| 8×4096×4096 | 0.018 | 15.2 | 0.040 | 6.7 | 0.44x |
| 32×4096×4096 | 0.018 | 60.6 | 0.039 | 27.4 | 0.45x |
| 128×4096×4096 | 0.023 | 187.5 | 0.039 | 109.8 | 0.59x |
| **4096×4096×4096** | **0.339** | **405.3** | **0.147** | **937.8** | **2.31x** |
| 32×4096×11008 | 0.028 | 103.4 | 0.060 | 48.4 | 0.47x |
| 128×4096×11008 | 0.046 | 250.3 | 0.060 | 193.6 | 0.77x |

### Key Findings — CUTLASS vs Hand-written

- **Large M (4096)**: CUTLASS achieves **1276 TFLOPS** (3.17x cuBLAS), **5.3x** faster than
  the hand-written kernel (240 TFLOPS). CUTLASS uses SM_120 wgmma instructions with
  CUTLASS's sophisticated pipeline scheduling.
- **Small M (1-32)**: CUTLASS has higher launch overhead (~0.025ms floor) vs the hand-written
  kernel (~0.008-0.012ms). For small shapes where compute is negligible, the hand-written
  kernel with split-K still has an advantage.
- **Medium M (128)**: Roughly parity between CUTLASS and cuBLAS.
- **Memory compression**: Unchanged — 3.6x compression vs FP16 weights.

### CUTLASS Configuration (SM_120)

- Tile shapes: 128×128×128 (M<512), 256×128×128 (M≥512)
- Cluster shape: 1×1×1 (no multi-SM clusters on consumer Blackwell)
- Data type: `nv_float4_t<float_e2m1_t>`, scale type: `float_ue4m3_t`
- Output: BF16 with FP32 accumulator, alpha epilogue fusion

## Fused Quantize Results (QuTLASS CUTLASS-based Quantization)

Replaces the hand-written `kQuantizeNVFP4` with a CUTLASS SM_80 GEMM that formulates
quantization as a matrix multiply (each 16-element group becomes a GEMM row). The key
advantage: **Hadamard rotation is free** — applied via the B matrix in the GEMM with
zero additional compute cost.

### Raw Kernel Comparison (no Python overhead)

| Shape | Old plain (ms) | Old+Rotation (ms) | CUTLASS+Rotation (ms) | Rotation overhead |
|-------|---------------|-------------------|----------------------|-------------------|
| 1×4096 | 0.003 | 0.003 | 0.004 | Old: 0%, CUTLASS: 0% |
| 128×4096 | 0.003 | 0.004 | 0.004 | Old: 52%, CUTLASS: 0% |
| 4096×4096 | 0.023 | 0.043 | 0.039 | Old: 85%, CUTLASS: 0% |
| 128×11008 | 0.004 | 0.006 | 0.006 | Old: 49%, CUTLASS: 0% |

**Key finding**: The old hand-written kernel is ~1.5x faster for plain quantize (no rotation).
But for quantize with Hadamard rotation (`rotate=True`, the new default):
- Small shapes (M ≤ 32): CUTLASS 0.004ms vs old fused 0.003ms — old kernel wins
- Large shapes (M = 4096): CUTLASS 0.039ms vs old fused 0.043ms — CUTLASS wins (1.1x)
- The main value is rotation at zero cost, not raw quantize speed

### CUTLASS Fused Quantize: AbsMax vs Quest (Rotation)

| Shape | AbsMax (ms) | Quest/Rotation (ms) | Rotation overhead |
|-------|------------|---------------------|-------------------|
| 1×4096 | 0.004 | 0.004 | 0% |
| 128×4096 | 0.004 | 0.004 | 0% |
| 4096×4096 | 0.037 | 0.037 | 0% |

Hadamard rotation adds **zero overhead** with the CUTLASS approach — the rotation matrix
is applied as the B operand in the GEMM, which is already being executed.

### End-to-End Pipeline (Quantize A + GEMM, B pre-quantized)

| Shape | Old kernel (ms) | CUTLASS (ms) | cuBLAS FP16 (ms) | vs cuBLAS |
|-------|----------------|-------------|------------------|-----------|
| 1×4096×4096 | 0.082 | 0.095 | 0.012 | 0.13x |
| 128×4096×4096 | 0.087 | 0.097 | 0.019 | 0.19x |
| 4096×4096×4096 | 0.268 | 0.297 | 0.335 | 1.13x |

For large M (≥ 4096), the NVFP4 pipeline (quantize + GEMM) exceeds cuBLAS FP16.
The quantize overhead is significant for small M — a fused quantize-into-GEMM epilogue
(future work) would eliminate this per-layer cost.

## Correctness
All GEMM outputs match the dequantize→torch.matmul reference with 0.000000 relative
error (identical quantized data, same FP32 accumulation). 59 tests pass including
non-aligned shapes, tall/skinny LLM shapes, large-batch shapes (up to 4096x4096x4096),
scale reordering round-trip tests, NVFP4 output epilogue tests, fused quantize tests
(padding, rotation, fallback, dtype conversion), and end-to-end pipeline tests.
