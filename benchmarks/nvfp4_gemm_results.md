# NVFP4 GEMM Benchmark Results

## Hardware
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM_120, 96GB GDDR7)
- CUDA: 13.1 (nvcc), PyTorch 2.9.1+cu130
- Driver: 580.95.05

## Kernel Implementation
- **NVFP4**: Correctness-first kernel (`kGemmNVFP4_simple`), one warp per m16n8 output tile,
  global memory loads, no shared memory, no software pipelining.
  Uses `mma.sync.aligned.block_scale` PTX instruction.
- **FP16**: cuBLAS via `torch.matmul` (highly optimized baseline)

## Results

| Shape | NVFP4 (ms) | FP16 (ms) | Speedup | NVFP4 TFLOPS | FP16 TFLOPS |
|-------|-----------|----------|---------|-------------|------------|
| 128x128x128 | 0.012 | 0.005 | 0.43x | 0.4T | 0.8T |
| 256x256x256 | 0.012 | 0.005 | 0.43x | 2.9T | 6.8T |
| 512x512x512 | 0.023 | 0.005 | 0.22x | 11.9T | 53.1T |
| 1024x1024x1024 | 0.124 | 0.010 | 0.08x | 17.4T | 208.4T |
| 2048x2048x2048 | 0.965 | 0.053 | 0.06x | 17.8T | 322.7T |
| 4096x4096x4096 | 7.571 | 0.347 | 0.05x | 18.2T | 396.5T |
| 1x4096x4096 | 0.092 | 0.010 | 0.11x | 5.8T | 3.3T |
| 8x4096x4096 | 0.090 | 0.010 | 0.11x | 6.0T | 25.9T |
| 32x4096x4096 | 0.111 | 0.012 | 0.11x | 9.7T | 86.9T |
| 128x4096x4096 | 0.267 | 0.019 | 0.07x | 16.1T | 231.6T |
| 32x4096x11008 | 0.260 | 0.023 | 0.09x | 11.1T | 127.0T |
| 128x4096x11008 | 0.621 | 0.041 | 0.07x | 18.6T | 280.6T |

## Memory Savings

| Weight Shape | FP16 | NVFP4 | Compression |
|-------------|------|-------|-------------|
| 4096x4096 | 32.0 MB | 9.0 MB | 3.6x |
| 4096x11008 | 86.0 MB | 24.1 MB | 3.6x |

## Analysis

The NVFP4 GEMM kernel peaks at ~18 TFLOPS, while cuBLAS FP16 reaches ~400 TFLOPS on
the RTX PRO 6000. The current kernel is **~20x slower** than cuBLAS at large matrix sizes.

### Why the NVFP4 kernel is slow

This is a **correctness-first implementation** with no performance optimization:
1. **Global memory loads per-element**: Each thread loads individual nibbles from global memory
   with manual bit manipulation (shifts and masks). No coalesced loads.
2. **No shared memory**: Data is loaded directly from global memory into registers.
   A tiled kernel would stage data in shared memory for reuse.
3. **No software pipelining**: K-dimension loop has no overlap between compute and memory.
4. **One warp per m16n8 tile**: Poor utilization of the SM's resources. A proper kernel
   would use multiple warps per threadblock with a larger tile (128x128x128).
5. **Per-element packing**: The nibble extraction loop is serial (8 iterations per register).

### Performance optimization path

To close the gap with cuBLAS FP16, the kernel would need:
1. Shared memory tiling (128x128x128 threadblock tile)
2. Coalesced global → shared memory loads (cp.async or vectorized loads)
3. 2-3 stage software pipelining for the K loop
4. Multiple warps per threadblock (e.g., 4 warps computing 128x128 output)
5. Vectorized nibble packing (load uint32/uint64 instead of byte-by-byte)

The theoretical speedup of NVFP4 over FP16 on Blackwell is ~2x (double the FLOPs per
cycle). Achieving this requires a kernel within ~50% of cuBLAS's FP16 efficiency.

### Current value

Despite the performance gap, the implementation provides:
- **3.6x memory savings**: Enables larger models in GPU memory
- **Correct GEMM output**: Verified against torch.matmul on dequantized inputs
  with 0.000000 relative error (same quantized data, different only in FP32 rounding)
- **Full Python API**: quantize/dequantize/GEMM/LinearNVFP4 all working end-to-end
- **NVFP4 output epilogue**: GEMM → quantize chain for layer chaining

## LinearNVFP4 End-to-End Benchmarks

LinearNVFP4 includes activation quantization overhead on top of the GEMM kernel.

| Config | NVFP4 (ms) | FP16 (ms) | Speedup |
|--------|-----------|----------|---------|
| bs=1, 4096→4096 (proj) | 0.120 | 0.010 | 0.09x |
| bs=1, 4096→11008 (FFN) | 0.128 | 0.019 | 0.15x |
| bs=8, 4096→4096 (proj) | 0.128 | 0.010 | 0.08x |
| bs=8, 4096→11008 (FFN) | 0.143 | 0.019 | 0.13x |
| bs=32, 4096→4096 (proj) | 0.147 | 0.013 | 0.08x |
| bs=32, 4096→11008 (FFN) | 0.228 | 0.021 | 0.09x |
| bs=128, 4096→4096 (proj) | 0.315 | 0.019 | 0.06x |
| bs=128, 4096→11008 (FFN) | 0.710 | 0.041 | 0.06x |
