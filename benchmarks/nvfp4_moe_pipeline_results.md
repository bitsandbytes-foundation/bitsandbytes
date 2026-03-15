# NVFP4 MoE Pipeline vs BF16 Benchmark — B200

**GPU**: NVIDIA B200 (SM_100, compute capability 10.0)
**Date**: 2026-03-09
**Benchmark**: `bench_moe_pipeline.py` (100 iterations, 20 warmup)

## GLM-4.7 (352B MoE) Shapes

### gate_up (K=4096, N=13696)

| Config | BF16 (ms) | NVFP4 (ms) | Speedup | BF16 TFLOPS | NVFP4 TFLOPS |
|---|---|---|---|---|---|
| 8e × 8 tokens (64 total) | 0.501 | 0.267 | **1.87x** | 14.33 | 26.85 |
| 8e × 32 tokens (256 total) | 0.533 | 0.321 | **1.66x** | 53.90 | 89.58 |
| 8e × 64 tokens (512 total) | 0.555 | 0.383 | **1.45x** | 103.52 | 150.17 |
| 8e × 128 tokens (1024 total) | 0.597 | 0.514 | **1.16x** | 192.55 | 223.57 |
| 8e skewed (255 total) | 0.538 | 0.506 | **1.07x** | 53.14 | 56.60 |

### down (K=13696, N=4096)

| Config | BF16 (ms) | NVFP4 (ms) | Speedup | BF16 TFLOPS | NVFP4 TFLOPS |
|---|---|---|---|---|---|
| 8e × 8 tokens (64 total) | 0.546 | 0.254 | **2.15x** | 13.16 | 28.29 |
| 8e × 32 tokens (256 total) | 0.588 | 0.271 | **2.17x** | 48.85 | 106.01 |
| 8e × 64 tokens (512 total) | 0.578 | 0.296 | **1.95x** | 99.39 | 194.12 |
| 8e × 128 tokens (1024 total) | 0.599 | 0.356 | **1.68x** | 191.81 | 322.79 |
| 8e skewed (255 total) | 0.562 | 0.308 | **1.83x** | 50.87 | 93.01 |

## Summary

- NVFP4 pipeline wins every configuration (1.07x–2.17x over BF16)
- Down projection benefits most (large K, small N → memory-bandwidth-bound → FP4's 2x smaller footprint helps)
- Few tokens per expert shows largest speedup (pipeline overhead elimination dominates)
- Peak throughput: 322.8 TFLOPS (down proj, 128 tok/expert)

## Method

- **BF16 baseline**: Per-expert `torch.matmul` in a Python loop (represents the standard MoE dispatch pattern)
- **NVFP4 pipeline**: 6-kernel fused pipeline (abs_max → quantize_raw → scatter → scale_swizzle → batched_GEMM → gather), zero host-GPU sync in compute path
