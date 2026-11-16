#pragma once
// Launcher declarations for the AMD WMMA 4-bit GEMM kernel.
// T must be bnb_bfloat16, half or float (fp32 is emulated with split bf16 operands). Compiled on ROCm only; the kernel
// body is only emitted for RDNA3 (gfx110x), RDNA3.5 (gfx115x) and RDNA4 (gfx120x) targets. Callers pass
// gemm_4bit_wmma_arch() and must check K % 64 == 0 and blocksize >= 32. absmax_8bit == nullptr selects the non-nested
// (standard) path.

#include <cstdint>

#include "compat.cuh" // CUDA/HIP runtime headers + bnb_stream_t

// WMMA tuning target of the current device: 11 (gfx1100-gfx1102), 113 (gfx1103), 115 (gfx115x), 12 (gfx120x), or 0
// if the kernel is not enabled for it. Tile rules per target: gemm_4bit_wmma_rules.cuh.
int gemm_4bit_wmma_arch();

// Returns false (nothing launched) if `arch` is not a target this build has WMMA kernels for.
template <typename T>
bool launch_gemm_4bit_wmma(
    int arch, const T* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, T* C, const T* bias, int M, int N, int K, int blocksize, int quant_type,
    bnb_stream_t stream
);
