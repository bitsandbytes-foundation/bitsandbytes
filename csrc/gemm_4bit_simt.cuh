#pragma once
// Launcher declarations for the SIMT 4-bit GEMM kernel.
// T must be __nv_bfloat16, half, or float.
// Compiles for all architectures (sm60+, no tensor core requirement).
// absmax_8bit == nullptr selects the non-nested (standard) path.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
void launch_gemm_4bit_simt(
    const T* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, T* C, const T* bias, int M, int N, int K, int blocksize, int quant_type,
    cudaStream_t stream
);
