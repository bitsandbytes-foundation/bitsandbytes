#pragma once
// Launcher declarations for the sm80+ MMA 4-bit GEMM kernel.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gemm_4bit_common.cuh"

template <typename T>
void launch_gemm_4bit_sm80_m16n8k16(
    const T* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, T* C, const T* bias, int M, int N, int K, int blocksize, int quant_type, GpuProps gpu,
    cudaStream_t stream
);
