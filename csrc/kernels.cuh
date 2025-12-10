// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <float.h>
#include <ops.cuh>

#ifndef kernels
#define kernels

__global__ void kQuantize(float* code, float* __restrict__ const A, unsigned char* out, const int n);
__global__ void kDequantize(float* code, unsigned char* A, float* out, const int n);

template <typename T, int BLOCK_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
__global__ void
    kDequantizeBlockwise(float* code, unsigned char* A, float* absmax, T* out, const int blocksize, const int n);

template <typename T, int OPTIMIZER>
__global__ void kPreconditionOptimizerStatic8bit1State(
    T* p, T* __restrict__ const g, unsigned char* __restrict__ const state1, float* unorm, const float beta1,
    const float beta2, const float eps, const int step, float* __restrict__ const quantiles1, float* max1,
    float* new_max1, const float weight_decay, const float gnorm_scale, const int n
);

template <typename T, int OPTIMIZER>
__global__ void kOptimizerStatic8bit1State(
    T* p, T* const g, unsigned char* state1, const float* unorm, const float max_unorm, const float param_norm,
    const float beta1, const float beta2, const float eps, const int step, const float lr,
    float* __restrict__ const quantiles1, float* max1, float* new_max1, float weight_decay, const float gnorm_scale,
    const int n
);

template <typename T, int OPTIMIZER>
__global__ void kPreconditionOptimizerStatic8bit2State(
    T* p, T* __restrict__ const g, unsigned char* __restrict__ const state1, unsigned char* __restrict__ const state2,
    float* unorm, const float beta1, const float beta2, const float eps, const int step,
    float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, float* max1, float* max2,
    float* new_max1, float* new_max2, const float gnorm_scale, const int n
);

template <typename T, int OPTIMIZER>
__global__ void kOptimizerStatic8bit2State(
    T* p, T* const g, unsigned char* state1, unsigned char* state2, const float* unorm, const float max_unorm,
    const float param_norm, const float beta1, const float beta2, const float eps, const int step, const float lr,
    float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, float* max1, float* max2,
    float* new_max1, float* new_max2, float weight_decay, const float gnorm_scale, const int n
);

template <typename T, int BLOCK_SIZE, int NUM_VALS>
__global__ void kPercentileClipping(T* __restrict__ g, float* gnorm_vec, int step, const int n);

template <typename T, int SPMM_ITEMS, int BITS>
__global__ void kspmm_coo_very_sparse_naive(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, T* B, half* out,
    float* __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB
);

template <typename T, int THREADS, int BITS>
__global__ void kgemm_4bit_inference_naive(
    int M, int N, int K, T* __restrict__ const A, unsigned char* B, float* absmax, const float* datatype, T* out,
    int lda, int ldb, int ldc, int blocksize
);

template <typename T, int FUNC> __global__ void kfunc(T* A, T* B, T value, long n);

#endif
