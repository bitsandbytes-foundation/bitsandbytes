// Copyright (c) Facebook, Inc. and its affiliates. 
//   
// This source code is licensed under the MIT license found in the 
// LICENSE file in the root directory of this source tree.

#include <float.h>
#include <ops.cuh>

#ifndef kernels
#define kernels

template<typename T>__global__ void kEstimateQuantiles(T *__restrict__ const A, float *code, const float offset, const T max_val, const int n);

__global__ void kQuantize(float * code, float * __restrict__ const A, unsigned char *out, const int n);
__global__ void kDequantize(float *code, unsigned char *A, float *out, const int n);

template<typename T, int BLOCK_SIZE, int NUM_PER_TH, int STOCHASTIC> __global__ void kQuantizeBlockwise(float * code, T * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n);
template<typename T, int BLOCK_SIZE, int THREADS, int NUM_PER_TH> __global__ void kDequantizeBlockwise(float *code, unsigned char * __restrict__ const A, float * __restrict__ const absmax, T *out, const int n);

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
__global__ void kPreconditionOptimizer32bit2State(T* g, T* p, 
                float* state1, float* state2, float *unorm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const int n);

template<typename T, int OPTIMIZER>
__global__ void kOptimizer32bit2State(T* g, T* p, 
                float* state1, float* state2, float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
__global__ void kPreconditionOptimizer32bit1State(T* g, T* p, 
                float* state1, float *unorm,
                const float beta1, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const int n);

template<typename T, int OPTIMIZER>
__global__ void kOptimizer32bit1State(T* g, T* p, 
                float* state1,  float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

template<typename T, int OPTIMIZER>
__global__ void
kPreconditionOptimizerStatic8bit1State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1, 
                float *unorm,
                const float beta1, 
                const float eps, const int step, 
                float* __restrict__ const quantiles1, 
                float* max1, float* new_max1, 
                const float weight_decay,
                const float gnorm_scale, const int n);


template<typename T, int OPTIMIZER>
__global__ void
kOptimizerStatic8bit1State(T* p, T* const g, unsigned char* state1, 
                const float *unorm, const float max_unorm, const float param_norm,
                const float beta1, 
                const float eps, const int step, const float lr, 
                float* __restrict__ const quantiles1, 
                float* max1, float* new_max1, 
                float weight_decay, const float gnorm_scale, const int n);



template<typename T, int OPTIMIZER>
__global__ void
kPreconditionOptimizerStatic8bit2State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1, unsigned char* __restrict__ const state2,
                float *unorm,
                const float beta1, const float beta2,
                const float eps, const int step, 
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                const float gnorm_scale, const int n);


template<typename T, int OPTIMIZER>
__global__ void
kOptimizerStatic8bit2State(T* p, T* const g, unsigned char* state1, unsigned char* state2,
                const float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr, 
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay, const float gnorm_scale, const int n);

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH> __global__ void kOptimizerStatic8bit2StateBlockwise(
		T* p, T* __restrict__ const g, unsigned char* state1, unsigned char* state2,
                const float beta1, const float beta2, const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, const bool skip_zeros, const int n);

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH> __global__ void kOptimizerStatic8bit1StateBlockwise(
		T* p, T* __restrict__ const g, unsigned char* state1,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1,
                float* absmax1,
                float weight_decay,
                const float gnorm_scale, const bool skip_zeros, const int n);


template<typename T, int BLOCK_SIZE, int NUM_VALS> __global__ void kPercentileClipping(T * __restrict__ g, float *gnorm_vec, int step, const int n);

__global__ void kHistogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, const int maxidx1, const int n);

#endif


