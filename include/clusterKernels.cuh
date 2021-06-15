#include <float.h>
#include <basicOps.cuh>

#ifndef clusterKernels
#define clusterKernels

template<typename T>__global__ void kEstimateQuantiles(T *__restrict__ const A, float *code, const float offset, const T max_val, const int n);

__global__ void kQuantize(float * code, float * __restrict__ const A, unsigned char *out, const int n);
__global__ void kDequantize(float *code, unsigned char *A, float *out, const int n);

template<typename T, int OPTIMIZER>
__global__ void kOptimizer_32bit_2State(T* g, T* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const bool is_sparse, const int n);


template<typename T, int OPTIMIZER>
__global__ void
kPreconditionOptimizerStatic8bit2State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1, unsigned char* __restrict__ const state2,
                const float beta1, const float beta2,
                const float eps, const int step, 
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                const int n);


template<typename T, int OPTIMIZER>
__global__ void
kOptimizerStatic8bit2State(T* p, T* const g, unsigned char* state1, unsigned char* state2,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr, 
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                const float gnorm_scale, 
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay, const int n);


#endif


