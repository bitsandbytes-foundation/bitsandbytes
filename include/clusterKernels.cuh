#include <float.h>
#include <basicOps.cuh>

#ifndef clusterKernels
#define clusterKernels

template<int operation> __global__ void kElementWise(const float *A, const float *B, float *out, const float scalar, int size);

template<typename T>__global__ void kEstimateQuantiles(T *__restrict__ const A, float *code, const float offset, const T max_val, const int n);

__global__ void kQuantize(float * code, float * __restrict__ const A, unsigned char *out, const int n);
__global__ void kDequantize(float *code, unsigned char *A, float *out, const int n);

template<typename T, int OPTIMIZER>
__global__ void kOptimizer_32bit_2State(T* g, T* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const bool is_sparse, const int n);

#endif


