#include <float.h>
#include <basicOps.cuh>

#include <mma.h>

#define COPY_BLOCK_SIZE 16

#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)

#define PI 3.1415926535897932f

#ifndef clusterKernels
#define clusterKernels

template<int operation> __global__ void kElementWise(const float *A, const float *B, float *out, const float scalar, int size);

#endif


