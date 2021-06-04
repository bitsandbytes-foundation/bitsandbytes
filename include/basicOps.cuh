
#ifndef basicOps_H
#define basicOps_H

#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

#define THREADS_PER_BLOCKS (512)

typedef enum Operations_t
{
	ksmul = 0,
} Operations_t;

template <int action> void elementWise(float *A, float *out, int n, float scalar);
void estimateQuantiles(float *A, float *code, float offset, int n);


#endif







