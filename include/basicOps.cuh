
#ifndef basicOps_H
#define basicOps_H

#include <stdio.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <stdio.h>

#define TILE_DIM (32)
#define BLOCK_ROWS (8)

#define RDM_NUMBERS_PER_THREAD (1024)
#define THREADS_PER_BLOCKS (512)
#define BLOCKS (4096)

#define DOT_BLOCKS (128)
#define TILE_SIZE (32)
#define DOT_REPS (4)

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)


#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)
#define NUM_RND_BURNIN                      300

/*
 * Defines for getting the values at the lower and upper 32 bits
 * of a 64-bit number.
 */
#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)


#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

typedef enum Operations_t
{
	kabs = 0,
	klog = 1,
	ksqrt = 2,
	kpow = 3,
	kadd = 4,
	ksub = 5,
	kdiv = 6,
	kmul = 7,
	klogistic = 8,
	klogistic_grad = 9,
	krectified = 10,
	krectified_grad = 11,
	keq = 12,
	klt = 13,
	kgt = 14,
	kge = 15,
	kle = 16,
	kne = 17,
	ksquared_diff = 18,



	kvadd = 19,
	kvsub = 20,
	ktmatrix = 21,


	ksmul = 22,
	ksgt = 23,

	kdropout = 24,
	kcopy = 25,
	kssub = 26,
	kELU = 27,
	kELU_grad = 28,
	kmod = 29,
	ktanh = 30,
	ktanh_grad = 31,
	kexp = 32,

	kdiff = 33,
	karange = 34,
	krowcopy = 35,
	kmarkrow = 36,

} Operations_t;

template <int action> void elementWise(float *A, float *out, int n, float scalar);


#endif







