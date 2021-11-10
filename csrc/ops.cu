// Copyright (c) Facebook, Inc. and its affiliates. 
//   
// This source code is licensed under the MIT license found in the 
// LICENSE file in the root directory of this source tree.

#include <ops.cuh>
#include <kernels.cuh>
#include <cub/device/device_scan.cuh>
#include <limits>
#include <BinSearch.h>


using namespace BinSearch;
using std::cout;
using std::endl;

#define BLOCK_SIZE 4096

struct quantize_block_args
{
  BinAlgo<Scalar, float, Direct2> *bin_searcher;
  float *code;
  float *A;
  float *absmax;
  unsigned char *out;
  int block_end;
  int block_idx;
  int threadidx;
};

void *quantize_block(void *arguments)
{
  // 1. find absmax in block
  // 2. divide input value by absmax to normalize into [-1.0, 1.0]
  // 3. do binary search to find the closest value
  // 4. check minimal distance
  // 5. store index

  struct quantize_block_args *args = (quantize_block_args*)arguments;

  // 1. find absmax in block
  float absmax_block = -FLT_MAX;
  for (int i = args->block_idx; i < args->block_end; i++)
    absmax_block = fmax(absmax_block, fabs(args->A[i]));

  args->absmax[args->block_idx/BLOCK_SIZE] = absmax_block;

  for (int i = args->block_idx; i < args->block_end; i++)
  {
    // 2. divide input value by absmax to normalize into [-1.0, 1.0]
    // 3. do binary search to find the closest value
    float normed_value = args->A[i]/absmax_block;
    int idx = args->bin_searcher->scalar(normed_value);

    // 4. check minimal distance
    // The binary search returns always the value to the left, which might not be the closest value
    if(idx < 255)
    {
      float dist_left = fabs(normed_value-(args->code[idx]));
      float dist_right = fabs(normed_value-(args->code[idx+1]));
      if(dist_right < dist_left){ idx+=1; }
    }

    // 5. store index
    args->out[i] = (unsigned char)idx;
  }

  return NULL;
}

void quantize_cpu(float *code, float *A, float *absmax, unsigned char *out, int n)
{

  // the default code is has range [-0.993, 1.0] which can cause an error in the binary search algorithm used below
  code[0] = -1.0f; 

  int num_blocks = n/BLOCK_SIZE;
  num_blocks += n % BLOCK_SIZE == 0 ? 0 : 1;

  pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t)*num_blocks);
  struct quantize_block_args **args = (quantize_block_args**)malloc(num_blocks*sizeof(quantize_block_args*));

  for(int i = 0; i < num_blocks; i++)
    args[i] = (quantize_block_args*)malloc(sizeof(quantize_block_args));

  const uint32 elements_code = 256;
  BinAlgo<Scalar, float, Direct2> bin_searcher(code, elements_code);

  for(int block_idx = 0; block_idx < n; block_idx+=BLOCK_SIZE)
  {
    int valid_items = n-block_idx >= BLOCK_SIZE ? BLOCK_SIZE : n - block_idx;
    int block_end = block_idx + valid_items;

    struct quantize_block_args *arg = args[block_idx/BLOCK_SIZE];
    arg->bin_searcher = &bin_searcher;
    arg->code = code;
    arg->A = A;
    arg->absmax = absmax;
    arg->out = out;
    arg->block_end = block_end;
    arg->block_idx = block_idx;
    arg->threadidx = block_idx/BLOCK_SIZE;
 
    pthread_create(&threads[block_idx/BLOCK_SIZE], NULL, &quantize_block, (void *)arg);
  }

  for(int i = 0; i < num_blocks; i++)
    int err = pthread_join(threads[i], NULL);

  free(threads);
  for(int i = 0; i < num_blocks; i++)
    free(args[i]);
  free(args);
}


void dequantize_cpu(float *code, unsigned char *A, float *absmax, float *out, int n)
{
  for(int block_idx = 0; block_idx < n; block_idx+=BLOCK_SIZE)
  {
    int valid_items = n-block_idx >= BLOCK_SIZE ? BLOCK_SIZE : n - block_idx;
    int block_end = block_idx + valid_items;
    for (int i = block_idx; i < block_end; i++)
      out[i] = code[A[i]]*absmax[block_idx/BLOCK_SIZE];
  }
}

void histogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, int maxidx1, int n)
{
  int threads = 512;
  int blocks = n/threads;
  blocks = n % threads == 0 ? blocks : blocks + 1;
  kHistogramScatterAdd2D<<<blocks, 512>>>(histogram, index1, index2, src, maxidx1, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
	CUDA_CHECK_RETURN(cudaMemset(code, 0, 256*sizeof(float)));
  kEstimateQuantiles<T><<<blocks, 512>>>(A, code, offset, std::numeric_limits<T>::max(), n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void quantize(float *code, float *A, unsigned char *out, int n)
{
  int blocks = n/1024;
  blocks = n % 1024 == 0 ? blocks : blocks + 1;
  kQuantize<<<blocks, 1024>>>(code, A, out, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void dequantize(float *code, unsigned char *A, float *out, int n)
{
  int blocks = n/1024;
  blocks = n % 1024 == 0 ? blocks : blocks + 1;
  kDequantize<<<blocks, 1024>>>(code, A, out, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T, int STOCHASTIC> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float *rand, int rand_offset, const int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
  kQuantizeBlockwise<T, 4096, 4, STOCHASTIC><<<blocks, 1024>>>(code, A, absmax, out, rand, rand_offset, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template<typename T> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize, const int n)
{
  int blocks = n/blocksize;
  blocks = n % blocksize == 0 ? blocks : blocks + 1;
  if(blocksize == 4096)
    kDequantizeBlockwise<T, 4096, 1024, 4><<<blocks, 4096/4>>>(code, A, absmax, out, n);
  else if(blocksize == 2048)
    kDequantizeBlockwise<T, 2048, 512, 4><<<blocks, 2048/4>>>(code, A, absmax, out, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p, 
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, bool skip_zeros, const int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
	switch(OPTIMIZER)
	{
		case ADAM:
      if(max_unorm > 0.0f)
			{ 
				CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1*sizeof(float)));
        kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8><<<blocks, 512>>>(g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
      }
			kOptimizer32bit2State<T, OPTIMIZER><<<blocks, 1024>>>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
			break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:

      if(max_unorm > 0.0f)
			{ 
				CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1*sizeof(float)));
				kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8><<<blocks, 512>>>(g, p, state1, unorm, beta1, eps, weight_decay, step, lr, gnorm_scale, n);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
			}

			kOptimizer32bit1State<T, OPTIMIZER><<<blocks, 1024>>>(g, p, state1, unorm, max_unorm, param_norm, beta1, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
			break;
	}
}

template<typename T, int OPTIMIZER> void optimizerStatic8bit(T* p, T* g,
                unsigned char* state1, unsigned char* state2,
                float *unorm, float max_unorm, float param_norm,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;

  if(max_unorm > 0.0f){ CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1*sizeof(float))); }

	switch(OPTIMIZER)
	{
		case ADAM:
			CUDA_CHECK_RETURN(cudaMemset(new_max1, 0, 1*sizeof(float)));
			CUDA_CHECK_RETURN(cudaMemset(new_max2, 0, 1*sizeof(float)));
			kPreconditionOptimizerStatic8bit2State<T, OPTIMIZER><<<blocks, 256>>>(p, g, state1, state2, unorm, beta1, beta2, eps, step, quantiles1, quantiles2, max1, max2, new_max1, new_max2, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			kOptimizerStatic8bit2State<T, OPTIMIZER><<<blocks, 1024>>>(p, g, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr,
																														quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
			CUDA_CHECK_RETURN(cudaMemset(new_max1, 0, 1*sizeof(float)));
			kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER><<<blocks, 256>>>(p, g, state1, unorm, beta1, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			kOptimizerStatic8bit1State<T, OPTIMIZER><<<blocks, 1024>>>(p, g, state1, unorm, max_unorm, param_norm, beta1, eps, step, lr,
																														quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			break;
		default:
			break;
	}
}

#define BLOCKSIZE_2STATE 2048
#define NUM_2STATE 8
#define BLOCKSIZE_1STATE 2048
#define NUM_1STATE 8

template<typename T, int OPTIMIZER> void optimizerStatic8bitBlockwise(T* p, T* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr, 
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n)
{

	int blocks = 0;
	switch(OPTIMIZER)
	{
		case ADAM:
			blocks = n/BLOCKSIZE_2STATE;
			blocks = n % BLOCKSIZE_2STATE == 0 ? blocks : blocks + 1;
			kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE><<<blocks, BLOCKSIZE_2STATE/NUM_2STATE>>>(p, g, state1, state2, beta1, beta2, eps, step, lr,
																														quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		break;
		case MOMENTUM:
		case RMSPROP:
    case ADAGRAD:
			blocks = n/BLOCKSIZE_1STATE;
			blocks = n % BLOCKSIZE_1STATE == 0 ? blocks : blocks + 1;
			kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE><<<blocks, BLOCKSIZE_1STATE/NUM_1STATE>>>(p, g, state1, beta1, beta2, eps, step, lr,
																														quantiles1, absmax1, weight_decay, gnorm_scale, skip_zeros, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		break;
	}
}



template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n)
{
  int blocks = n/2048;
  blocks = n % 2048 == 0 ? blocks : blocks + 1;
	CUDA_CHECK_RETURN(cudaMemset(&gnorm_vec[step % 100], 0, 1*sizeof(float)));
  kPercentileClipping<T, 2048, 4><<<blocks, 512>>>(g, gnorm_vec, step, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void estimateQuantiles(half *A, float *code, float offset, int n);
template void estimateQuantiles(float *A, float *code, float offset, int n);

template void quantizeBlockwise<half, 0>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, const int n);
template void quantizeBlockwise<float, 0>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, const int n);
template void quantizeBlockwise<half, 1>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, const int n);
template void quantizeBlockwise<float, 1>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, const int n);
template void dequantizeBlockwise<half>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<float>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);

#define MAKE_optimizer32bit(name, gtype) \
template void optimizer32bit<gtype, name>(gtype* g, gtype* p, \
                float* state1, float* state2, float* unorm, float max_unorm, float param_norm, \
                const float beta1, const float beta2, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

MAKE_optimizer32bit(ADAM, half)
MAKE_optimizer32bit(ADAM, float)
MAKE_optimizer32bit(MOMENTUM, half)
MAKE_optimizer32bit(MOMENTUM, float)
MAKE_optimizer32bit(RMSPROP, half)
MAKE_optimizer32bit(RMSPROP, float)
MAKE_optimizer32bit(ADAGRAD, half)
MAKE_optimizer32bit(ADAGRAD, float)

#define MAKE_optimizerStatic8bit(name, gtype) \
template void optimizerStatic8bit<gtype, name>(gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, \
                float *unorm, float max_unorm, float param_norm, \
                float beta1, float beta2, \
                float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, \
                const float gnorm_scale, int n); \

MAKE_optimizerStatic8bit(ADAM, half)
MAKE_optimizerStatic8bit(ADAM, float)
MAKE_optimizerStatic8bit(MOMENTUM, half)
MAKE_optimizerStatic8bit(MOMENTUM, float)
MAKE_optimizerStatic8bit(RMSPROP, half)
MAKE_optimizerStatic8bit(RMSPROP, float)

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name) \
template void optimizerStatic8bitBlockwise<gtype, optim_name>(gtype* p, gtype* g, \
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n); \

MAKE_optimizerStatic8bitBlockwise(half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(half, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(float, ADAGRAD);

template void percentileClipping(float * g, float *gnorm_vec, int step, const int n);
template void percentileClipping(half * g, float *gnorm_vec, int step, const int n);
