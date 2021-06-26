#include <basicOps.cuh>
#include <clusterKernels.cuh>
#include <cuda_profiler_api.h>
#include <cub/device/device_scan.cuh>
#include <limits>

using std::cout;
using std::endl;

template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
  kEstimateQuantiles<T><<<blocks, 1024>>>(A, code, offset, std::numeric_limits<T>::max(), n);
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

template<typename T, int OPTIMIZER> void optimizer_32bit(T* g, T* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const bool is_sparse, const float gnorm_scale, const int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
	switch(OPTIMIZER)
	{
		case ADAM:
			kOptimizer_32bit_2State<T, OPTIMIZER><<<blocks, 1024>>>(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n);
			break;
		case MOMENTUM:
			kOptimizer_32bit_1State<T, OPTIMIZER><<<blocks, 1024>>>(g, p, state1, beta1, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n);
			break;
		default:
			break;
	}
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template<typename T, int OPTIMIZER> void optimizerStatic8bit(T* p, T* g,
                unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
	switch(OPTIMIZER)
	{
		case ADAM:
		kPreconditionOptimizerStatic8bit2State<T, OPTIMIZER><<<blocks, 256>>>(p, g, state1, state2, beta1, beta2, eps, step, quantiles1, quantiles2, max1, max2, new_max1, new_max2, gnorm_scale, n);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		kOptimizerStatic8bit2State<T, OPTIMIZER><<<blocks, 1024>>>(p, g, state1, state2, beta1, beta2, eps, step, lr,
																													quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		break;
		case MOMENTUM:
			kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER><<<blocks, 256>>>(p, g, state1, beta1, eps, step, quantiles1, max1, new_max1, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			kOptimizerStatic8bit1State<T, OPTIMIZER><<<blocks, 1024>>>(p, g, state1, beta1, eps, step, lr,
																														quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			break;
		default:
			break;
	}
}

template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n)
{
  int blocks = n/2048;
  blocks = n % 2048 == 0 ? blocks : blocks + 1;
  kPercentileClipping<T, 2048, 4><<<blocks, 512>>>(g, gnorm_vec, step, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void estimateQuantiles(half *A, float *code, float offset, int n);
template void estimateQuantiles(float *A, float *code, float offset, int n);

template void optimizer_32bit<half, ADAM>(half* g, half* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const bool is_sparse, const float gnorm_scale, const int n);
template void optimizer_32bit<float, ADAM>(float* g, float* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const bool is_sparse, const float gnorm_scale, const int n);

template void optimizer_32bit<half, MOMENTUM>(half* g, half* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const bool is_sparse, const float gnorm_scale, const int n);
template void optimizer_32bit<float, MOMENTUM>(float* g, float* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const bool is_sparse, const float gnorm_scale, const int n);

template void optimizerStatic8bit<half, ADAM>(half* p, half* g, unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n);

template void optimizerStatic8bit<float, ADAM>(float* p, float* g, unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n);

template void optimizerStatic8bit<half, MOMENTUM>(half* p, half* g, unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n);

template void optimizerStatic8bit<float, MOMENTUM>(float* p, float* g, unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n);

template void percentileClipping(float * g, float *gnorm_vec, int step, const int n);
template void percentileClipping(half * g, float *gnorm_vec, int step, const int n);
