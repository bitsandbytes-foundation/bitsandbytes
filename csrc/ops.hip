// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <ops.hiph>
#include <kernels.hiph>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>
#ifndef NO_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif
#include <limits>
#include <BinSearch.h>
#include <cassert>
#include <common.h>


using namespace BinSearch;
using std::cout;
using std::endl;

void histogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, int maxidx1, int n)
{
  int threads = 512;
  int num_blocks = n/threads;
  num_blocks = n % threads == 0 ? num_blocks : num_blocks + 1;
 hipLaunchKernelGGL(( kHistogramScatterAdd2D), dim3(num_blocks), dim3(512), 0, 0, histogram, index1, index2, src, maxidx1, n);
  HIP_CHECK_RETURN(hipPeekAtLastError());
}

template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
	HIP_CHECK_RETURN(hipMemset(code, 0, 256*sizeof(float)));
 hipLaunchKernelGGL(( kEstimateQuantiles<T>), dim3(num_blocks), dim3(512), 0, 0, A, code, offset, std::numeric_limits<T>::max(), n);
  HIP_CHECK_RETURN(hipPeekAtLastError());
}

void quantize(float *code, float *A, unsigned char *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
 hipLaunchKernelGGL(( kQuantize), dim3(num_blocks), dim3(1024), 0, 0, code, A, out, n);
  HIP_CHECK_RETURN(hipPeekAtLastError());
}

void dequantize(float *code, unsigned char *A, float *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
 hipLaunchKernelGGL(( kDequantize), dim3(num_blocks), dim3(1024), 0, 0, code, A, out, n);
  HIP_CHECK_RETURN(hipPeekAtLastError());
}

template <typename T, int STOCHASTIC> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float *rand, int rand_offset, int blocksize, const int n)
{
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  if(STOCHASTIC == 1)
    assert(blocksize == 4096);

  if(blocksize == 4096)
   hipLaunchKernelGGL(( kQuantizeBlockwise<T, 4096, 4, STOCHASTIC>), dim3(num_blocks), dim3(1024), 0, 0, code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 2048)
   hipLaunchKernelGGL(( kQuantizeBlockwise<T, 2048, 4, 0>), dim3(num_blocks), dim3(512), 0, 0, code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 1024)
   hipLaunchKernelGGL(( kQuantizeBlockwise<T, 1024, 4, 0>), dim3(num_blocks), dim3(256), 0, 0, code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 512)
   hipLaunchKernelGGL(( kQuantizeBlockwise<T, 512, 2, 0>), dim3(num_blocks), dim3(256), 0, 0, code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 256)
   hipLaunchKernelGGL(( kQuantizeBlockwise<T, 256, 2, 0>), dim3(num_blocks), dim3(128), 0, 0, code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 128)
   hipLaunchKernelGGL(( kQuantizeBlockwise<T, 128, 2, 0>), dim3(num_blocks), dim3(64), 0, 0, code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 64)
   hipLaunchKernelGGL(( kQuantizeBlockwise<T, 64, 1, 0>), dim3(num_blocks), dim3(64), 0, 0, code, A, absmax, out, rand, rand_offset, n);


  HIP_CHECK_RETURN(hipPeekAtLastError());
}

template<typename T> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize, const int n)
{
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  if(blocksize == 4096)
   hipLaunchKernelGGL(( kDequantizeBlockwise<T, 4096, 1024, 4>), dim3(num_blocks), dim3(4096/4), 0, 0, code, A, absmax, out, n);
  else if(blocksize == 2048)
   hipLaunchKernelGGL(( kDequantizeBlockwise<T, 2048, 512, 4>), dim3(num_blocks), dim3(2048/4), 0, 0, code, A, absmax, out, n);
  else if(blocksize == 1024)
   hipLaunchKernelGGL(( kDequantizeBlockwise<T, 1024, 256, 4>), dim3(num_blocks), dim3(1024/4), 0, 0, code, A, absmax, out, n);
  else if(blocksize == 512)
   hipLaunchKernelGGL(( kDequantizeBlockwise<T, 512, 256, 2>), dim3(num_blocks), dim3(512/2), 0, 0, code, A, absmax, out, n);
  else if(blocksize == 256)
   hipLaunchKernelGGL(( kDequantizeBlockwise<T, 256, 128, 2>), dim3(num_blocks), dim3(256/2), 0, 0, code, A, absmax, out, n);
  else if(blocksize == 128)
   hipLaunchKernelGGL(( kDequantizeBlockwise<T, 128, 64, 2>), dim3(num_blocks), dim3(128/2), 0, 0, code, A, absmax, out, n);
  else if(blocksize == 64)
   hipLaunchKernelGGL(( kDequantizeBlockwise<T, 64, 64, 1>), dim3(num_blocks), dim3(64/1), 0, 0, code, A, absmax, out, n);

  HIP_CHECK_RETURN(hipPeekAtLastError());
}

template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p,
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, bool skip_zeros, const int n)
{
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
	switch(OPTIMIZER)
	{
		case ADAM:
      if(max_unorm > 0.0f)
			{
				HIP_CHECK_RETURN(hipMemset(unorm, 0, 1*sizeof(float)));
       hipLaunchKernelGGL(( kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8>), dim3(num_blocks), dim3(512), 0, 0, g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
        HIP_CHECK_RETURN(hipPeekAtLastError());
      }
		hipLaunchKernelGGL((	kOptimizer32bit2State<T, OPTIMIZER>), dim3(num_blocks), dim3(1024), 0, 0, g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      HIP_CHECK_RETURN(hipPeekAtLastError());
			break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:

      if(max_unorm > 0.0f)
			{
				HIP_CHECK_RETURN(hipMemset(unorm, 0, 1*sizeof(float)));
			hipLaunchKernelGGL((	kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>), dim3(num_blocks), dim3(512), 0, 0, g, p, state1, unorm, beta1, eps, weight_decay, step, lr, gnorm_scale, n);
        HIP_CHECK_RETURN(hipPeekAtLastError());
			}

		hipLaunchKernelGGL((	kOptimizer32bit1State<T, OPTIMIZER>), dim3(num_blocks), dim3(1024), 0, 0, g, p, state1, unorm, max_unorm, param_norm, beta1, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      HIP_CHECK_RETURN(hipPeekAtLastError());
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
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;

  if(max_unorm > 0.0f){ HIP_CHECK_RETURN(hipMemset(unorm, 0, 1*sizeof(float))); }

	switch(OPTIMIZER)
	{
		case ADAM:
			HIP_CHECK_RETURN(hipMemset(new_max1, 0, 1*sizeof(float)));
			HIP_CHECK_RETURN(hipMemset(new_max2, 0, 1*sizeof(float)));
		hipLaunchKernelGGL((	kPreconditionOptimizerStatic8bit2State<T, OPTIMIZER>), dim3(num_blocks), dim3(256), 0, 0, p, g, state1, state2, unorm, beta1, beta2, eps, step, quantiles1, quantiles2, max1, max2, new_max1, new_max2, gnorm_scale, n);
			HIP_CHECK_RETURN(hipPeekAtLastError());
		hipLaunchKernelGGL((	kOptimizerStatic8bit2State<T, OPTIMIZER>), dim3(num_blocks), dim3(1024), 0, 0, p, g, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr,
																														quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n);
			HIP_CHECK_RETURN(hipPeekAtLastError());
		break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
			HIP_CHECK_RETURN(hipMemset(new_max1, 0, 1*sizeof(float)));
		hipLaunchKernelGGL((	kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER>), dim3(num_blocks), dim3(256), 0, 0, p, g, state1, unorm, beta1, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
			HIP_CHECK_RETURN(hipPeekAtLastError());
		hipLaunchKernelGGL((	kOptimizerStatic8bit1State<T, OPTIMIZER>), dim3(num_blocks), dim3(1024), 0, 0, p, g, state1, unorm, max_unorm, param_norm, beta1, eps, step, lr,
																														quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
			HIP_CHECK_RETURN(hipPeekAtLastError());
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

	int num_blocks = 0;
	switch(OPTIMIZER)
	{
		case ADAM:
			num_blocks = n/BLOCKSIZE_2STATE;
			num_blocks = n % BLOCKSIZE_2STATE == 0 ? num_blocks : num_blocks + 1;
		hipLaunchKernelGGL((	kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE>), dim3(num_blocks), dim3(BLOCKSIZE_2STATE/NUM_2STATE), 0, 0, p, g, state1, state2, beta1, beta2, eps, step, lr,
																														quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros, n);
			HIP_CHECK_RETURN(hipPeekAtLastError());
		break;
		case MOMENTUM:
		case RMSPROP:
    case ADAGRAD:
			num_blocks = n/BLOCKSIZE_1STATE;
			num_blocks = n % BLOCKSIZE_1STATE == 0 ? num_blocks : num_blocks + 1;
		hipLaunchKernelGGL((	kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE>), dim3(num_blocks), dim3(BLOCKSIZE_1STATE/NUM_1STATE), 0, 0, p, g, state1, beta1, beta2, eps, step, lr,
																														quantiles1, absmax1, weight_decay, gnorm_scale, skip_zeros, n);
			HIP_CHECK_RETURN(hipPeekAtLastError());
		break;
	}
}



template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n)
{
  int num_blocks = n/2048;
  num_blocks = n % 2048 == 0 ? num_blocks : num_blocks + 1;
	HIP_CHECK_RETURN(hipMemset(&gnorm_vec[step % 100], 0, 1*sizeof(float)));
 hipLaunchKernelGGL(( kPercentileClipping<T, 2048, 4>), dim3(num_blocks), dim3(512), 0, 0, g, gnorm_vec, step, n);
  HIP_CHECK_RETURN(hipPeekAtLastError());
}

void gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc)
{
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	hipblasStatus_t status;

			status = hipblasGemmEx(context->m_handle,
					transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
					transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
					m, n,	k,
					alpha, A, HIPBLAS_R_8I, lda, B, HIPBLAS_R_8I, ldb, beta,
					C, HIPBLAS_R_32I, ldc,
          HIPBLAS_R_32I, HIPBLAS_GEMM_DEFAULT);

    if (status != HIPBLAS_STATUS_SUCCESS)
    {
      std::cout << "hipBLAS ERROR: Status " << status << std::endl;
    }

}

void strided_gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc,
                    long long int strideA, long long int strideB, long long int strideC, int batchCount)
{
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	hipblasStatus_t status;

  //cout << transposeA << transposeB << endl;
  //printf("%i %i %i\n", m,n,k);
  //printf("%i %i %i\n", lda,ldb,ldc);
  //printf("%i %i %i\n", strideA, strideB, strideC);
  //printf("%i\n", batchCount);

			status = hipblasGemmStridedBatchedEx(context->m_handle,
					transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
					transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
					m, n,	k,
					alpha, A, HIPBLAS_R_8I, lda, (long long int)strideA, B, HIPBLAS_R_8I, ldb, (long long int)strideB, beta,
					C, HIPBLAS_R_32I, ldc, (long long int)strideC, batchCount,
          HIPBLAS_R_32I, HIPBLAS_GEMM_DEFAULT);

    if (status != HIPBLAS_STATUS_SUCCESS)
    {
      std::cout << "hipBLAS ERROR: Status " << status << std::endl;
    }

}

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}


#ifndef NO_HIPBLASLT
template<int ORDER> hipblasLtOrder_t get_order()
{
	switch(ORDER)
	{
		case ROW:
      return CUBLASLT_ORDER_ROW;
			break;
    case COL:
      return CUBLASLT_ORDER_COL;
      break;
    case COL32:
      return CUBLASLT_ORDER_COL32;
      break;
    case COL_TURING:
      return CUBLASLT_ORDER_COL4_4R2_8C;
      break;
    case COL_AMPERE:
      return CUBLASLT_ORDER_COL32_2R_4R4;
      break;
		default:
			break;
  }

	return CUBLASLT_ORDER_ROW;
}

template hipblasLtOrder_t get_order<ROW>();
template hipblasLtOrder_t get_order<COL>();
template hipblasLtOrder_t get_order<COL32>();
template hipblasLtOrder_t get_order<COL_TURING>();
template hipblasLtOrder_t get_order<COL_AMPERE>();
#endif


template<int ORDER> int get_leading_dim(int dim1, int dim2)
{
	switch(ORDER)
	{
		case ROW:
      return dim2;
			break;
    case COL:
      return dim1;
      break;
    case COL32:
      // 32*row tiles
      return dim1*32;
      break;
    case COL_TURING:
      return 32*roundoff(dim1, 8);
      break;
    case COL_AMPERE:
      // 32*32 tiles
      return 32*roundoff(dim1, 32);
      break;
		default:
			return 0;
			break;
  }
}

template int get_leading_dim<ROW>(int dim1, int dim2);
template int get_leading_dim<COL>(int dim1, int dim2);
template int get_leading_dim<COL32>(int dim1, int dim2);

#ifndef NO_HIPBLASLT
template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform(hipblasLtHandle_t ltHandle, T *A, T *out, int dim1, int dim2)
{
  hipblasLtOrder_t orderA = get_order<SRC>();
  hipblasLtOrder_t orderOut = get_order<TARGET>();
  int ldA = get_leading_dim<SRC>(dim1, dim2);
  int ldOut = get_leading_dim<TARGET>(dim1, dim2);

  hipblasLtMatrixLayout_t A_desc = NULL, out_desc = NULL;
  hipblasLtMatrixTransformDesc_t A2Out_desc = NULL;
  rocblas_operation opTranspose = rocblas_operation_transpose;
  float transformAlpha = 1.0f, transformBeta = 0.0f;


  if(DTYPE == 8)
  {
    checkCublasStatus(hipblasLtMatrixLayoutCreate(&A_desc, HIPBLAS_R_8I, dim1, dim2, ldA));
    checkCublasStatus(hipblasLtMatrixLayoutCreate(&out_desc, HIPBLAS_R_8I, dim1, dim2, ldOut));
  }
  else if(DTYPE == 32)
  {
    checkCublasStatus(hipblasLtMatrixLayoutCreate(&A_desc, HIPBLAS_R_32I, dim1, dim2, ldA));
    checkCublasStatus(hipblasLtMatrixLayoutCreate(&out_desc, HIPBLAS_R_32I, dim1, dim2, ldOut));
  }
  else
  {
    printf("ERROR WRONG TYPE FOR TRANSFORM: %i\n", DTYPE);
  }

  checkCublasStatus(hipblasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
  checkCublasStatus(hipblasLtMatrixLayoutSetAttribute(out_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderOut, sizeof(orderOut)));

  checkCublasStatus(hipblasLtMatrixTransformDescCreate(&A2Out_desc, HIPBLAS_R_32F));

  if(transpose){ checkCublasStatus(hipblasLtMatrixTransformDescSetAttribute(A2Out_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose))); }

  checkCublasStatus(hipblasLtMatrixTransform(ltHandle, A2Out_desc, &transformAlpha, A, A_desc, &transformBeta, NULL, NULL, out, out_desc, 0));

  if (A_desc) checkCublasStatus(hipblasLtMatrixLayoutDestroy(A_desc));
  if (out_desc) checkCublasStatus(hipblasLtMatrixLayoutDestroy(out_desc));
  if (A2Out_desc) checkCublasStatus(hipblasLtMatrixTransformDescDestroy(A2Out_desc));
}

template void transform<int8_t, ROW, COL, false, 8>(hipblasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, ROW, false, 8>(hipblasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL32, false, 8>(hipblasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, ROW, COL32, false, 32>(hipblasLtHandle_t ltHandle, int32_t *A, int32_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_TURING, false, 8>(hipblasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_AMPERE, false, 8>(hipblasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, COL32, ROW, false, 8>(hipblasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, COL32, ROW, false, 32>(hipblasLtHandle_t ltHandle, int32_t *A, int32_t *out, int dim1, int dim2);

template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt(hipblasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
{
    int has_error = 0;
    hipblasLtMatmulDesc_t matmulDesc = NULL;
    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    rocblas_operation opT = rocblas_operation_transpose;
    hipblasLtPointerMode_t alphaVec = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
    hipblasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
    hipblasLtOrder_t col_turing = CUBLASLT_ORDER_COL4_4R2_8C;
    hipblasLtOrder_t col_ampere = CUBLASLT_ORDER_COL32_2R_4R4;

    has_error |= checkCublasStatus(hipblasLtMatrixLayoutCreate(&Adesc, HIPBLAS_R_8I, m, k, lda));
    has_error |= checkCublasStatus(hipblasLtMatrixLayoutCreate(&Bdesc, HIPBLAS_R_8I, n, k, ldb));

    has_error |= checkCublasStatus(hipblasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
    if(FORMATB == COL_TURING)
      has_error |= checkCublasStatus(hipblasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_turing, sizeof(col_turing)));
    else
      has_error |= checkCublasStatus(hipblasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_ampere, sizeof(col_ampere)));

    if(DTYPE_OUT == 32)
    {
      has_error |= checkCublasStatus(hipblasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, HIPBLAS_R_32I));
      has_error |= checkCublasStatus(hipblasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
      has_error |= checkCublasStatus(hipblasLtMatrixLayoutCreate(&Cdesc, HIPBLAS_R_32I, m, n, ldc));
      has_error |= checkCublasStatus(hipblasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
      int alpha = 1, beta = 0;
      has_error |= checkCublasStatus(hipblasLtMatmul(ltHandle, matmulDesc,&alpha, A, Adesc, B, Bdesc, &beta, (int32_t*)C, Cdesc, (int32_t*)C, Cdesc, NULL, NULL, 0, 0));
    }
    else
    {
      has_error |= checkCublasStatus(hipblasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, HIPBLAS_R_32F));
      has_error |= checkCublasStatus(hipblasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
      has_error |= checkCublasStatus(hipblasLtMatrixLayoutCreate(&Cdesc, HIPBLAS_R_8I, m, n, ldc));
      has_error |= checkCublasStatus(hipblasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
      if(!SCALE_ROWS)
      {
        float alpha = 1.0f, beta = 0.0f;
        has_error |= checkCublasStatus(hipblasLtMatmul(ltHandle, matmulDesc,&alpha, A, Adesc, B, Bdesc, &beta, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, NULL, NULL, 0, 0));
      }
      else
      {
        has_error |= checkCublasStatus(hipblasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &alphaVec, sizeof(alphaVec)));
        has_error |= checkCublasStatus(hipblasLtMatmul(ltHandle, matmulDesc, row_scale, A, Adesc, B, Bdesc, NULL, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, NULL, NULL, 0, 0));
      }
    }


    if (Cdesc) has_error |= checkCublasStatus(hipblasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) has_error |= checkCublasStatus(hipblasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) has_error |= checkCublasStatus(hipblasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) has_error |= checkCublasStatus(hipblasLtMatmulDescDestroy(matmulDesc));
    if(has_error == 1)
      printf("error detected");

    return has_error;
}
#endif

int fill_up_to_nearest_multiple(int value, int multiple)
{
  return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

void dequant_mm_int32_fp16(int *A, float *rowStats, float *colStats, half *out, float* newRowStats, float* newcolStats, half *bias, int numRows, int numCols)
{
  int threads = 512;
  int tileCols = fill_up_to_nearest_multiple(numCols, 32);
  int n = numRows*tileCols;
  int subtile_rows = 128;
  int tilesize = 32*subtile_rows;
  int num_blocks = numRows/subtile_rows;
  num_blocks += (numRows % subtile_rows == 0) ? 0 : 1;
  num_blocks = num_blocks*(tileCols/32);
  assert(threads <= tilesize);

 hipLaunchKernelGGL(( kdequant_mm_int32_fp16<4, 128, 512>), dim3(num_blocks), dim3(threads), 0, 0, A, rowStats, colStats, out, newRowStats, newcolStats, bias, numRows, numCols, tileCols, n);
  HIP_CHECK_RETURN(hipPeekAtLastError());
}

#define STATS_THREADS 64
#define STATS_ITEMS 4
#define STATS_ROWS 16
void getColRowStats(half * A, float *rowStats, float *colStats, int *nnz_count_row, float nnz_threshold, int rows, int cols)
{
  int tile_cols = STATS_THREADS*STATS_ITEMS;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, STATS_ROWS);
	int row_tiles = (tiledRows/STATS_ROWS);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;

  if(nnz_threshold == 0.0)
   hipLaunchKernelGGL(( kgetColRowStats<half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>), dim3(num_blocks), dim3(STATS_THREADS), 0, 0, A, rowStats, colStats, nnz_count_row, nnz_threshold, rows, cols, tiledRows, tiledCols);
  else if(nnz_threshold != 0.0)
   hipLaunchKernelGGL(( kgetColRowStats<half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 1>), dim3(num_blocks), dim3(STATS_THREADS), 0, 0, A, rowStats, colStats, nnz_count_row, nnz_threshold, rows, cols, tiledRows, tiledCols);
  HIP_CHECK_RETURN(hipPeekAtLastError());

}

void doubleRowColQuant(half * A, float *rowStats, float *colStats, char *out_col_normed, char *out_row_normed, int *rowidx, int *colidx, half *val, int *nnz_block_ptr, float threshold, int rows, int cols)
{
  int threads = 64;
  int items_per_thread = 4;
  int tile_cols = threads*items_per_thread;
  int tile_rows = 16;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;


  if(threshold > 0.0f)
   hipLaunchKernelGGL(( kDoubleRowColQuant<64, 4, 16, 64*4, 1>), dim3(num_blocks), dim3(threads), 0, 0, A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols);
  else
   hipLaunchKernelGGL(( kDoubleRowColQuant<64, 4, 16, 64*4, 0>), dim3(num_blocks), dim3(threads), 0, 0, A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols);

  HIP_CHECK_RETURN(hipPeekAtLastError());
}

template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols)
{
  int threads = 256;
  int items_per_thread = 8;
  // we load 128 column values per warp
  int tile_cols = 32*items_per_thread;
  int tile_rows = 32;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;

  int outCols = fill_up_to_nearest_multiple(cols, 32);
  int outRows = fill_up_to_nearest_multiple(rows, 32);
  if(FORMAT == COL_TURING)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 8);
    else
      outRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 32);
    else
      outRows = fill_up_to_nearest_multiple(rows, 32);
  }
  else
  {
    if(TRANSPOSE)
    {
      outCols = fill_up_to_nearest_multiple(rows, 32);
      outRows = cols;
    }
  }

 hipLaunchKernelGGL(( kTransformRowToFormat<256, 8, 32, 32*8, TRANSPOSE, FORMAT>), dim3(num_blocks), dim3(threads), 0, 0, A, out, rows, cols, tiledCols, outRows, outCols);
  HIP_CHECK_RETURN(hipPeekAtLastError());
}

void spmm_coo(hipsparseHandle_t handle, int *A_rowidx, int *A_colidx, half *A_vals, int A_nnz, int A_rows, int A_cols, int B_cols, int ldb, half *B, int ldc, half* C, bool transposed_B)
{
    hipsparseSpMatDescr_t descA;
    hipsparseDnMatDescr_t descB, descC;

    float alpha = 1.0f;
    float beta = 0.0f;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    CHECK_HIPSPARSE( hipsparseCreateCoo(&descA, A_rows, A_cols, A_nnz,
                                      A_rowidx, A_colidx, A_vals,
                                      HIPSPARSE_INDEX_32I,
                                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_16F) );
    // Create dense matrix C
    CHECK_HIPSPARSE( hipsparseCreateDnMat(&descC, A_rows, B_cols, ldc, C,
                                        HIP_R_16F, HIPSPARSE_ORDER_ROW) );
    // Create dense matrix B
    if(transposed_B)
    {
      int tmp = A_cols;
      A_cols = B_cols;
      B_cols = tmp;
    }

    CHECK_HIPSPARSE( hipsparseCreateDnMat(&descB, A_cols, B_cols, ldb, B,
                                        HIP_R_16F, HIPSPARSE_ORDER_ROW) );
    // allocate an external buffer if needed
    CHECK_HIPSPARSE( hipsparseSpMM_bufferSize(
                                 handle,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 transposed_B ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, descA, descB, &beta, descC, HIP_R_32F,
                                 HIPSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
    HIP_CHECK_RETURN( hipMalloc(&dBuffer, bufferSize) );

    // execute SpMM
    CHECK_HIPSPARSE( hipsparseSpMM(handle,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 transposed_B ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, descA, descB, &beta, descC, HIP_R_32F,
                                 HIPSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    // destroy matrix/vector descriptors
    CHECK_HIPSPARSE( hipsparseDestroySpMat(descA) );
    CHECK_HIPSPARSE( hipsparseDestroyDnMat(descB) );
    CHECK_HIPSPARSE( hipsparseDestroyDnMat(descC) );
    HIP_CHECK_RETURN( hipFree(dBuffer) );
}

template <typename T, int BITS> void spmm_coo_very_sparse_naive(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, T *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB)
{

 hipLaunchKernelGGL(( kspmm_coo_very_sparse_naive<T, 8, BITS>), dim3(nnz_rows), dim3(256), 0, 0, max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz, rowsA, rowsB, colsB);
  HIP_CHECK_RETURN(hipPeekAtLastError());
}


template <int FORMAT> void extractOutliers(char * A, int *idx, char *out, int idx_size, int rows, int cols)
{
  int threads = 256;
  // we load 128 column values per warp
  int tiledCols = tiledCols = fill_up_to_nearest_multiple(cols, 32);
  int tiledRows = 0;

	int num_blocks = idx_size;

  if(FORMAT == COL_TURING)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 32);
	}

 hipLaunchKernelGGL(( kExtractOutliers<FORMAT>), dim3(num_blocks), dim3(threads), 0, 0, A, idx, out, idx_size, rows, cols, tiledRows, tiledCols);
  HIP_CHECK_RETURN(hipPeekAtLastError());
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void extractOutliers<COL_TURING>(char * A, int *idx, char *out, int idx_size, int rows, int cols);
template void extractOutliers<COL_AMPERE>(char * A, int *idx, char *out, int idx_size, int rows, int cols);

template void spmm_coo_very_sparse_naive<half, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, half *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);
template void spmm_coo_very_sparse_naive<signed char, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, signed char *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);

#ifndef NO_HIPBLASLT
template int igemmlt<COL_TURING, 32, 0>(hipblasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 0>(hipblasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 1>(hipblasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 32, 0>(hipblasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 0>(hipblasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 1>(hipblasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
#endif

template void transformRowToFormat<COL32, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL32, 1>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_TURING, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_TURING, 1>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_AMPERE, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_AMPERE, 1>(char * A, char *out, int rows, int cols);

template void estimateQuantiles(half *A, float *code, float offset, int n);
template void estimateQuantiles(float *A, float *code, float offset, int n);

template void quantizeBlockwise<half, 0>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<half, 1>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 1>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
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
