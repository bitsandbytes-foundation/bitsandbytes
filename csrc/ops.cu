#include <ops.cuh>
#include <kernels.cuh>
#include <cub/device/device_scan.cuh>
#include <limits>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cassert>

using std::cout;
using std::endl;

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
                const int step, const float lr, const float gnorm_scale, const int n)
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
			kOptimizer32bit2State<T, OPTIMIZER><<<blocks, 1024>>>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
			break;
		case MOMENTUM:
    case RMSPROP:
      if(max_unorm > 0.0f)
			{ 
				CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1*sizeof(float)));
				kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8><<<blocks, 512>>>(g, p, state1, unorm, beta1, eps, weight_decay, step, lr, gnorm_scale, n);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
			}

			kOptimizer32bit1State<T, OPTIMIZER><<<blocks, 1024>>>(g, p, state1, unorm, max_unorm, param_norm, beta1, eps, weight_decay, step, lr, gnorm_scale, n);
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
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, int n)
{

	int blocks = 0;
	switch(OPTIMIZER)
	{
		case ADAM:
			blocks = n/BLOCKSIZE_2STATE;
			blocks = n % BLOCKSIZE_2STATE == 0 ? blocks : blocks + 1;
			kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE><<<blocks, BLOCKSIZE_2STATE/NUM_2STATE>>>(p, g, state1, state2, beta1, beta2, eps, step, lr,
																														quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		break;
		case MOMENTUM:
		case RMSPROP:
			blocks = n/BLOCKSIZE_1STATE;
			blocks = n % BLOCKSIZE_1STATE == 0 ? blocks : blocks + 1;
			kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE><<<blocks, BLOCKSIZE_1STATE/NUM_1STATE>>>(p, g, state1, beta1, beta2, eps, step, lr,
																														quantiles1, absmax1, weight_decay, gnorm_scale, n);
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

#define UNSIGNED_CHAR 0

void gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc)
{
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	cublasStatus_t status;

			status = cublasGemmEx(context->m_handle,
					transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
					transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
					m, n,	k,
					alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I, ldb, beta,
					C, CUDA_R_32I, ldc,
          CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }

}

void strided_gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc, 
                    long long int strideA, long long int strideB, long long int strideC, int batchCount)
{
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	cublasStatus_t status;

  //cout << transposeA << transposeB << endl;
  //printf("%i %i %i\n", m,n,k);
  //printf("%i %i %i\n", lda,ldb,ldc);
  //printf("%i %i %i\n", strideA, strideB, strideC);
  //printf("%i\n", batchCount);

			status = cublasGemmStridedBatchedEx(context->m_handle,
					transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
					transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
					m, n,	k,
					alpha, A, CUDA_R_8I, lda, (long long int)strideA, B, CUDA_R_8I, ldb, (long long int)strideB, beta,
					C, CUDA_R_32I, ldc, (long long int)strideC, batchCount,
          CUDA_R_32I, CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }

}

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}

//void igemmLt(Context *context, bool transposeA, bool transposeB, int m, int n, int k, const void *A, const void *B, void *C, int lda, int ldb, int ldc)
//{
//
//	cublasLtHandle_t ltHandle = (cublasLtHandle_t)context->m_handle;
//
//	cublasLtMatmulDesc_t matmulDesc = NULL;
//	cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
//	cublasOperation_t opTranspose = CUBLAS_OP_T;
//	const void *D = C;
//
//  const int falpha = 1;
//  const int fbeta = 0;
//  const void * alpha = &falpha;
//  const void * beta = &fbeta;
//
//	cublasLtMatrixTransformDesc_t transformDesc = NULL;
//	int8_t *Atransform = NULL, *Btransform = NULL;
//	int32_t *Ctransform                   = NULL;
//	cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
//	float transformAlpha = 1.0f, transformBeta = 0.0f;
//	cublasLtOrder_t order_COL = CUBLASLT_ORDER_COL;
//	cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
//	cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
//
//	//int ldatransform = 32 * m;
//	//int ldbtransform = 32 * roundoff(n, 8);
//	//int ldctransform = 32 * m;
//
//	//checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Atransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldatransform));
//	//checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Btransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldbtransform));
//	//checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Ctransform), sizeof(int32_t) * roundoff(n, 32) / 32 * ldctransform));
//
//	//checkCublasStatus(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));
//
//	checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
//	if(transposeA)
//		checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
//	if(transposeB)
//		checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose)));
//
//	// ---------------------------------------------------------------------------------------------
//	// create descriptors for original matrices
//
//	checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda));
//	checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, ldb));
//	checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc));
//
//	// ---------------------------------------------------------------------------------------------
//	// create descriptors for transformed matrices
//
//	cout << m << " " << n << " " << k << endl;
//	checkCublasStatus(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, lda));
//	checkCublasStatus(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldb));
//	checkCublasStatus(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldc));
//
//	checkCublasStatus(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL, sizeof(order_COL)));
//	checkCublasStatus(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL, sizeof(order_COL)));
//	checkCublasStatus(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL, sizeof(order_COL)));
//
//	// no need to transform C matrix as beta is assumed to be 0
// checkCublasStatus(cublasLtMatmul(ltHandle,
//  															 matmulDesc,
//  															 alpha,
//  															 A,
//  															 AtransformDesc,
//  															 B,
//  															 BtransformDesc,
//  															 beta,
//  															 D,
//  															 CtransformDesc,
//  															 C,
//  															 CtransformDesc,
//  															 NULL,
//  															 NULL,
//  															 0,
//  															 0));
// 
//}

typedef enum Transform_t
{
	ROW = 0,
	COL = 1,
  COL32 = 2,
  COL_TURING = 3,
  COL_AMPERE = 4,
} Transform_t;

template<int ORDER> cublasLtOrder_t get_order()
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
  }
}

template cublasLtOrder_t get_order<ROW>();
template cublasLtOrder_t get_order<COL>();
template cublasLtOrder_t get_order<COL32>();
template cublasLtOrder_t get_order<COL_TURING>();
template cublasLtOrder_t get_order<COL_AMPERE>();


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
      // 32*8 tiles
      //int num_blocks = (dim1 + (8 - (dim1 % 8)))/8;
      //return 32*(dim1 + (8 - (dim1 % 8)));
      //return 8*(dim1  + (32 - (dim1 % 32)));
      //return 8*(dim2 + (32 - (dim2 % 32)));
      //return 256;
      return 32*roundoff(dim1, 8);
      //return 32*roundoff(dim2, 8);
      //return 8*roundoff(dim2, 32);
      //return 8*roundoff(dim1, 32);
      break;
    case COL_AMPERE:
      // 32*32 tiles
      return 32*roundoff(dim1, 32);
      break;
  }
}

template int get_leading_dim<ROW>(int dim1, int dim2);
template int get_leading_dim<COL>(int dim1, int dim2);
template int get_leading_dim<COL32>(int dim1, int dim2);

template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform(cublasLtHandle_t ltHandle, T *A, T *out, int dim1, int dim2)
{

  cublasLtOrder_t orderA = get_order<SRC>();
  cublasLtOrder_t orderOut = get_order<TARGET>();
  int ldA = get_leading_dim<SRC>(dim1, dim2);
  int ldOut = get_leading_dim<TARGET>(dim1, dim2);
  
  cublasLtMatrixLayout_t A_desc = NULL, out_desc = NULL;
  cublasLtMatrixTransformDesc_t A2Out_desc = NULL;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  float transformAlpha = 1.0f, transformBeta = 0.0f;


  if(DTYPE == 8)
  {
    checkCublasStatus(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_8I, dim1, dim2, ldA));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&out_desc, CUDA_R_8I, dim1, dim2, ldOut));
  }
  else if(DTYPE == 32)
  {
    checkCublasStatus(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_32I, dim1, dim2, ldA));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&out_desc, CUDA_R_32I, dim1, dim2, ldOut));
  }
  else
  {
    printf("ERROR WRONG TYPE FOR TRANSFORM: %i\n", DTYPE);
  }

  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(out_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderOut, sizeof(orderOut)));

  checkCublasStatus(cublasLtMatrixTransformDescCreate(&A2Out_desc, CUDA_R_32F));

  if(transpose){ checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(A2Out_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose))); }

  checkCublasStatus(cublasLtMatrixTransform(ltHandle, A2Out_desc, &transformAlpha, A, A_desc, &transformBeta, NULL, NULL, out, out_desc, 0));

  if (A_desc) checkCublasStatus(cublasLtMatrixLayoutDestroy(A_desc));
  if (out_desc) checkCublasStatus(cublasLtMatrixLayoutDestroy(out_desc));
  if (A2Out_desc) checkCublasStatus(cublasLtMatrixTransformDescDestroy(A2Out_desc));
}

template void transform<int8_t, ROW, COL, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, ROW, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL32, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, ROW, COL32, false, 32>(cublasLtHandle_t ltHandle, int32_t *A, int32_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_TURING, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_AMPERE, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, COL32, ROW, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, COL32, ROW, false, 32>(cublasLtHandle_t ltHandle, int32_t *A, int32_t *out, int dim1, int dim2);

void LtIgemm(cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const int8_t *A,
                   const int8_t *B,
                   int32_t *C,
                   int lda,
                   int ldb,
                   int ldc) 
{
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    int32_t alpha = 1, beta = 0;
    cublasOperation_t opT = CUBLAS_OP_T;
    //cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
    //cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t col_turing = CUBLASLT_ORDER_COL4_4R2_8C;

    //cublasLtMatmulPreference_t pref = NULL;
    //checkCublasStatus(cublasLtMatmulPreferenceCreate(&pref)
    //checkCublasStatus(cublasLtMatmulPreferenceInit(pref)

    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, n, k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc));

    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_turing, sizeof(col_turing)));

    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     matmulDesc,
                                     &alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     &beta,
                                     C,
                                     Cdesc,
                                     C,
                                     Cdesc,
                                     NULL,
                                     NULL,
                                     0,
                                     0));

    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
}

void cutlass_igemm(bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc)
{

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::ColumnMajor,              // LayoutA
    int8_t,
    cutlass::layout::ColumnMajor,              // LayoutB
    int32_t,
    cutlass::layout::ColumnMajor,              // LayoutOutput
    int32_t,                                     // ElementAccumulator
    cutlass::arch::OpClassWmmaTensorOp,            // tag indicating Tensor Cores
    cutlass::arch::Sm75,                        // tag indicating target GPU compute architecture
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 16, 16>
    //cutlass::gemm::GemmShape<32, 32, 16>
  >;

  //using Gemm = cutlass::gemm::device::Gemm<
  //  int8_t,
  //  cutlass::layout::RowMajor,              // LayoutA
  //  int8_t,
  //  cutlass::layout::ColumnMajor,              // LayoutB
  //  int32_t,
  //  cutlass::layout::ColumnMajor,              // LayoutOutput
  //  int32_t,                                     // ElementAccumulator
  //  cutlass::arch::OpClassTensorOp,            // tag indicating Tensor Cores
  //  cutlass::arch::Sm75                        // tag indicating target GPU compute architecture
  //  //cutlass::gemm::GemmShape<64, 128, 64>,
  //  //cutlass::gemm::GemmShape<32, 32, 64>,
  //  //cutlass::gemm::GemmShape<16, 16, 16>
  //  //cutlass::gemm::GemmShape<32, 32, 16>
  //>;

  Gemm gemm_op;
  cutlass::Status status;

  int alpha = 1;
  int beta = 0;

  int8_t const *ptrA = (int8_t*)A;
  int8_t const *ptrB = (int8_t*)B;
  int32_t const *ptrC = (int32_t*)C;

  int32_t       *ptrD = (int32_t*)C;
	int ldd = ldc;

  //
  // Launch GEMM on the device
  //
  status = gemm_op({
    {m, n, k},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrD, ldd},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
    });

  if (status != cutlass::Status::kSuccess)
	{
		printf("ERROR\n");
  }
} 

int fill_up_to_nearest_multiple(int value, int multiple)
{
  return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

void dequant_mm_int32_fp16(int *A, float *rowStats, float *colStats, half *out, float* newRowStats, float* newcolStats, int numRows, int numCols)
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

  //cout << num_blocks << " blocks" << endl;

  kdequant_mm_int32_fp16<4, 128, 512><<<num_blocks, threads>>>(A, rowStats, colStats, out, newRowStats, newcolStats, numRows, numCols, tileCols, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

#define STATS_THREADS 64
#define STATS_ITEMS 4
#define STATS_ROWS 16
void getColRowStats(half * A, float *rowStats, float *colStats, int rows, int cols)
{
  int tile_cols = STATS_THREADS*STATS_ITEMS;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, STATS_ROWS);
  int num_blocks = (tiledCols/tile_cols) * (tiledRows/STATS_ROWS);

  kgetColRowStats<half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS><<<num_blocks, STATS_THREADS>>>(A, rowStats, colStats, rows, cols, tiledRows, tiledCols);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());

}

void doubleRowColQuant(half * A, float *rowStats, float *colStats, char *out_col_normed, char *out_row_normed, int rows, int cols)
{
  int threads = 64;
  int items_per_thread = 4;
  int tile_cols = threads*items_per_thread;
  int tile_rows = 16;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
  int num_blocks = (tiledCols/tile_cols) * (tiledRows/tile_rows);

  //cout << cols << " " << tiledCols << " " << tiledRows << endl;
  //cout << "num blocks " << num_blocks << endl;

  //cout << A << " " << out_col_normed << endl;
  kDoubleRowColQuant<64, 4, 16, 64*4><<<num_blocks, threads>>>(A, rowStats, colStats, out_col_normed, out_row_normed, rows, cols, tiledCols);
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
                const int step, const float lr, const float gnorm_scale, const int n);

MAKE_optimizer32bit(ADAM, half)
MAKE_optimizer32bit(ADAM, float)
MAKE_optimizer32bit(MOMENTUM, half)
MAKE_optimizer32bit(MOMENTUM, float)
MAKE_optimizer32bit(RMSPROP, half)
MAKE_optimizer32bit(RMSPROP, float)

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
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, int n); \

MAKE_optimizerStatic8bitBlockwise(half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);

template void percentileClipping(float * g, float *gnorm_vec, int step, const int n);
template void percentileClipping(half * g, float *gnorm_vec, int step, const int n);
