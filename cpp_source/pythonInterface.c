/*
 * export.c
 *
 *  Created on: Nov 24, 2015
 *      Author: tim
 */

#include <basicOps.cuh>

#define ADAM 0
#define MOMENTUM 1
#define RMSPROP 2

// We cannot call templated code from C, so we wrap the template in a C compatible call here if necessary.
// We use macro functions to expand all the different optimizers. Looks ugly, and is ugly, but its better than to 
// maintain all that boilerplate
//===================================================================================
//                               UNMANGLED CALLS
//===================================================================================

void estimateQuantiles_fp32(float *A, float *code, float offset, int n){ estimateQuantiles<float>(A, code, offset, n); }
void estimateQuantiles_fp16(half *A, float *code, float offset, int n){ estimateQuantiles<half>(A, code, offset, n); }


#define MAKE_FUNC32(fname, oname, gtype, gbits) \
void fname##32bit_g##gbits(gtype *g, gtype *p, \
               float* state1, float* state2, float *unorm, float max_unorm, float param_norm, \
               const float beta1, const float beta2, const float eps, const float weight_decay, \
               const int step, const float lr, const bool is_sparse, float gnorm_scale, const int n) \
{ optimizer32bit<gtype, oname>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n); } \

MAKE_FUNC32(momentum, MOMENTUM, float, 32)
MAKE_FUNC32(momentum, MOMENTUM, half, 16)
MAKE_FUNC32(adam, ADAM, float, 32)
MAKE_FUNC32(adam, ADAM, half, 16)
MAKE_FUNC32(rmsprop, RMSPROP, float, 32)
MAKE_FUNC32(rmsprop, RMSPROP, half, 16)

#define MAKE_FUNC8(fname, oname, gtype, gbits) \
void fname##_static_8bit_g##gbits(gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, \
								float *unorm, float max_unorm, float param_norm, \
                float beta1, float beta2, \
                float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, float gnorm_scale, int n) \
{  \
	optimizerStatic8bit<gtype, oname>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, \
			                                  quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n); \
} \

MAKE_FUNC8(adam, ADAM, float, 32)
MAKE_FUNC8(adam, ADAM, half, 16)
MAKE_FUNC8(momentum, MOMENTUM, float, 32)
MAKE_FUNC8(momentum, MOMENTUM, half, 16)
MAKE_FUNC8(rmsprop, RMSPROP, float, 32)
MAKE_FUNC8(rmsprop, RMSPROP, half, 16)

void optimizerStatic8bitBlockwise_fp32(float* p, float* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr, 
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, int n)
{	optimizerStatic8bitBlockwise<float, ADAM>(p, g, state1, state2, beta1, beta2, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, n); }
void optimizerStatic8bitBlockwise_fp16(half* p, half* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr, 
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, int n)
{	optimizerStatic8bitBlockwise<half, ADAM>(p, g, state1, state2, beta1, beta2, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, n); }


void percentileClipping_g32(float * g, float *gnorm_vec, int step, const int n){ percentileClipping<float>(g, gnorm_vec, step, n); }
void percentileClipping_g16(half * g, float *gnorm_vec, int step, const int n){ percentileClipping<half>(g, gnorm_vec, step, n); }

void quantizeBlockwise_fp16(float * code, half *A, float *absmax, unsigned char *out, const int n){ quantizeBlockwise<half>(code, A, absmax, out, n); }
void quantizeBlockwise_fp32(float * code, float *A, float *absmax, unsigned char *out, const int n){ quantizeBlockwise<float>(code, A, absmax, out, n); }

void dequantizeBlockwise_fp16(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise<half>(code, A, absmax, out, blocksize, n); } \
void dequantizeBlockwise_fp32(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise<float>(code, A, absmax, out, blocksize, n); }

extern "C"
{
	void cestimate_quantiles_fp32(float *A, float *code, float offset, int n){ estimateQuantiles_fp32(A, code, offset, n); }
	void cestimate_quantiles_fp16(half *A, float *code, float offset, int n){ estimateQuantiles_fp16(A, code, offset, n); }
	void cquantize(float *code, float *A, unsigned char *out, int n){ quantize(code, A, out, n); }
	void cdequantize(float *code, unsigned char *A, float *out, int n){ dequantize(code, A, out, n); }
  void cquantize_blockwise_fp16(float * code, half *A, float *absmax, unsigned char *out, const int n){ quantizeBlockwise_fp16(code, A, absmax, out, n); }
  void cquantize_blockwise_fp32(float * code, float *A, float *absmax, unsigned char *out, const int n){ quantizeBlockwise_fp32(code, A, absmax, out, n); }
  void cdequantize_blockwise_fp16(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n){ dequantizeBlockwise_fp16(code, A, absmax, out, blocksize, n); }
  void cdequantize_blockwise_fp32(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n){ dequantizeBlockwise_fp32(code, A, absmax, out, blocksize, n); }

	#define MAKE_CFUNC32(name, gtype, gbits) \
	void c##name##32bit_g##gbits(gtype *g, gtype *p, \
								 float* state1, float* state2, float *unorm, float max_unorm, float param_norm, \
								 const float beta1, const float beta2, const float eps, const float weight_decay, \
								 const int step, const float lr, const bool is_sparse, const float gnorm_scale, const int n) \
	{ name##32bit_g##gbits(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n); } \

	MAKE_CFUNC32(adam, float, 32)
	MAKE_CFUNC32(adam, half, 16)
	MAKE_CFUNC32(momentum, float, 32)
	MAKE_CFUNC32(momentum, half, 16)
	MAKE_CFUNC32(rmsprop, float, 32)
	MAKE_CFUNC32(rmsprop, half, 16)

	#define MAKE_CFUNC8(name, gtype, gbits) \
	void c##name##_static_8bit_g##gbits(gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, \
                float *unorm, float max_unorm, float param_norm, \
                float beta1, float beta2, \
                float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, float gnorm_scale, int n) \
  {  \
	    name##_static_8bit_g##gbits(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, \
			                                 quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n); \
  } \

	MAKE_CFUNC8(adam, float, 32)
	MAKE_CFUNC8(adam, half, 16)
	MAKE_CFUNC8(momentum, float, 32)
	MAKE_CFUNC8(momentum, half, 16)
	MAKE_CFUNC8(rmsprop, float, 32)
	MAKE_CFUNC8(rmsprop, half, 16)

  void coptimizer_static_8bit_blockwise_fp32(float* p, float* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr, 
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, int n)
  {	optimizerStatic8bitBlockwise_fp32(p, g, state1, state2, beta1, beta2, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, n); }
  void coptimizer_static_8bit_blockwise_fp16(half* p, half* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr, 
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, int n)
  {	optimizerStatic8bitBlockwise_fp16(p, g, state1, state2, beta1, beta2, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, n); }

	void cpercentile_clipping_g32(float * g, float *gnorm_vec, int step, const int n){ percentileClipping_g32(g, gnorm_vec, step, n); }
	void cpercentile_clipping_g16(half * g, float *gnorm_vec, int step, const int n){ percentileClipping_g16(g, gnorm_vec, step, n); }
}


