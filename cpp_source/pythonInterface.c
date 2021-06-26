/*
 * export.c
 *
 *  Created on: Nov 24, 2015
 *      Author: tim
 */

#include <basicOps.cuh>

#define ADAM 0
#define MOMENTUM 1

// We cannot call templated code from C, so we wrap the template in a C compatible call here if necessary.
//===================================================================================
//                               UNMANGLED CALLS
//===================================================================================

void estimateQuantiles_fp32(float *A, float *code, float offset, int n){ estimateQuantiles<float>(A, code, offset, n); }
void estimateQuantiles_fp16(half *A, float *code, float offset, int n){ estimateQuantiles<half>(A, code, offset, n); }

void momentum32bit_g32(float *g, float *p,
               float* state1, float* state2,
               const float beta1, const float beta2, const float eps, const float weight_decay,
               const int step, const float lr, const bool is_sparse, float gnorm_scale, const int n)
{ optimizer_32bit<float, MOMENTUM>(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n); }

void momentum32bit_g16(half *g, half *p,
               float* state1, float* state2,
               const float beta1, const float beta2, const float eps, const float weight_decay,
               const int step, const float lr, const bool is_sparse, float gnorm_scale, const int n)
{ optimizer_32bit<half, MOMENTUM>(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n); }

void adam32bit_g32(float *g, float *p,
               float* state1, float* state2,
               const float beta1, const float beta2, const float eps, const float weight_decay,
               const int step, const float lr, const bool is_sparse, float gnorm_scale, const int n)
{ optimizer_32bit<float, ADAM>(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n); }

void adam32bit_g16(half *g, half *p,
               float* state1, float* state2,
               const float beta1, const float beta2, const float eps, const float weight_decay,
               const int step, const float lr, const bool is_sparse, float gnorm_scale, const int n)
{ optimizer_32bit<half, ADAM>(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n); }

void optimizer_static_8bit_2state_g16(half* p, half* g, unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay, float gnorm_scale, int n)
{ 
	optimizerStatic8bit<half, ADAM>(g, p, state1, state2, beta1, beta2, eps, step, lr,
			                                  quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n);
}

void optimizer_static_8bit_2state_g32(float* p, float* g, unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay, float gnorm_scale, int n)
{ 
	optimizerStatic8bit<float, ADAM>(g, p, state1, state2, beta1, beta2, eps, step, lr,
			                                  quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n);
}

void percentileClipping_g32(float * g, float *gnorm_vec, int step, const int n){ percentileClipping<float>(g, gnorm_vec, step, n); }
void percentileClipping_g16(half * g, float *gnorm_vec, int step, const int n){ percentileClipping<half>(g, gnorm_vec, step, n); }

extern "C"
{
	void cestimate_quantiles_fp32(float *A, float *code, float offset, int n){ estimateQuantiles_fp32(A, code, offset, n); }
	void cestimate_quantiles_fp16(half *A, float *code, float offset, int n){ estimateQuantiles_fp16(A, code, offset, n); }
	void cquantize(float *code, float *A, unsigned char *out, int n){ quantize(code, A, out, n); }
	void cdequantize(float *code, unsigned char *A, float *out, int n){ dequantize(code, A, out, n); }
	void cadam32bit_g32(float *g, float *p,
								 float* state1, float* state2,
								 const float beta1, const float beta2, const float eps, const float weight_decay,
								 const int step, const float lr, const bool is_sparse, const float gnorm_scale, const int n)
	{ adam32bit_g32(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n); }
	void cadam32bit_g16(half *g, half *p,
								 float* state1, float* state2,
								 const float beta1, const float beta2, const float eps, const float weight_decay,
								 const int step, const float lr, const bool is_sparse, const float gnorm_scale, const int n)
	{ adam32bit_g16(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n); }

	void cmomentum32bit_g32(float *g, float *p,
								 float* state1, float* state2,
								 const float beta1, const float beta2, const float eps, const float weight_decay,
								 const int step, const float lr, const bool is_sparse, const float gnorm_scale, const int n)
	{ momentum32bit_g32(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n); }
	void cmomentum32bit_g16(half *g, half *p,
								 float* state1, float* state2,
								 const float beta1, const float beta2, const float eps, const float weight_decay,
								 const int step, const float lr, const bool is_sparse, const float gnorm_scale, const int n)
	{ momentum32bit_g16(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, gnorm_scale, n); }

	void coptimizer_static_8bit_2state_g16(half* p, half* g, unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay, float gnorm_scale, int n)
  { 
	    optimizer_static_8bit_2state_g16(g, p, state1, state2, beta1, beta2, eps, step, lr,
			                                 quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n);
  }

	void coptimizer_static_8bit_2state_g32(float* p, float* g, unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay, float gnorm_scale, int n)
  { 
	    optimizer_static_8bit_2state_g32(g, p, state1, state2, beta1, beta2, eps, step, lr,
			                                 quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n);
  }

	void cpercentile_clipping_g32(float * g, float *gnorm_vec, int step, const int n){ percentileClipping_g32(g, gnorm_vec, step, n); }
	void cpercentile_clipping_g16(half * g, float *gnorm_vec, int step, const int n){ percentileClipping_g16(g, gnorm_vec, step, n); }
}


