/*
 * export.c
 *
 *  Created on: Nov 24, 2015
 *      Author: tim
 */

#include <basicOps.cuh>

#define ADAM 0

// We cannot call templated code from C, so we wrap the template in a C compatible call here if necessary.
void estimateQuantiles_fp32(float *A, float *code, float offset, int n){ estimateQuantiles<float>(A, code, offset, n); }
void estimateQuantiles_fp16(half *A, float *code, float offset, int n){ estimateQuantiles<half>(A, code, offset, n); }

void adam32bit_g32(float *g, float *p,
               float* state1, float* state2,
               const float beta1, const float beta2, const float eps, const float weight_decay,
               const int step, const float lr, const bool is_sparse, const int n){ optimizer_32bit_2State<float, ADAM>(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, n); }

void adam32bit_g16(half *g, half *p,
               float* state1, float* state2,
               const float beta1, const float beta2, const float eps, const float weight_decay,
               const int step, const float lr, const bool is_sparse, const int n){ optimizer_32bit_2State<half, ADAM>(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, n); }

void optimizer_static_8bit_2state_g16(half* p, half* g, unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* new_quantiles1, float* new_quantiles2,
                float gnorm_scale, 
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay, int n)
{ 
	optimizerStatic8bit2State<half, ADAM>(g, p, state1, state2, beta1, beta2, eps, step, lr,
			                                  quantiles1, quantiles2, new_quantiles1, new_quantiles2, gnorm_scale, max1, max2, new_max1, new_max2, weight_decay, n);
}


extern "C"
{
	void cestimate_quantiles_fp32(float *A, float *code, float offset, int n){ estimateQuantiles_fp32(A, code, offset, n); }
	void cestimate_quantiles_fp16(half *A, float *code, float offset, int n){ estimateQuantiles_fp16(A, code, offset, n); }
	void cquantize(float *code, float *A, unsigned char *out, int n){ quantize(code, A, out, n); }
	void cdequantize(float *code, unsigned char *A, float *out, int n){ dequantize(code, A, out, n); }
	void cadam32bit_g32(float *g, float *p,
								 float* state1, float* state2,
								 const float beta1, const float beta2, const float eps, const float weight_decay,
								 const int step, const float lr, const bool is_sparse, const int n){ adam32bit_g32(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, n); }
	void cadam32bit_g16(half *g, half *p,
								 float* state1, float* state2,
								 const float beta1, const float beta2, const float eps, const float weight_decay,
								 const int step, const float lr, const bool is_sparse, const int n){ adam32bit_g16(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, n); }
	void coptimizer_static_8bit_2state_g16(half* p, half* g, unsigned char* state1, unsigned char* state2,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* new_quantiles1, float* new_quantiles2,
                float gnorm_scale, 
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay, int n)
  { 
	    optimizer_static_8bit_2state_g16(g, p, state1, state2, beta1, beta2, eps, step, lr,
			                                 quantiles1, quantiles2, new_quantiles1, new_quantiles2, gnorm_scale, max1, max2, new_max1, new_max2, weight_decay, n);
  }

}


