/*
 * export.c
 *
 *  Created on: Nov 24, 2015
 *      Author: tim
 */

#include <basicOps.cuh>

void estimateQuantiles_fp32(float *A, float *code, float offset, int n){ estimateQuantiles<float>(A, code, offset, n); }
void estimateQuantiles_fp16(half *A, float *code, float offset, int n){ estimateQuantiles<half>(A, code, offset, n); }


extern "C"
{
	void estimate_quantiles_fp32(float *A, float *code, float offset, int n){ estimateQuantiles_fp32(A, code, offset, n); }
	void estimate_quantiles_fp16(half *A, float *code, float offset, int n){ estimateQuantiles_fp16(A, code, offset, n); }
}


