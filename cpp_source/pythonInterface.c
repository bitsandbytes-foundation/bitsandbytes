/*
 * export.c
 *
 *  Created on: Nov 24, 2015
 *      Author: tim
 */

#include <basicOps.cuh>

void scalarMul2(float *A, float *out, int n, float scalar){ elementWise<ksmul>(A, out, n, scalar); }


extern "C"
{
  void ffscalar_mul(float *A, float *out, int n, float scalar){ scalarMul2(A, out, n, scalar); }
	void estimate_quantiles(float *A, float *code, float offset, int n){ estimateQuantiles(A, code, offset, n); }
}


