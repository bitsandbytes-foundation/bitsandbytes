#ifndef BITSANDBYTES_CPU_OPS_H
#define BITSANDBYTES_CPU_OPS_H

#include <iostream>
#include <stdio.h>

void quantize_cpu(float *code, float *A, float *absmax, unsigned char *out, long long blocksize, long long n);
void dequantize_cpu(float *code, unsigned char *A, float *absmax, float *out, long long blocksize, long long n);

#endif
