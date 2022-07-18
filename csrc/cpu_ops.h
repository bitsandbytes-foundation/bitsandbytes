#ifndef BITSANDBYTES_CPU_OPS_H
#define BITSANDBYTES_CPU_OPS_H


void quantize_cpu(float *code, float *A, float *absmax, unsigned char *out, int n);

void dequantize_cpu(float *code, unsigned char *A, float *absmax, float *out, int n);

#endif
