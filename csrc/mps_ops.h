#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// MPS function declarations for Apple Silicon support
void quantize_blockwise_mps(float* code, float* A, float* absmax, unsigned char* out, 
                           long long blocksize, long long n);

void dequantize_blockwise_mps(float* code, unsigned char* A, float* absmax, float* out,
                             long long blocksize, long long n);

void gemm_4bit_inference_naive_mps(int m, int n, int k, float* A, unsigned char* B, float* C,
                                  int lda, int ldb, int ldc);

#ifdef __cplusplus
}
#endif