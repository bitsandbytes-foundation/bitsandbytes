#ifndef BITSANDBYTES_CPU_OPS_H
#define BITSANDBYTES_CPU_OPS_H

#include <iostream>
#include <stdio.h>

void quantize_cpu(float* code, float* A, float* absmax, unsigned char* out, long long blocksize, long long n);

typedef enum DataType_t {
    General8bit = 0,
    FP4 = 1,
    NF4 = 2,
} DataType_t;

using fp16_t = _Float16;

struct bf16_t {
    uint16_t v;
};

template <typename T, int DATA_TYPE>
void dequantizeBlockwiseCpu(float* code, unsigned char* A, float* absmax, T* out, long long blocksize, long long n);

template <typename T, int DATA_TYPE>
void dequantizeBlockwise4bitCpu(float* code, unsigned char* A, float* absmax, T* out, long long blocksize, long long n)

#endif
