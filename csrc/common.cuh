#pragma once

// TODO: Let's make some of these constexpr and put in a namespace.

#define BNB_CC_MAXWELL          500
#define BNB_CC_MAXWELL2         520
#define BNB_CC_MAXWELL2_X1      530
#define BNB_CC_PASCAL           600
#define BNB_CC_PASCAL_X2        620
#define BNB_CC_VOLTA            700
#define BNB_CC_VOLTA_XAVIER     720
#define BNB_CC_TURING           750
#define BNB_CC_AMPERE           800
#define BNB_CC_AMPERE2          860
#define BNB_CC_AMPERE2_ORIN     870
#define BNB_CC_ADA              890
#define BNB_CC_HOPPER           900
#define BNB_CC_BLACKWELL        1000

#define BNB_FP16_AVAILABLE      (__CUDA_ARCH__ >= BNB_CC_MAXWELL2_X1)
#define BNB_FP16_MMA_AVAILABLE  (__CUDA_ARCH__ >= BNB_CC_VOLTA)
#define BNB_INT8_MMA_AVAILABLE  (__CUDA_ARCH__ >= BNB_CC_VOLTA_XAVIER)
#define BNB_BF16_AVAILABLE      (__CUDA_ARCH__ >= BNB_CC_AMPERE)
#define BNB_FP8_AVAILABLE       (__CUDA_ARCH__ >= BNB_CC_ADA)

#define BNB_WARP_SIZE   32

// The maximum number of resident threads per SM varies by arch.
// For A100/H100 and all prior to Turing, it is 2048, which allows
// for 2 full blocks of 1024 threads per SM.
// Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
#if __CUDA_ARCH__ == 750
#define BNB_MAX_THREADS_PER_SM      1024
#elif __CUDA_ARCH__ >= 860 && __CUDA_ARCH__ <= 890
#define BNB_MAX_THREADS_PER_SM      1536
#else
#define BNB_MAX_THREADS_PER_SM      2048
#endif

// Maximum resident warps per SM is always directly related to the number of threads.
#define BNB_MAX_WARPS_PER_SM        ((BNB_MAX_THREADS_PER_SM) / (BNB_WARP_SIZE))

// Maximum resident blocks per SM may vary.
#if __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870
#define BNB_MAX_BLOCKS_PER_SM       16
#else
#define BNB_MAX_BLOCKS_PER_SM       ((BNB_MAX_WARPS_PER_SM) / 2)
#endif
