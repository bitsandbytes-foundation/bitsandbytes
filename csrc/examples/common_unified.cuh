// common_unified.cuh — Merged architecture constants for CUDA and HIP
//
// This replaces both csrc/common.cuh and csrc/common_hip.cuh.
// Platform detection uses compat.cuh's BNB_HIP macro.

#pragma once

#include "compat.cuh"

// ============================================================================
// Warp size
// ============================================================================

#if BNB_HIP
// AMD GFX9 (CDNA) uses 64-wide warps; RDNA uses 32-wide
#ifdef __GFX9__
#define BNB_WARP_SIZE 64
#else
#define BNB_WARP_SIZE 32
#endif
#else
#define BNB_WARP_SIZE 32
#endif

// ============================================================================
// BF16 availability
// ============================================================================

#if BNB_HIP
// BF16 is available on all currently-supported ROCm architectures (CDNA2+, RDNA3+)
#define BNB_BF16_AVAILABLE true
#else
#define BNB_BF16_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_AMPERE)
#endif

// ============================================================================
// CUDA compute capability constants (CUDA-only, but harmless to define on HIP)
// ============================================================================

#define BNB_CC_PASCAL 600
#define BNB_CC_PASCAL_X2 620
#define BNB_CC_VOLTA 700
#define BNB_CC_VOLTA_XAVIER 720
#define BNB_CC_TURING 750
#define BNB_CC_AMPERE 800
#define BNB_CC_AMPERE2 860
#define BNB_CC_AMPERE2_ORIN 870
#define BNB_CC_ADA 890
#define BNB_CC_HOPPER 900
#define BNB_CC_BLACKWELL 1000

// ============================================================================
// Feature availability based on arch (CUDA uses __CUDA_ARCH__, HIP is simpler)
// ============================================================================

#if BNB_HIP
// HIP: MMA not supported via mma.h; FP8 support varies by arch
#define BNB_FP16_MMA_AVAILABLE 0
#define BNB_INT8_MMA_AVAILABLE 0
#define BNB_FP8_AVAILABLE 0
#else
#define BNB_FP16_MMA_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_VOLTA)
#define BNB_INT8_MMA_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_VOLTA_XAVIER)
#define BNB_FP8_AVAILABLE (__CUDA_ARCH__ >= BNB_CC_ADA)
#endif

// ============================================================================
// Maximum threads per SM/CU
// ============================================================================

#if BNB_HIP
// For currently supported ROCm architectures (CDNA2, RDNA3)
#define BNB_MAX_THREADS_PER_SM 2048
#else
// The maximum number of resident threads per SM varies by NVIDIA arch.
// Reference: CUDA Programming Guide, Technical Specifications per Compute Capability
#if __CUDA_ARCH__ == 750
#define BNB_MAX_THREADS_PER_SM 1024
#elif __CUDA_ARCH__ >= 860 && __CUDA_ARCH__ <= 890
#define BNB_MAX_THREADS_PER_SM 1536
#else
#define BNB_MAX_THREADS_PER_SM 2048
#endif
#endif

// Maximum resident warps per SM/CU
#define BNB_MAX_WARPS_PER_SM ((BNB_MAX_THREADS_PER_SM) / (BNB_WARP_SIZE))

// Maximum resident blocks per SM/CU
#if !BNB_HIP && (defined(__CUDA_ARCH__)) && (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870)
#define BNB_MAX_BLOCKS_PER_SM 16
#else
#define BNB_MAX_BLOCKS_PER_SM ((BNB_MAX_WARPS_PER_SM) / 2)
#endif
