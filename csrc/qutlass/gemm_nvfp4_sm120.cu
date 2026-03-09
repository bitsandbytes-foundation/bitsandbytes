/*
 * NVFP4 GEMM for SM_120 (consumer Blackwell) using CUTLASS.
 *
 * This file contains both the dense GEMM and batched MoE GEMM implementations.
 * Both route through a single runGemm function to ensure device_kernel
 * is only instantiated once per tile type.
 *
 * CRITICAL: The template parameter for runGemm must be the Gemm adapter type
 * directly (e.g., FpGemmSmall::Gemm), NOT the config struct (FpGemmSmall).
 * Using a config struct as the template parameter causes nvcc to generate
 * separate device_kernel stubs due to dependent type resolution differences,
 * even when the underlying GemmKernel type is identical.
 *
 * CUDA Graph Support:
 *   gemm.initialize() calls cudaFuncSetAttribute and is NOT graph-capturable.
 *   gemm.run() only launches the kernel and IS graph-capturable.
 *   The _init functions do can_implement + initialize (call once, outside capture).
 *   The _run functions call gemm.run(stream) only (graph-capturable).
 *
 * nvcc workaround: nvcc cannot bind GemmUniversalAdapter objects to template
 * function reference parameters due to a type resolution bug with deeply nested
 * CUTLASS template types. The init functions therefore inline the CUTLASS setup
 * code rather than routing through runGemm.
 *
 * Dense GEMM: derived from QuTLASS (https://github.com/IST-DASLab/qutlass)
 * Copyright (C) 2025 Roberto L. Castro (Roberto.LopezCastro@ist.ac.at)
 * Licensed under the Apache License, Version 2.0
 */

#include <cstdio>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "include/gemm_nvfp4_sm120_types.h"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

// =====================================================================
// Single entry point for non-graph CUTLASS GEMM calls.
// Creates its own local Gemm object — does can_implement + initialize + run.
// Template parameter MUST be the Gemm adapter type directly.
// =====================================================================
template <typename Gemm, typename ElementScaleFactor>
static int runGemm(
    const void* A_ptr, const void* B_ptr,
    const void* SFA_ptr, const void* SFB_ptr,
    void* D_ptr, const float* alpha_ptr,
    int M, int N, int K, int L,
    typename Gemm::GemmKernel::StrideA const& stride_A,
    typename Gemm::GemmKernel::StrideB const& stride_B,
    typename Gemm::GemmKernel::StrideD const& stride_D,
    typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA const& layout_SFA,
    typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB const& layout_SFB,
    void* workspace, cudaStream_t stream
) {
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementD = cutlass::bfloat16_t;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        {static_cast<ElementA const*>(A_ptr), stride_A,
         static_cast<ElementB const*>(B_ptr), stride_B,
         static_cast<ElementScaleFactor const*>(SFA_ptr), layout_SFA,
         static_cast<ElementScaleFactor const*>(SFB_ptr), layout_SFB},
        {{}, static_cast<ElementD const*>(D_ptr), stride_D,
         static_cast<ElementD*>(D_ptr), stride_D}
    };
    arguments.epilogue.thread.alpha_ptr = alpha_ptr;

    Gemm gemm;
    cutlass::Status status;

    status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM can_implement failed: %d\n", (int)status);
        return -1;
    }

    status = gemm.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM initialize failed: %d\n", (int)status);
        return -2;
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM run failed: %d\n", (int)status);
        return -3;
    }

    return 0;
}

// Convenience aliases
using GemmSmall = FpGemmSmall::Gemm;
using GemmLarge = FpGemmLarge::Gemm;
using ESF = cutlass::float_ue4m3_t;

// =====================================================================
// Helper: initialize a Gemm adapter object (can_implement + initialize).
// Uses void* to sidestep nvcc's reference binding bug with CUTLASS types.
// The caller must ensure gemm_ptr points to a valid Gemm object.
// =====================================================================
template <typename Gemm, typename ElementScaleFactor>
static int initGemmAdapter(
    void* gemm_ptr,
    const void* A_ptr, const void* B_ptr,
    const void* SFA_ptr, const void* SFB_ptr,
    void* D_ptr, const float* alpha_ptr,
    int M, int N, int K, int L,
    typename Gemm::GemmKernel::StrideA const& stride_A,
    typename Gemm::GemmKernel::StrideB const& stride_B,
    typename Gemm::GemmKernel::StrideD const& stride_D,
    typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA const& layout_SFA,
    typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB const& layout_SFB,
    void* workspace, cudaStream_t stream
) {
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementD = cutlass::bfloat16_t;

    Gemm* gemm = static_cast<Gemm*>(gemm_ptr);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        {static_cast<ElementA const*>(A_ptr), stride_A,
         static_cast<ElementB const*>(B_ptr), stride_B,
         static_cast<ElementScaleFactor const*>(SFA_ptr), layout_SFA,
         static_cast<ElementScaleFactor const*>(SFB_ptr), layout_SFB},
        {{}, static_cast<ElementD const*>(D_ptr), stride_D,
         static_cast<ElementD*>(D_ptr), stride_D}
    };
    arguments.epilogue.thread.alpha_ptr = alpha_ptr;

    cutlass::Status status;

    status = gemm->can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM can_implement failed: %d\n", (int)status);
        return -1;
    }

    status = gemm->initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM initialize failed: %d\n", (int)status);
        return -2;
    }

    return 0;
}

// =====================================================================
// Helper: launch a pre-initialized Gemm adapter (graph-capturable).
// Uses void* to sidestep nvcc's reference binding bug.
// =====================================================================
template <typename Gemm>
static int launchGemm(void* gemm_ptr, cudaStream_t stream) {
    Gemm* gemm = static_cast<Gemm*>(gemm_ptr);
    cutlass::Status status = gemm->run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM run failed: %d\n", (int)status);
        return -3;
    }
    return 0;
}

// =====================================================================
// Persistent GEMM state for CUDA graph support
// =====================================================================
struct MoeGemmState {
    bool initialized = false;
    bool use_large_tile = false;

    int cutlass_M, cutlass_N, cutlass_K, num_experts;

    // Initialized Gemm objects (persist between init and run for graph capture)
    GemmSmall gemm_small;
    GemmLarge gemm_large;

    cutlass::KernelHardwareInfo hw_info;
    void* workspace_dev = nullptr;
    size_t workspace_size = 0;
};

static MoeGemmState s_state;

// Dense GEMM state for CUDA graph support (init/run split)
struct DenseGemmState {
    bool initialized = false;
    bool use_large_tile = false;

    GemmSmall gemm_small;
    GemmLarge gemm_large;
};

static DenseGemmState s_dense_state;

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED

// =====================================================================
// extern "C" — Dense GEMM
// =====================================================================

// Non-graph path: full init+run in one call (backward compatible)
extern "C" void cgemm_nvfp4_cutlass(
    const void* A, const void* B, const void* SFA, const void* SFB,
    void* D, int M, int N, int K, const float* alpha, cudaStream_t stream
) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    if (M < 512) {
        using Config = FpGemmSmall;
        auto stride_A = cutlass::make_cute_packed_stride(Config::StrideA{}, {M, K, 1});
        auto stride_B = cutlass::make_cute_packed_stride(Config::StrideB{}, {N, K, 1});
        auto stride_D = cutlass::make_cute_packed_stride(Config::StrideD{}, {M, N, 1});
        auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
        runGemm<GemmSmall, ESF>(A, B, SFA, SFB, D, alpha, M, N, K, 1,
            stride_A, stride_B, stride_D, layout_SFA, layout_SFB, nullptr, stream);
    } else {
        using Config = FpGemmLarge;
        auto stride_A = cutlass::make_cute_packed_stride(Config::StrideA{}, {M, K, 1});
        auto stride_B = cutlass::make_cute_packed_stride(Config::StrideB{}, {N, K, 1});
        auto stride_D = cutlass::make_cute_packed_stride(Config::StrideD{}, {M, N, 1});
        auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
        runGemm<GemmLarge, ESF>(A, B, SFA, SFB, D, alpha, M, N, K, 1,
            stride_A, stride_B, stride_D, layout_SFA, layout_SFB, nullptr, stream);
    }
#endif
}

// Graph-compatible path: initialize once (call outside graph capture)
extern "C" int cgemm_nvfp4_dense_init(
    const void* A, const void* B, const void* SFA, const void* SFB,
    void* D, int M, int N, int K, const float* alpha,
    void* workspace, size_t workspace_size, cudaStream_t stream
) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    auto& st = s_dense_state;
    st.use_large_tile = (M >= 512);

    int ret;
    if (st.use_large_tile) {
        using Config = FpGemmLarge;
        auto stride_A = cutlass::make_cute_packed_stride(Config::StrideA{}, {M, K, 1});
        auto stride_B = cutlass::make_cute_packed_stride(Config::StrideB{}, {N, K, 1});
        auto stride_D = cutlass::make_cute_packed_stride(Config::StrideD{}, {M, N, 1});
        auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
        ret = initGemmAdapter<GemmLarge, ESF>(&st.gemm_large,
            A, B, SFA, SFB, D, alpha, M, N, K, 1,
            stride_A, stride_B, stride_D, layout_SFA, layout_SFB, workspace, stream);
    } else {
        using Config = FpGemmSmall;
        auto stride_A = cutlass::make_cute_packed_stride(Config::StrideA{}, {M, K, 1});
        auto stride_B = cutlass::make_cute_packed_stride(Config::StrideB{}, {N, K, 1});
        auto stride_D = cutlass::make_cute_packed_stride(Config::StrideD{}, {M, N, 1});
        auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
        ret = initGemmAdapter<GemmSmall, ESF>(&st.gemm_small,
            A, B, SFA, SFB, D, alpha, M, N, K, 1,
            stride_A, stride_B, stride_D, layout_SFA, layout_SFB, workspace, stream);
    }
    if (ret == 0) st.initialized = true;
    return ret;
#else
    return -1;
#endif
}

// Graph-compatible path: launch only (CUDA-graph-capturable)
extern "C" int cgemm_nvfp4_dense_run(cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    auto& st = s_dense_state;
    if (!st.initialized) { fprintf(stderr, "Dense NVFP4 GEMM not initialized.\n"); return -1; }

    if (st.use_large_tile) {
        return launchGemm<GemmLarge>(&st.gemm_large, stream);
    } else {
        return launchGemm<GemmSmall>(&st.gemm_small, stream);
    }
#else
    return -1;
#endif
}

// =====================================================================
// extern "C" — Batched MoE GEMM
// =====================================================================

extern "C" size_t cgemm_nvfp4_moe_sm120_sfa_size(int N_output, int max_M, int K_hidden, int num_experts) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden, L = num_experts;
    auto layout_SFA = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
    return size(filter_zeros(layout_SFA)) * sizeof(FpGemmSmall::ElementSF);
#else
    return 0;
#endif
}

extern "C" size_t cgemm_nvfp4_moe_sm120_sfb_size(int N_output, int max_M, int K_hidden, int num_experts) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden, L = num_experts;
    auto layout_SFB = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));
    return size(filter_zeros(layout_SFB)) * sizeof(FpGemmSmall::ElementSF);
#else
    return 0;
#endif
}

extern "C" size_t cgemm_nvfp4_moe_sm120_sfa_size_per_expert(int N_output, int max_M, int K_hidden) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden;
    auto layout = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    return size(filter_zeros(layout)) * sizeof(FpGemmSmall::ElementSF);
#else
    return 0;
#endif
}

extern "C" size_t cgemm_nvfp4_moe_sm120_sfb_size_per_expert(int N_output, int max_M, int K_hidden) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden;
    auto layout = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    return size(filter_zeros(layout)) * sizeof(FpGemmSmall::ElementSF);
#else
    return 0;
#endif
}

extern "C" int cgemm_nvfp4_moe_sm120_init(
    int N_output, int max_M, int K_hidden, int num_experts,
    const void* A_dev, const void* B_dev,
    const void* SFA_dev, const void* SFB_dev,
    void* D_dev, const float* alpha_dev,
    void* workspace_dev, size_t workspace_size, cudaStream_t stream
) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    auto& st = s_state;
    st.cutlass_M = max_M; st.cutlass_N = N_output;
    st.cutlass_K = K_hidden; st.num_experts = num_experts;
    st.use_large_tile = (max_M >= 512);

    int M = max_M, N = N_output, K = K_hidden, L = num_experts;

    st.hw_info.device_id = 0;
    st.hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    st.workspace_dev = workspace_dev;
    st.workspace_size = workspace_size;

    // Initialize the CUTLASS Gemm adapter (cudaFuncSetAttribute etc.)
    // This must happen outside CUDA graph capture.
    int ret;
    if (st.use_large_tile) {
        using Config = FpGemmLarge;
        auto stride_A = cutlass::make_cute_packed_stride(Config::StrideA{}, {M, K, L});
        auto stride_B = cutlass::make_cute_packed_stride(Config::StrideB{}, {N, K, L});
        auto stride_D = cutlass::make_cute_packed_stride(Config::StrideD{}, {M, N, L});
        auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
        auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));
        ret = initGemmAdapter<GemmLarge, ESF>(&st.gemm_large,
            A_dev, B_dev, SFA_dev, SFB_dev, D_dev, alpha_dev,
            M, N, K, L, stride_A, stride_B, stride_D,
            layout_SFA, layout_SFB, workspace_dev, stream);
    } else {
        using Config = FpGemmSmall;
        auto stride_A = cutlass::make_cute_packed_stride(Config::StrideA{}, {M, K, L});
        auto stride_B = cutlass::make_cute_packed_stride(Config::StrideB{}, {N, K, L});
        auto stride_D = cutlass::make_cute_packed_stride(Config::StrideD{}, {M, N, L});
        auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
        auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));
        ret = initGemmAdapter<GemmSmall, ESF>(&st.gemm_small,
            A_dev, B_dev, SFA_dev, SFB_dev, D_dev, alpha_dev,
            M, N, K, L, stride_A, stride_B, stride_D,
            layout_SFA, layout_SFB, workspace_dev, stream);
    }

    if (ret == 0) st.initialized = true;
    return ret;
#else
    return -1;
#endif
}

extern "C" size_t cgemm_nvfp4_moe_sm120_workspace_size(int N_output, int max_M, int K_hidden, int num_experts) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    // Workspace is used by the cooperative tile scheduler.
    // For these kernel configurations, 4MB is sufficient.
    (void)N_output; (void)max_M; (void)K_hidden; (void)num_experts;
    return 4 * 1024 * 1024;
#else
    return 0;
#endif
}

// CUDA-graph-capturable: only launches the kernel (no cudaFuncSetAttribute)
extern "C" int cgemm_nvfp4_moe_sm120_run(cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    auto& st = s_state;
    if (!st.initialized) { fprintf(stderr, "MoE SM120 GEMM not initialized.\n"); return -1; }

    if (st.use_large_tile) {
        return launchGemm<GemmLarge>(&st.gemm_large, stream);
    } else {
        return launchGemm<GemmSmall>(&st.gemm_small, stream);
    }
#else
    return -1;
#endif
}
