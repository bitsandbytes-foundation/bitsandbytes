/*
 * Batched NVFP4 GEMM for SM_120 (consumer Blackwell: RTX 5090/5080) using CUTLASS.
 *
 * Simple batched GEMM: all experts compute max_M × N_output, with L = num_experts.
 * CUDA-graph friendly: fixed shape, no host-side routing, no pointer arrays.
 * Caller pads activations to max_M rows per expert (zero-padded rows produce
 * ignored output) and slices the result to actual token counts.
 *
 * Key design choices:
 *   - Block-scaled GEMM with auto-selected schedule (SM_120)
 *   - Rank-4 problem shape (M, N, K, L) — standard batched GEMM
 *   - Batched layout: single base pointer + stride per operand
 *   - BF16 output with alpha epilogue (tensor scales folded in via device ptr)
 *   - Tile shape selected by max_M: 128x128x128 (small) or 256x128x128 (large)
 *
 * CUTLASS dimension mapping:
 *   CUTLASS M = max_M   (max tokens per expert, fixed)
 *   CUTLASS N = N_output (weight output dim, fixed)
 *   CUTLASS K = K_hidden (hidden dim)
 *   CUTLASS L = num_experts (batch dimension)
 *
 * Data layout:
 *   A (activations):  (num_experts, max_M, K_hidden) row-major per expert
 *   B (weights):      (num_experts, N_output, K_hidden) col-major per expert
 *   D (output):       (num_experts, max_M, N_output) row-major
 *   SFA (act scales): batched swizzled layout
 *   SFB (wt scales):  batched swizzled layout
 *
 * IMPORTANT: Uses the shared FpGemm type from gemm_nvfp4_sm120_types.h to ensure
 * identical GemmKernel types with the dense kernel, which is required for proper
 * CUDA device kernel registration.
 */

#include <cstdio>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "include/gemm_nvfp4_sm120_types.h"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

// =========================================================================
// Persistent state (initialized once, reused across calls)
// =========================================================================
struct MoeGemmState {
    bool initialized = false;
    bool use_large_tile = false;

    int cutlass_M;   // = max_M
    int cutlass_N;   // = N_output
    int cutlass_K;   // = K_hidden
    int num_experts; // = L

    // Strides for both tile configs (only one set is active)
    FpGemmSmall::StrideA stride_A_small;
    FpGemmSmall::StrideB stride_B_small;
    FpGemmSmall::StrideC stride_C_small;
    FpGemmSmall::StrideD stride_D_small;
    FpGemmSmall::LayoutSFA layout_SFA_small;
    FpGemmSmall::LayoutSFB layout_SFB_small;

    FpGemmLarge::StrideA stride_A_large;
    FpGemmLarge::StrideB stride_B_large;
    FpGemmLarge::StrideC stride_C_large;
    FpGemmLarge::StrideD stride_D_large;
    FpGemmLarge::LayoutSFA layout_SFA_large;
    FpGemmLarge::LayoutSFB layout_SFB_large;

    cutlass::KernelHardwareInfo hw_info;

    void* workspace_dev = nullptr;
    size_t workspace_size = 0;
};

static MoeGemmState s_state;

// =========================================================================
// Helper: run batched GEMM for a specific tile configuration
// =========================================================================
template <typename Config>
static int runBatchedGemm(
    const void* A_dev, const void* B_dev,
    const void* SFA_dev, const void* SFB_dev,
    void* D_dev,
    const float* alpha_dev,
    int M, int N, int K, int L,
    typename Config::StrideA& stride_A,
    typename Config::StrideB& stride_B,
    typename Config::StrideC& stride_C,
    typename Config::StrideD& stride_D,
    typename Config::LayoutSFA& layout_SFA,
    typename Config::LayoutSFB& layout_SFB,
    void* workspace, size_t workspace_size,
    cutlass::KernelHardwareInfo const& hw_info,
    cudaStream_t stream
) {
    using Gemm = typename Config::Gemm;
    using ElementD = typename Config::ElementD;

    // Use internal array element types from the collective mainloop
    using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
    using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;
    using InternalElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        {reinterpret_cast<const ArrayElementA*>(A_dev), stride_A,
         reinterpret_cast<const ArrayElementB*>(B_dev), stride_B,
         reinterpret_cast<const InternalElementSF*>(SFA_dev), layout_SFA,
         reinterpret_cast<const InternalElementSF*>(SFB_dev), layout_SFB},
        {{},
         static_cast<ElementD const*>(D_dev), stride_C,
         static_cast<ElementD*>(D_dev), stride_D},
        hw_info
    };
    // SM_120 uses device pointer for alpha; beta=0 means no C residual
    arguments.epilogue.thread.alpha_ptr = alpha_dev;
    arguments.epilogue.thread.beta = 0.0f;

    Gemm gemm;

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "MoE SM120 GEMM can_implement failed: %d\n", (int)status);
        return -1;
    }

    status = gemm.initialize(arguments, workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "MoE SM120 GEMM initialize failed: %d\n", (int)status);
        return -2;
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "MoE SM120 GEMM run failed: %d\n", (int)status);
        return -3;
    }

    return 0;
}

// =========================================================================
// Helper: query workspace size for a specific tile configuration
// =========================================================================
template <typename Config>
static size_t queryWorkspaceSize(int M, int N, int K, int L,
    typename Config::StrideA& stride_A,
    typename Config::StrideB& stride_B,
    typename Config::StrideC& stride_C,
    typename Config::StrideD& stride_D,
    typename Config::LayoutSFA& layout_SFA,
    typename Config::LayoutSFB& layout_SFB,
    cutlass::KernelHardwareInfo const& hw_info
) {
    using Gemm = typename Config::Gemm;
    using ElementD = typename Config::ElementD;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        {nullptr, stride_A, nullptr, stride_B,
         nullptr, layout_SFA, nullptr, layout_SFB},
        {{},
         static_cast<ElementD const*>(nullptr), stride_C,
         static_cast<ElementD*>(nullptr), stride_D},
        hw_info
    };
    arguments.epilogue.thread.alpha_ptr = nullptr;

    Gemm gemm;
    return gemm.get_workspace_size(arguments);
}

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED

// =========================================================================
// extern "C" interface
// =========================================================================

extern "C" size_t cgemm_nvfp4_moe_sm120_sfa_size(
    int N_output, int max_M, int K_hidden, int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden, L = num_experts;
    auto layout_SFA = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, L));
    return size(filter_zeros(layout_SFA)) * sizeof(FpGemmSmall::ElementSF);
#else
    return 0;
#endif
}

extern "C" size_t cgemm_nvfp4_moe_sm120_sfb_size(
    int N_output, int max_M, int K_hidden, int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden, L = num_experts;
    auto layout_SFB = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, L));
    return size(filter_zeros(layout_SFB)) * sizeof(FpGemmSmall::ElementSF);
#else
    return 0;
#endif
}

extern "C" size_t cgemm_nvfp4_moe_sm120_sfa_size_per_expert(
    int N_output, int max_M, int K_hidden
) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden;
    auto layout = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    return size(filter_zeros(layout)) * sizeof(FpGemmSmall::ElementSF);
#else
    return 0;
#endif
}

extern "C" size_t cgemm_nvfp4_moe_sm120_sfb_size_per_expert(
    int N_output, int max_M, int K_hidden
) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden;
    auto layout = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));
    return size(filter_zeros(layout)) * sizeof(FpGemmSmall::ElementSF);
#else
    return 0;
#endif
}

extern "C" int cgemm_nvfp4_moe_sm120_init(
    int N_output,
    int max_M,
    int K_hidden,
    int num_experts,
    void* workspace_dev,
    size_t workspace_size
) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    auto& st = s_state;

    st.cutlass_M = max_M;
    st.cutlass_N = N_output;
    st.cutlass_K = K_hidden;
    st.num_experts = num_experts;
    st.use_large_tile = (max_M >= 512);

    int M = max_M, N = N_output, K = K_hidden, L = num_experts;

    st.stride_A_small = cutlass::make_cute_packed_stride(FpGemmSmall::StrideA{}, {M, K, L});
    st.stride_B_small = cutlass::make_cute_packed_stride(FpGemmSmall::StrideB{}, {N, K, L});
    st.stride_C_small = cutlass::make_cute_packed_stride(FpGemmSmall::StrideC{}, {M, N, L});
    st.stride_D_small = cutlass::make_cute_packed_stride(FpGemmSmall::StrideD{}, {M, N, L});
    st.layout_SFA_small = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, L));
    st.layout_SFB_small = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, L));

    st.stride_A_large = cutlass::make_cute_packed_stride(FpGemmLarge::StrideA{}, {M, K, L});
    st.stride_B_large = cutlass::make_cute_packed_stride(FpGemmLarge::StrideB{}, {N, K, L});
    st.stride_C_large = cutlass::make_cute_packed_stride(FpGemmLarge::StrideC{}, {M, N, L});
    st.stride_D_large = cutlass::make_cute_packed_stride(FpGemmLarge::StrideD{}, {M, N, L});
    st.layout_SFA_large = FpGemmLarge::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, L));
    st.layout_SFB_large = FpGemmLarge::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, L));

    st.hw_info.device_id = 0;
    st.hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    st.workspace_dev = workspace_dev;
    st.workspace_size = workspace_size;
    st.initialized = true;
    return 0;

#else
    return -1;
#endif
}

extern "C" size_t cgemm_nvfp4_moe_sm120_workspace_size(
    int N_output, int max_M, int K_hidden, int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden, L = num_experts;

    auto stride_A_s = cutlass::make_cute_packed_stride(FpGemmSmall::StrideA{}, {M, K, L});
    auto stride_B_s = cutlass::make_cute_packed_stride(FpGemmSmall::StrideB{}, {N, K, L});
    auto stride_C_s = cutlass::make_cute_packed_stride(FpGemmSmall::StrideC{}, {M, N, L});
    auto stride_D_s = cutlass::make_cute_packed_stride(FpGemmSmall::StrideD{}, {M, N, L});
    auto layout_SFA_s = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, L));
    auto layout_SFB_s = FpGemmSmall::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, L));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    size_t ws_small = queryWorkspaceSize<FpGemmSmall>(M, N, K, L,
        stride_A_s, stride_B_s, stride_C_s, stride_D_s,
        layout_SFA_s, layout_SFB_s, hw_info);

    auto stride_A_l = cutlass::make_cute_packed_stride(FpGemmLarge::StrideA{}, {M, K, L});
    auto stride_B_l = cutlass::make_cute_packed_stride(FpGemmLarge::StrideB{}, {N, K, L});
    auto stride_C_l = cutlass::make_cute_packed_stride(FpGemmLarge::StrideC{}, {M, N, L});
    auto stride_D_l = cutlass::make_cute_packed_stride(FpGemmLarge::StrideD{}, {M, N, L});
    auto layout_SFA_l = FpGemmLarge::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, L));
    auto layout_SFB_l = FpGemmLarge::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, L));

    size_t ws_large = queryWorkspaceSize<FpGemmLarge>(M, N, K, L,
        stride_A_l, stride_B_l, stride_C_l, stride_D_l,
        layout_SFA_l, layout_SFB_l, hw_info);

    return (ws_small > ws_large) ? ws_small : ws_large;
#else
    return 0;
#endif
}

// Run the batched GEMM.
// alpha_dev: device pointer to float scalar (product of tensor scales)
extern "C" int cgemm_nvfp4_moe_sm120_run(
    const void* A_dev,
    const void* B_dev,
    const void* SFA_dev,
    const void* SFB_dev,
    void* D_dev,
    const float* alpha_dev,
    cudaStream_t stream
) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    auto& st = s_state;
    if (!st.initialized) {
        fprintf(stderr, "MoE SM120 GEMM not initialized.\n");
        return -1;
    }

    int M = st.cutlass_M, N = st.cutlass_N, K = st.cutlass_K, L = st.num_experts;

    if (st.use_large_tile) {
        return runBatchedGemm<FpGemmLarge>(
            A_dev, B_dev, SFA_dev, SFB_dev, D_dev, alpha_dev,
            M, N, K, L,
            st.stride_A_large, st.stride_B_large,
            st.stride_C_large, st.stride_D_large,
            st.layout_SFA_large, st.layout_SFB_large,
            st.workspace_dev, st.workspace_size,
            st.hw_info, stream);
    } else {
        return runBatchedGemm<FpGemmSmall>(
            A_dev, B_dev, SFA_dev, SFB_dev, D_dev, alpha_dev,
            M, N, K, L,
            st.stride_A_small, st.stride_B_small,
            st.stride_C_small, st.stride_D_small,
            st.layout_SFA_small, st.layout_SFB_small,
            st.workspace_dev, st.workspace_size,
            st.hw_info, stream);
    }

#else
    return -1;
#endif
}
