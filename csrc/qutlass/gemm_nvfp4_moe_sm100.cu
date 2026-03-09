/*
 * Batched NVFP4 GEMM for SM_100 (data-center Blackwell: B200/B100) using CUTLASS.
 *
 * Simple batched GEMM: all experts compute max_M × N_output, with L = num_experts.
 * CUDA-graph friendly: fixed shape, no host-side routing, no pointer arrays.
 * Caller pads activations to max_M rows per expert (zero-padded rows produce
 * ignored output) and slices the result to actual token counts.
 *
 * Key design choices:
 *   - TMA-based block-scaled GEMM with auto-selected schedule
 *   - Rank-4 problem shape (M, N, K, L) — standard batched GEMM
 *   - Batched layout: single base pointer + stride per operand
 *   - BF16 output with LinearCombination epilogue (device-side alpha_ptr)
 *   - Two tile sizes: 128x128x256 (M < 512) and 128x256x256 (M >= 512)
 *
 * CUDA Graph Support:
 *   gemm.initialize() calls cudaFuncSetAttribute and is NOT graph-capturable.
 *   gemm.run() only launches the kernel and IS graph-capturable.
 *   The _init function does can_implement + initialize (call once, outside capture).
 *   The _run function calls gemm.run(stream) only (graph-capturable).
 *
 * CUTLASS dimension mapping:
 *   CUTLASS M = max_M   (max tokens per expert, fixed)
 *   CUTLASS N = N_output (weight output dim, fixed)
 *   CUTLASS K = K_hidden (hidden dim)
 *   CUTLASS L = num_experts (batch dimension)
 *
 * Data layout:
 *   A (activations):  (num_experts, max_M, K_hidden) row-major per expert  [TMA load]
 *   B (weights):      (num_experts, N_output, K_hidden) col-major per expert [TMA load]
 *   D (output):       (num_experts, max_M, N_output) row-major
 *   SFA (act scales): batched swizzled layout
 *   SFB (wt scales):  batched swizzled layout
 */

#include <cstdio>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "include/gemm_nvfp4_sm100_types.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// =========================================================================
// Helper: initialize a Gemm adapter object (can_implement + initialize).
// Uses void* to sidestep nvcc's reference binding bug with CUTLASS types.
// The caller must ensure gemm_ptr points to a valid Gemm object.
//
// SM_100 variant: uses kBatched mode and LinearCombination epilogue
// with explicit alpha/beta and device-side alpha_ptr.
// =========================================================================
template <typename Gemm, typename Config>
static int initGemmAdapter(
    void* gemm_ptr,
    const void* A_ptr, const void* B_ptr,
    const void* SFA_ptr, const void* SFB_ptr,
    void* D_ptr, const float* alpha_ptr,
    int M, int N, int K, int L,
    void* workspace, cudaStream_t stream
) {
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementD = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementSF = typename Config::ElementSF;

    auto stride_A = cutlass::make_cute_packed_stride(typename Config::StrideA{}, {M, K, L});
    auto stride_B = cutlass::make_cute_packed_stride(typename Config::StrideB{}, {N, K, L});
    auto stride_C = cutlass::make_cute_packed_stride(typename Config::StrideC{}, {M, N, L});
    auto stride_D = cutlass::make_cute_packed_stride(typename Config::StrideD{}, {M, N, L});
    auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
    auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));

    Gemm* gemm = static_cast<Gemm*>(gemm_ptr);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kBatched,
        {M, N, K, L},
        {static_cast<ElementA const*>(A_ptr), stride_A,
         static_cast<ElementB const*>(B_ptr), stride_B,
         static_cast<ElementSF const*>(SFA_ptr), layout_SFA,
         static_cast<ElementSF const*>(SFB_ptr), layout_SFB},
        {{},
         static_cast<ElementC const*>(nullptr), stride_C,
         static_cast<ElementD*>(D_ptr), stride_D},
    };
    // LinearCombination epilogue: set alpha_ptr for device-side alpha,
    // beta = 0 (no accumulation into C).
    arguments.epilogue.thread.alpha = 1.0f;  // fallback (ignored when alpha_ptr set)
    arguments.epilogue.thread.alpha_ptr = alpha_ptr;
    arguments.epilogue.thread.beta = 0.0f;

    cutlass::Status status;

    status = gemm->can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "MoE GEMM can_implement failed: %d\n", (int)status);
        return -1;
    }

    status = gemm->initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "MoE GEMM initialize failed: %d\n", (int)status);
        return -2;
    }

    return 0;
}

// =========================================================================
// Helper: launch a pre-initialized Gemm adapter (graph-capturable).
// Uses void* to sidestep nvcc's reference binding bug.
// =========================================================================
template <typename Gemm>
static int launchGemm(void* gemm_ptr, cudaStream_t stream) {
    Gemm* gemm = static_cast<Gemm*>(gemm_ptr);
    cutlass::Status status = gemm->run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "MoE GEMM run failed: %d\n", (int)status);
        return -3;
    }
    return 0;
}

// =========================================================================
// Persistent state (initialized once, reused across calls)
// =========================================================================
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

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// =========================================================================
// extern "C" interface
// =========================================================================

// Query SFA (activation scale factor) buffer size in bytes for batched layout.
extern "C" size_t cgemm_nvfp4_moe_sm100_sfa_size(
    int N_output, int max_M, int K_hidden, int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden, L = num_experts;
    auto layout_SFA = FpGemmLarge::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
    return size(filter_zeros(layout_SFA)) * sizeof(ESF);
#else
    return 0;
#endif
}

// Query SFB (weight scale factor) buffer size in bytes for batched layout.
extern "C" size_t cgemm_nvfp4_moe_sm100_sfb_size(
    int N_output, int max_M, int K_hidden, int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden, L = num_experts;
    auto layout_SFB = FpGemmLarge::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));
    return size(filter_zeros(layout_SFB)) * sizeof(ESF);
#else
    return 0;
#endif
}

// Query per-expert SFA size (single expert, L=1).
extern "C" size_t cgemm_nvfp4_moe_sm100_sfa_size_per_expert(
    int N_output, int max_M, int K_hidden
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden;
    auto layout = FpGemmLarge::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    return size(filter_zeros(layout)) * sizeof(ESF);
#else
    return 0;
#endif
}

// Query per-expert SFB size (single expert, L=1).
extern "C" size_t cgemm_nvfp4_moe_sm100_sfb_size_per_expert(
    int N_output, int max_M, int K_hidden
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden;
    auto layout = FpGemmLarge::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    return size(filter_zeros(layout)) * sizeof(ESF);
#else
    return 0;
#endif
}

// Query workspace size.
extern "C" size_t cgemm_nvfp4_moe_sm100_workspace_size(
    int N_output, int max_M, int K_hidden, int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    // Workspace is used by the cooperative tile scheduler.
    // For these kernel configurations, 4MB is sufficient.
    (void)N_output; (void)max_M; (void)K_hidden; (void)num_experts;
    return 4 * 1024 * 1024;
#else
    return 0;
#endif
}

// Initialize the batched GEMM (call once per model configuration).
// All data pointers are baked into the CUTLASS params — the caller writes
// new data into the same buffers and calls _run() to launch the kernel.
extern "C" int cgemm_nvfp4_moe_sm100_init(
    int N_output,
    int max_M,
    int K_hidden,
    int num_experts,
    const void* A_dev,
    const void* B_dev,
    const void* SFA_dev,
    const void* SFB_dev,
    void* D_dev,
    const float* alpha_dev,
    void* workspace_dev,
    size_t workspace_size,
    cudaStream_t stream
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    auto& st = s_state;

    st.cutlass_M = max_M;
    st.cutlass_N = N_output;
    st.cutlass_K = K_hidden;
    st.num_experts = num_experts;
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
        ret = initGemmAdapter<GemmLarge, FpGemmLarge>(
            &st.gemm_large,
            A_dev, B_dev, SFA_dev, SFB_dev, D_dev, alpha_dev,
            M, N, K, L, workspace_dev, stream);
    } else {
        ret = initGemmAdapter<GemmSmall, FpGemmSmall>(
            &st.gemm_small,
            A_dev, B_dev, SFA_dev, SFB_dev, D_dev, alpha_dev,
            M, N, K, L, workspace_dev, stream);
    }

    if (ret == 0) st.initialized = true;
    return ret;

#else
    return -1;
#endif
}

// CUDA-graph-capturable: only launches the kernel (no cudaFuncSetAttribute).
// All data pointers were baked during _init — caller writes new data into
// the same buffers, then calls this to launch the GEMM.
extern "C" int cgemm_nvfp4_moe_sm100_run(cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    auto& st = s_state;
    if (!st.initialized) {
        fprintf(stderr, "MoE GEMM not initialized. Call cgemm_nvfp4_moe_sm100_init first.\n");
        return -1;
    }

    if (st.use_large_tile) {
        return launchGemm<GemmLarge>(&st.gemm_large, stream);
    } else {
        return launchGemm<GemmSmall>(&st.gemm_small, stream);
    }
#else
    return -1;
#endif
}
