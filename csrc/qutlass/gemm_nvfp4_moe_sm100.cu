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
 *   - BF16 output with alpha epilogue (tensor scales folded in)
 *
 * CUTLASS dimension mapping (same as existing dense/grouped GEMM):
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
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// =========================================================================
// Type definitions
// =========================================================================

using KernelSchedule   = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

using ElementInput = cutlass::float_e2m1_t;
using ElementSF    = cutlass::float_ue4m3_t;
using ElementA     = cutlass::nv_float4_t<ElementInput>;  // activations
using ElementB     = cutlass::nv_float4_t<ElementInput>;  // weights

// CUTLASS A = activations (RowMajor), B = weights (ColumnMajor)
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;

// Output RowMajor: (max_M, N_output) per expert
using ElementC     = cutlass::bfloat16_t;
using ElementD     = cutlass::bfloat16_t;
using LayoutCTag   = cutlass::layout::RowMajor;
using LayoutDTag   = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ElementCompute     = float;

using ArchTag      = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using MmaTileMNK    = Shape<_128, _256, _256>;
using ClusterShapeMNK = Shape<_1, _1, _1>;

// =========================================================================
// CUTLASS kernel type assembly
// =========================================================================
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileMNK, ClusterShapeMNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    EpilogueSchedule,
    cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileMNK, ClusterShapeMNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

// Rank-4 batched problem shape: (M, N, K, L)
using ProblemShape = Shape<int,int,int,int>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Internal type aliases
using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// =========================================================================
// Persistent state (initialized once, reused across calls)
// =========================================================================
struct MoeGemmState {
    bool initialized = false;

    // Fixed dimensions
    int cutlass_M;  // = max_M (max tokens per expert)
    int cutlass_N;  // = N_output (weight output dim)
    int cutlass_K;  // = K_hidden
    int num_experts; // = L (batch dimension)

    // Strides (fixed after init)
    StrideA stride_A;
    StrideB stride_B;
    StrideC stride_C;
    StrideD stride_D;
    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;

    // Hardware info (queried once)
    cutlass::KernelHardwareInfo hw_info;

    // Workspace
    void* workspace_dev = nullptr;
    size_t workspace_size = 0;

    // Persistent GEMM object: avoids stack allocation per call, keeps
    // params_ alive for CUDA graph replay.  init() triggers the one-time
    // cudaFuncSetAttribute call; run() reuses the object.
    Gemm gemm;
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
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
    return size(filter_zeros(layout_SFA)) * sizeof(ElementSF);
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
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));
    return size(filter_zeros(layout_SFB)) * sizeof(ElementSF);
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
    auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    return size(filter_zeros(layout)) * sizeof(ElementSF);
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
    auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    return size(filter_zeros(layout)) * sizeof(ElementSF);
#else
    return 0;
#endif
}

// Initialize the batched GEMM (call once per model configuration).
extern "C" int cgemm_nvfp4_moe_sm100_init(
    int N_output,
    int max_M,
    int K_hidden,
    int num_experts,
    void* workspace_dev,
    size_t workspace_size
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    auto& st = s_state;

    // CUTLASS dimension mapping (M = tokens, N = features, L = experts)
    st.cutlass_M = max_M;
    st.cutlass_N = N_output;
    st.cutlass_K = K_hidden;
    st.num_experts = num_experts;

    int M = st.cutlass_M;
    int N = st.cutlass_N;
    int K = st.cutlass_K;
    int L = num_experts;

    // Compute strides (batched: one set shared by all experts)
    st.stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, L});
    st.stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, L});
    st.stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, L});
    st.stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, L});
    st.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
    st.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));

    // Hardware info
    st.hw_info.device_id = 0;
    st.hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    st.workspace_dev = workspace_dev;
    st.workspace_size = workspace_size;

    // Build arguments with fixed shape and validate
    ProblemShape problem_size{M, N, K, L};

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kBatched,
        problem_size,
        {nullptr, st.stride_A,
         nullptr, st.stride_B,
         nullptr, st.layout_SFA,
         nullptr, st.layout_SFB},
        {{}, nullptr, st.stride_C, nullptr, st.stride_D},
        st.hw_info
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    auto status = st.gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "MoE GEMM can_implement failed: %d\n", (int)status);
        return -1;
    }

    // Initialize the persistent Gemm object: triggers cudaFuncSetAttribute
    // (one-time, not graph-safe) and fills internal params_ with dummy pointers.
    status = st.gemm.initialize(arguments, st.workspace_dev, nullptr);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "MoE GEMM initial initialize failed: %d\n", (int)status);
        return -2;
    }

    st.initialized = true;
    return 0;

#else
    return -1;
#endif
}

// Query workspace size.
extern "C" size_t cgemm_nvfp4_moe_sm100_workspace_size(
    int N_output, int max_M, int K_hidden, int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    int M = max_M, N = N_output, K = K_hidden, L = num_experts;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, L});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, L});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, L});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, L});
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));

    cutlass::KernelHardwareInfo hw;
    hw.device_id = 0;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    ProblemShape problem_size{M, N, K, L};

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kBatched,
        problem_size,
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{}, nullptr, stride_C, nullptr, stride_D},
        hw
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    return Gemm::get_workspace_size(arguments);
#else
    return 0;
#endif
}

// Run the batched GEMM.
// A_dev: activations (num_experts, max_M, K) packed FP4, row-major per expert
// B_dev: weights     (num_experts, N_output, K) packed FP4, col-major per expert
// SFA_dev: activation scale factors (batched swizzled layout)
// SFB_dev: weight scale factors (batched swizzled layout)
// D_dev: output (num_experts, max_M, N_output) BF16, row-major per expert
// alpha_dev: device pointer to float alpha (= act_scale * weight_scale)
//
// Graph-safe: only host-side param building + kernel launch.
// cudaFuncSetAttribute was already called during _init.
extern "C" int cgemm_nvfp4_moe_sm100_run(
    const void* A_dev,        // activations (packed FP4)
    const void* B_dev,        // weights (packed FP4)
    const void* SFA_dev,      // activation scale factors
    const void* SFB_dev,      // weight scale factors
    void* D_dev,              // output (BF16)
    const float* alpha_dev,   // device pointer to alpha scalar
    cudaStream_t stream
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    auto& st = s_state;
    if (!st.initialized) {
        fprintf(stderr, "MoE GEMM not initialized. Call cgemm_nvfp4_moe_sm100_init first.\n");
        return -1;
    }

    int M = st.cutlass_M;
    int N = st.cutlass_N;
    int K = st.cutlass_K;
    int L = st.num_experts;

    ProblemShape problem_size{M, N, K, L};

    using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
    using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;
    using InternalElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kBatched,
        problem_size,
        {reinterpret_cast<const ArrayElementA*>(A_dev), st.stride_A,
         reinterpret_cast<const ArrayElementB*>(B_dev), st.stride_B,
         reinterpret_cast<const InternalElementSF*>(SFA_dev), st.layout_SFA,
         reinterpret_cast<const InternalElementSF*>(SFB_dev), st.layout_SFB},
        {{},
         static_cast<const ElementC*>(nullptr), st.stride_C,
         static_cast<ElementD*>(D_dev), st.stride_D},
        st.hw_info
    };
    // Device-side alpha: if alpha_dev is non-null, kernel reads from device ptr.
    // alpha_ptr takes precedence over the scalar alpha value.
    arguments.epilogue.thread.alpha = 1.0f;  // fallback (ignored when alpha_ptr set)
    arguments.epilogue.thread.alpha_ptr = alpha_dev;
    arguments.epilogue.thread.beta = 0.0f;

    // Rebuild params from arguments (host-side only, no CUDA API calls).
    // cudaFuncSetAttribute was already called during _init on the persistent
    // gemm object, so we call initialize() which is idempotent for the
    // attribute and only updates params_.
    auto status = st.gemm.initialize(arguments, st.workspace_dev, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "MoE GEMM initialize failed: %d\n", (int)status);
        return -2;
    }

    status = st.gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "MoE GEMM run failed: %d\n", (int)status);
        return -3;
    }

    return 0;
#else
    return -1;
#endif
}
