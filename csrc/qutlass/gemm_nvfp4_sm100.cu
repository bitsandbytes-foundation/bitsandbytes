/*
 * NVFP4 GEMM for SM_100 (data-center Blackwell: B200/B100) using CUTLASS.
 *
 * Derived from the SM_120 variant and CUTLASS example 72a.
 * Uses block-scaled FP4 tensor core MMA (tcgen05.mma.blockscaled) on SM_100a.
 *
 * SM_100 vs SM_120 differences:
 *   - ArchTag: cutlass::arch::Sm100 (not Sm120)
 *   - Supports larger tile shapes (256x256x256) from the hardware MMA
 *   - Supports multi-CTA clusters (future optimization)
 *   - May require workspace allocation for certain cluster configs
 */

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// =========================================================================
// FpGemm: CUTLASS GEMM template for block-scaled FP4 operations (SM_100)
// =========================================================================
template <
    typename MmaTileShape, typename ClusterShape, typename ArchTag, typename ElementA,
    typename LayoutATag, int AlignmentA, typename ElementB, typename LayoutBTag, int AlignmentB>
struct FpGemmSm100 {
    using ElementD = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementAccumulator = float;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // SM_100: epilogue and mainloop use the same MmaTileShape
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB, ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// =========================================================================
// runGemmSm100: raw-pointer GEMM runner with workspace support
// =========================================================================
template <typename Gemm, typename ScaleType>
static int runGemmSm100(
    void* D_ptr, const void* A_ptr, const void* B_ptr, const void* A_sf_ptr, const void* B_sf_ptr,
    const float* alpha_ptr, int M, int N, int K, cudaStream_t stream
) {
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementD = typename Gemm::ElementD;
    using ElementSFA = ScaleType;
    using ElementSFB = ScaleType;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {static_cast<ElementA const*>(A_ptr), stride_A, static_cast<ElementB const*>(B_ptr), stride_B,
         static_cast<ElementSFA const*>(A_sf_ptr), layout_SFA, static_cast<ElementSFB const*>(B_sf_ptr), layout_SFB},
        {{}, static_cast<ElementD const*>(D_ptr), stride_D, static_cast<ElementD*>(D_ptr), stride_D}
    };
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = alpha_ptr;

    Gemm gemm;

    // SM_100 may need workspace for certain cluster/scheduler configs
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        cudaError_t alloc_err = cudaMallocAsync(&workspace_ptr, workspace_size, stream);
        if (alloc_err != cudaSuccess) {
            fprintf(stderr, "CUTLASS SM100 workspace allocation failed: %s\n", cudaGetErrorString(alloc_err));
            return -4;
        }
    }

    cutlass::Status status;

    status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS SM100 GEMM can_implement failed: %d\n", (int)status);
        if (workspace_ptr) cudaFreeAsync(workspace_ptr, stream);
        return -1;
    }

    status = gemm.initialize(arguments, workspace_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS SM100 GEMM initialize failed: %d\n", (int)status);
        if (workspace_ptr) cudaFreeAsync(workspace_ptr, stream);
        return -2;
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS SM100 GEMM run failed: %d\n", (int)status);
        if (workspace_ptr) cudaFreeAsync(workspace_ptr, stream);
        return -3;
    }

    if (workspace_ptr) cudaFreeAsync(workspace_ptr, stream);
    return 0;
}

// =========================================================================
// extern "C" interface for bitsandbytes (SM_100)
// =========================================================================

extern "C" void cgemm_nvfp4_cutlass_sm100(
    const void* A,       // packed E2M1 data, shape (M, K/2), row-major
    const void* B,       // packed E2M1 data, shape (N, K/2), col-major (TN)
    const void* SFA,     // E4M3 block scales for A, in to_blocked() layout
    const void* SFB,     // E4M3 block scales for B, in to_blocked() layout
    void* D,             // BF16 output, shape (M, N), row-major
    int M, int N, int K, // logical dimensions (K is unpacked)
    const float* alpha,  // epilogue scale factor (device pointer)
    cudaStream_t stream
) {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;

    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;

    using ArchTag = cutlass::arch::Sm100;

    // SM_100 block-scaled MMA constraint: per-CTA M-mode must be 128.
    // With ClusterShape 1x1x1, MmaTileShape M must be 128.
    // With ClusterShape 2x1x1, MmaTileShape M can be 256 (per-CTA = 128).
    //
    // For large M: use 256x256x256 tile with 2x4x1 cluster (NVIDIA example 72a).
    // For small M: use 128x128x256 tile with 1x1x1 cluster.
    if (M >= 512) {
        using ClusterShape = Shape<_2, _4, _1>;
        using MmaTileShape = Shape<_256, _256, _256>;
        runGemmSm100<
            FpGemmSm100<
                MmaTileShape, ClusterShape, ArchTag, ElementA, LayoutATag, AlignmentA, ElementB,
                LayoutBTag, AlignmentB>::Gemm,
            cutlass::float_ue4m3_t>(D, A, B, SFA, SFB, alpha, M, N, K, stream);
    } else {
        using ClusterShape = Shape<_1, _1, _1>;
        using MmaTileShape = Shape<_128, _128, _256>;
        runGemmSm100<
            FpGemmSm100<
                MmaTileShape, ClusterShape, ArchTag, ElementA, LayoutATag, AlignmentA, ElementB,
                LayoutBTag, AlignmentB>::Gemm,
            cutlass::float_ue4m3_t>(D, A, B, SFA, SFB, alpha, M, N, K, stream);
    }
}
