/*
 * Shared CUTLASS type definitions for SM_100 block-scaled FP4 GEMM.
 *
 * Both the dense and batched MoE kernels use these common type aliases
 * to ensure they instantiate identical GemmKernel types, which is required
 * for proper CUDA device kernel registration across translation units.
 *
 * SM_100 uses LinearCombination epilogue (supports device-side alpha_ptr)
 * and two tile configurations for adaptive tile selection based on M.
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// =========================================================================
// FpGemm_SM100: CUTLASS GEMM template for block-scaled FP4 operations (SM_100)
//
// Key difference from SM_120: uses LinearCombination epilogue for explicit
// alpha/beta fusion with device-side alpha_ptr support.
// =========================================================================
template <
    typename MmaTileShape,
    typename ClusterShape>
struct FpGemm_SM100 {
    // Element types
    using ElementInput = cutlass::float_e2m1_t;
    using ElementA     = cutlass::nv_float4_t<ElementInput>;  // activations
    using ElementB     = cutlass::nv_float4_t<ElementInput>;  // weights
    using ElementC     = cutlass::bfloat16_t;
    using ElementD     = cutlass::bfloat16_t;
    using ElementSF    = cutlass::float_ue4m3_t;
    using ElementAccumulator = float;
    using ElementCompute     = float;

    // Layouts
    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;

    // Alignments
    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using ArchTag       = cutlass::arch::Sm100;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // Epilogue with LinearCombination (device-side alpha_ptr support)
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>
    >::CollectiveOp;

    // Mainloop
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // Kernel and adapter
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Derived type aliases
    using StrideA  = typename GemmKernel::StrideA;
    using StrideB  = typename GemmKernel::StrideB;
    using StrideC  = typename GemmKernel::StrideC;
    using StrideD  = typename GemmKernel::StrideD;
    using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;
    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

// Cluster shape (1x1x1 for all SM_100 block-scaled configs)
using ClusterShape_SM100 = Shape<_1, _1, _1>;

// Small tile (128x128x256) — for M < 512 (decode)
using FpGemmSmall = FpGemm_SM100<Shape<_128, _128, _256>, ClusterShape_SM100>;

// Large tile (128x256x256) — for M >= 512 (prefill), same as current single tile
using FpGemmLarge = FpGemm_SM100<Shape<_128, _256, _256>, ClusterShape_SM100>;

// Convenience aliases used by gemm_nvfp4_moe_sm100.cu
using GemmSmall = FpGemmSmall::Gemm;
using GemmLarge = FpGemmLarge::Gemm;
using ESF       = cutlass::float_ue4m3_t;

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
