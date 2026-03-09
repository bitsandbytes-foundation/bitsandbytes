/*
 * Grouped NVFP4 GEMM for SM_100 (data-center Blackwell: B200/B100) using CUTLASS.
 *
 * Fuses all expert GEMMs into a single kernel launch for MoE inference.
 * Uses CUTLASS 3 grouped GEMM with block-scaled FP4 tensor cores on SM_100a.
 *
 * Based on CUTLASS example 75 (75_blackwell_grouped_gemm_block_scaled).
 *
 * Performance design:
 *   - Zero cudaMalloc/cudaFree per call (all buffers pre-allocated by Python)
 *   - Host pointer arrays passed directly (no device-to-host copies)
 *   - Single cudaMemcpyAsync for metadata (problem sizes, strides, layouts, pointers)
 *   - Workspace and metadata buffers reused across calls
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
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// =========================================================================
// Type definitions
// =========================================================================
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

using ElementInput = cutlass::float_e2m1_t;
using ElementSF    = cutlass::float_ue4m3_t;
using ElementA     = cutlass::nv_float4_t<ElementInput>;
using ElementB     = cutlass::nv_float4_t<ElementInput>;
using LayoutA      = cutlass::layout::RowMajor;
using LayoutB      = cutlass::layout::ColumnMajor;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;

using ElementC     = cutlass::bfloat16_t;
using ElementD     = cutlass::bfloat16_t;
using LayoutC      = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;

using ArchTag      = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Runtime cluster shape for grouped GEMM
using ClusterShape = Shape<int32_t, int32_t, _1>;

// 1SM config: 128x256x256 tile (safe for all problem sizes)
using MmaTileShape     = Shape<_128, _256, _256>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

// =========================================================================
// CUTLASS kernel type assembly
// =========================================================================
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    Shape<_128, _64>,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutC *, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Internal type aliases
using StrideA   = typename Gemm::GemmKernel::InternalStrideA;
using StrideB   = typename Gemm::GemmKernel::InternalStrideB;
using StrideC   = typename Gemm::GemmKernel::InternalStrideC;
using StrideD   = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;
using InternalElementSF = typename Gemm::GemmKernel::ElementSF;

// =========================================================================
// Metadata buffer layout
// =========================================================================
// All per-expert arrays packed contiguously into a single device buffer.
// Offsets computed from type sizes and alignment.

struct MetaOffsets {
    size_t problem_sizes;  // ProblemShape::UnderlyingProblemShape[N]
    size_t stride_A;       // StrideA[N]
    size_t stride_B;       // StrideB[N]
    size_t stride_C;       // StrideC[N]
    size_t stride_D;       // StrideD[N]
    size_t layout_SFA;     // LayoutSFA[N]
    size_t layout_SFB;     // LayoutSFB[N]
    size_t ptr_A;          // const ArrayElementA*[N]
    size_t ptr_B;          // const ArrayElementB*[N]
    size_t ptr_SFA;        // const InternalElementSF*[N]
    size_t ptr_SFB;        // const InternalElementSF*[N]
    size_t ptr_C;          // const ElementC*[N]
    size_t ptr_D;          // ElementD*[N]
    size_t total;
};

static MetaOffsets compute_offsets(int n) {
    MetaOffsets o;
    size_t off = 0;
    auto align = [&](size_t a) { off = (off + a - 1) & ~(a - 1); };

    align(16); o.problem_sizes = off; off += n * sizeof(typename ProblemShape::UnderlyingProblemShape);
    align(16); o.stride_A = off;      off += n * sizeof(StrideA);
    align(16); o.stride_B = off;      off += n * sizeof(StrideB);
    align(16); o.stride_C = off;      off += n * sizeof(StrideC);
    align(16); o.stride_D = off;      off += n * sizeof(StrideD);
    align(16); o.layout_SFA = off;    off += n * sizeof(LayoutSFA);
    align(16); o.layout_SFB = off;    off += n * sizeof(LayoutSFB);
    align(8);  o.ptr_A = off;         off += n * sizeof(const ArrayElementA*);
    align(8);  o.ptr_B = off;         off += n * sizeof(const ArrayElementB*);
    align(8);  o.ptr_SFA = off;       off += n * sizeof(const InternalElementSF*);
    align(8);  o.ptr_SFB = off;       off += n * sizeof(const InternalElementSF*);
    align(8);  o.ptr_C = off;         off += n * sizeof(const ElementC*);
    align(8);  o.ptr_D = off;         off += n * sizeof(ElementD*);
    align(16); o.total = off;
    return o;
}

// Cached SM count (queried once)
static int s_sm_count = 0;

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// =========================================================================
// Batched swizzle kernel (defined in scale_reorder.cu)
// =========================================================================
extern __global__ void kScaleToBlockedBatched(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const int* __restrict__ expert_row_offsets,
    const int* __restrict__ expert_M,
    const int* __restrict__ expert_out_offsets,
    int W,
    int num_experts
);

// =========================================================================
// extern "C" interface
// =========================================================================

extern "C" size_t cgemm_nvfp4_grouped_sm100_meta_size(int max_experts) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return compute_offsets(max_experts).total;
#else
    return 0;
#endif
}

extern "C" size_t cgemm_nvfp4_grouped_sm100_workspace_size(
    const int* host_M, int N, int K, int num_experts
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (num_experts <= 0) return 0;

    // Build minimal arguments to query workspace size
    std::vector<typename ProblemShape::UnderlyingProblemShape> ps(num_experts);
    for (int i = 0; i < num_experts; i++) ps[i] = {host_M[i], N, K};

    if (s_sm_count == 0)
        s_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = s_sm_count;
    hw_info.cluster_shape = dim3(1, 1, 1);
    hw_info.cluster_shape_fallback = dim3(1, 1, 1);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts, nullptr, ps.data()},
        {}, {}, hw_info, {}
    };
    return Gemm::get_workspace_size(arguments);
#else
    return 0;
#endif
}

extern "C" void cgemm_nvfp4_grouped_cutlass_sm100(
    // Host arrays (directly from Python ctypes — no device-to-host copies)
    const int64_t* host_ptr_A,    // [num_experts] raw pointer values
    const int64_t* host_ptr_B,
    const int64_t* host_ptr_SFA,
    const int64_t* host_ptr_SFB,
    const int64_t* host_ptr_D,
    const int* host_M_per_expert,
    int N, int K,
    int num_experts,
    float alpha_val,
    // Pre-allocated device buffers (from PyTorch tensors)
    void* metadata_dev,           // device buffer for all per-expert arrays
    void* workspace_dev,          // device buffer for CUTLASS workspace
    size_t workspace_size,
    cudaStream_t stream
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (num_experts <= 0) return;

    auto offsets = compute_offsets(num_experts);

    // Build all metadata on host into a single contiguous buffer
    std::vector<uint8_t> host_buf(offsets.total, 0);

    auto* ps = reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(host_buf.data() + offsets.problem_sizes);
    auto* sa = reinterpret_cast<StrideA*>(host_buf.data() + offsets.stride_A);
    auto* sb = reinterpret_cast<StrideB*>(host_buf.data() + offsets.stride_B);
    auto* sc = reinterpret_cast<StrideC*>(host_buf.data() + offsets.stride_C);
    auto* sd = reinterpret_cast<StrideD*>(host_buf.data() + offsets.stride_D);
    auto* lsfa = reinterpret_cast<LayoutSFA*>(host_buf.data() + offsets.layout_SFA);
    auto* lsfb = reinterpret_cast<LayoutSFB*>(host_buf.data() + offsets.layout_SFB);
    auto* pa = reinterpret_cast<const ArrayElementA**>(host_buf.data() + offsets.ptr_A);
    auto* pb = reinterpret_cast<const ArrayElementB**>(host_buf.data() + offsets.ptr_B);
    auto* psfa = reinterpret_cast<const InternalElementSF**>(host_buf.data() + offsets.ptr_SFA);
    auto* psfb = reinterpret_cast<const InternalElementSF**>(host_buf.data() + offsets.ptr_SFB);
    auto* pc = reinterpret_cast<const ElementC**>(host_buf.data() + offsets.ptr_C);
    auto* pd = reinterpret_cast<ElementD**>(host_buf.data() + offsets.ptr_D);

    for (int i = 0; i < num_experts; i++) {
        int M = host_M_per_expert[i];
        ps[i] = {M, N, K};
        sa[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        sb[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        sc[i] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        sd[i] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
        lsfa[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        lsfb[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

        pa[i]   = reinterpret_cast<const ArrayElementA*>(host_ptr_A[i]);
        pb[i]   = reinterpret_cast<const ArrayElementB*>(host_ptr_B[i]);
        psfa[i] = reinterpret_cast<const InternalElementSF*>(host_ptr_SFA[i]);
        psfb[i] = reinterpret_cast<const InternalElementSF*>(host_ptr_SFB[i]);
        pc[i]   = nullptr;
        pd[i]   = reinterpret_cast<ElementD*>(host_ptr_D[i]);
    }

    // Single memcpy: host metadata buffer → pre-allocated device buffer
    cudaMemcpyAsync(metadata_dev, host_buf.data(), offsets.total,
                    cudaMemcpyHostToDevice, stream);

    // Build CUTLASS arguments pointing into the device metadata buffer
    auto dev_at = [&](size_t off) { return static_cast<uint8_t*>(metadata_dev) + off; };

    if (s_sm_count == 0)
        s_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = s_sm_count;
    hw_info.cluster_shape = dim3(1, 1, 1);
    hw_info.cluster_shape_fallback = dim3(1, 1, 1);

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = alpha_val;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 0};
    fusion_args.beta = 0.0f;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dBeta = {_0{}, _0{}, 0};

    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts,
         reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(dev_at(offsets.problem_sizes)),
         ps},  // host problem sizes for initialize()
        {reinterpret_cast<const ArrayElementA**>(dev_at(offsets.ptr_A)),
         reinterpret_cast<StrideA*>(dev_at(offsets.stride_A)),
         reinterpret_cast<const ArrayElementB**>(dev_at(offsets.ptr_B)),
         reinterpret_cast<StrideB*>(dev_at(offsets.stride_B)),
         reinterpret_cast<const InternalElementSF**>(dev_at(offsets.ptr_SFA)),
         reinterpret_cast<LayoutSFA*>(dev_at(offsets.layout_SFA)),
         reinterpret_cast<const InternalElementSF**>(dev_at(offsets.ptr_SFB)),
         reinterpret_cast<LayoutSFB*>(dev_at(offsets.layout_SFB))},
        {fusion_args,
         reinterpret_cast<const ElementC**>(dev_at(offsets.ptr_C)),
         reinterpret_cast<StrideC*>(dev_at(offsets.stride_C)),
         reinterpret_cast<ElementD**>(dev_at(offsets.ptr_D)),
         reinterpret_cast<StrideD*>(dev_at(offsets.stride_D))},
        hw_info, scheduler
    };

    Gemm gemm;
    cutlass::Status status;

    status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS SM100 grouped GEMM can_implement failed: %d\n", (int)status);
        return;
    }

    status = gemm.initialize(arguments, workspace_dev, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS SM100 grouped GEMM initialize failed: %d\n", (int)status);
        return;
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS SM100 grouped GEMM run failed: %d\n", (int)status);
    }

#else
    fprintf(stderr, "SM100 grouped GEMM not supported on this architecture\n");
#endif
}

// =========================================================================
// Fused dispatch: SFA swizzle + grouped GEMM in one C call
// =========================================================================
// Takes raw base pointers and host-side expert offsets. Computes all byte
// offsets, launches the batched SFA swizzle, builds CUTLASS metadata, and
// launches the grouped GEMM. Zero Python overhead beyond the single ctypes call.
//
// Buffer layout for sfa_swizzle_meta (3 * num_experts * sizeof(int)):
//   [expert_row_offsets | expert_M | expert_sfa_out_offsets]

extern "C" void cgemm_nvfp4_grouped_sm100_fused(
    // Base pointers to concatenated device tensors
    const void* A_concat,        // packed FP4 activations, (total_tokens, K/2)
    const void* B_all,           // packed FP4 weights, (num_experts * N, K/2)
    const void* SFA_rowmajor,    // row-major activation scales, (total_tokens * scale_W)
    const void* SFB_per_expert,  // per-expert pre-swizzled weight scales
    void* D_concat,              // BF16 output, (total_tokens, N)
    // Expert offsets on HOST
    const int* host_offsets,     // [num_experts + 1] cumulative token offsets
    int N, int K,
    int num_experts,
    float alpha_val,
    // Pre-allocated device buffers
    void* sfa_swizzle_out,       // output buffer for batched SFA swizzle
    void* sfa_swizzle_meta,      // device buffer for swizzle metadata (3 * num_experts * 4 bytes)
    void* gemm_metadata_dev,     // device buffer for CUTLASS metadata
    void* workspace_dev,         // device buffer for CUTLASS workspace
    size_t workspace_size,
    cudaStream_t stream
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (num_experts <= 0) return;

    const int half_K = K / 2;
    const int scale_W = K / 16;
    const int n_col_blocks = (scale_W + 3) / 4;
    const int total_tokens = host_offsets[num_experts];

    // N-dimension constants for SFB
    const int n_sfb_row_blocks = (N + 127) / 128;
    const int sfb_per_expert_bytes = n_sfb_row_blocks * n_col_blocks * 512;

    // -----------------------------------------------------------------
    // Step 1: Build SFA swizzle metadata on host and upload
    // -----------------------------------------------------------------
    // Stack-allocate for small num_experts (typical: 8-64)
    int sfa_meta_host[3 * 64];  // supports up to 64 experts on stack
    int* meta_buf = (num_experts <= 64) ? sfa_meta_host : new int[3 * num_experts];

    int* row_offsets = meta_buf;
    int* M_values = meta_buf + num_experts;
    int* out_offsets = meta_buf + 2 * num_experts;

    int max_row_blocks = 0;
    int sfa_out_offset = 0;

    // Also build SFA output byte offsets for CUTLASS pointers
    int sfa_expert_out_offsets[64];
    int* sfa_offs = (num_experts <= 64) ? sfa_expert_out_offsets : new int[num_experts];

    for (int e = 0; e < num_experts; e++) {
        int M_e = host_offsets[e + 1] - host_offsets[e];
        row_offsets[e] = host_offsets[e];
        M_values[e] = M_e;
        out_offsets[e] = sfa_out_offset;
        sfa_offs[e] = sfa_out_offset;

        int n_row_blocks_e = (M_e > 0) ? (M_e + 127) / 128 : 0;
        if (n_row_blocks_e > max_row_blocks) max_row_blocks = n_row_blocks_e;
        sfa_out_offset += n_row_blocks_e * n_col_blocks * 512;
    }

    // Upload swizzle metadata (small: 3 * num_experts * 4 bytes)
    cudaMemcpyAsync(sfa_swizzle_meta, meta_buf, 3 * num_experts * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // -----------------------------------------------------------------
    // Step 2: Launch batched SFA swizzle
    // -----------------------------------------------------------------
    if (max_row_blocks > 0 && sfa_out_offset > 0) {
        const int* dev_row_offsets = static_cast<const int*>(sfa_swizzle_meta);
        const int* dev_M_values = dev_row_offsets + num_experts;
        const int* dev_out_offsets = dev_M_values + num_experts;

        dim3 grid(max_row_blocks, n_col_blocks, num_experts);
        dim3 block(512);

        kScaleToBlockedBatched<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(SFA_rowmajor),
            static_cast<uint8_t*>(sfa_swizzle_out),
            dev_row_offsets,
            dev_M_values,
            dev_out_offsets,
            scale_W,
            num_experts
        );
    }

    // -----------------------------------------------------------------
    // Step 3: Build CUTLASS metadata from base pointers + offsets
    // -----------------------------------------------------------------
    auto meta_offsets = compute_offsets(num_experts);

    std::vector<uint8_t> host_buf(meta_offsets.total, 0);

    auto* ps   = reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(host_buf.data() + meta_offsets.problem_sizes);
    auto* sa   = reinterpret_cast<StrideA*>(host_buf.data() + meta_offsets.stride_A);
    auto* sb   = reinterpret_cast<StrideB*>(host_buf.data() + meta_offsets.stride_B);
    auto* sc   = reinterpret_cast<StrideC*>(host_buf.data() + meta_offsets.stride_C);
    auto* sd   = reinterpret_cast<StrideD*>(host_buf.data() + meta_offsets.stride_D);
    auto* lsfa = reinterpret_cast<LayoutSFA*>(host_buf.data() + meta_offsets.layout_SFA);
    auto* lsfb = reinterpret_cast<LayoutSFB*>(host_buf.data() + meta_offsets.layout_SFB);
    auto* pa   = reinterpret_cast<const ArrayElementA**>(host_buf.data() + meta_offsets.ptr_A);
    auto* pb   = reinterpret_cast<const ArrayElementB**>(host_buf.data() + meta_offsets.ptr_B);
    auto* psfa = reinterpret_cast<const InternalElementSF**>(host_buf.data() + meta_offsets.ptr_SFA);
    auto* psfb = reinterpret_cast<const InternalElementSF**>(host_buf.data() + meta_offsets.ptr_SFB);
    auto* pc   = reinterpret_cast<const ElementC**>(host_buf.data() + meta_offsets.ptr_C);
    auto* pd   = reinterpret_cast<ElementD**>(host_buf.data() + meta_offsets.ptr_D);

    auto A_base   = static_cast<const uint8_t*>(A_concat);
    auto B_base   = static_cast<const uint8_t*>(B_all);
    auto SFA_base = static_cast<const uint8_t*>(sfa_swizzle_out);
    auto SFB_base = static_cast<const uint8_t*>(SFB_per_expert);
    auto D_base   = static_cast<uint8_t*>(D_concat);

    for (int i = 0; i < num_experts; i++) {
        int M = M_values[i];
        ps[i] = {M, N, K};
        sa[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        sb[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        sc[i] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        sd[i] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
        lsfa[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        lsfb[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

        pa[i]   = reinterpret_cast<const ArrayElementA*>(A_base + host_offsets[i] * half_K);
        pb[i]   = reinterpret_cast<const ArrayElementB*>(B_base + i * N * half_K);
        psfa[i] = reinterpret_cast<const InternalElementSF*>(SFA_base + sfa_offs[i]);
        psfb[i] = reinterpret_cast<const InternalElementSF*>(SFB_base + i * sfb_per_expert_bytes);
        pc[i]   = nullptr;
        pd[i]   = reinterpret_cast<ElementD*>(D_base + host_offsets[i] * N * 2);
    }

    // Upload CUTLASS metadata
    cudaMemcpyAsync(gemm_metadata_dev, host_buf.data(), meta_offsets.total,
                    cudaMemcpyHostToDevice, stream);

    // -----------------------------------------------------------------
    // Step 4: Launch CUTLASS grouped GEMM
    // -----------------------------------------------------------------
    auto dev_at = [&](size_t off) { return static_cast<uint8_t*>(gemm_metadata_dev) + off; };

    if (s_sm_count == 0)
        s_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = s_sm_count;
    hw_info.cluster_shape = dim3(1, 1, 1);
    hw_info.cluster_shape_fallback = dim3(1, 1, 1);

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = alpha_val;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 0};
    fusion_args.beta = 0.0f;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dBeta = {_0{}, _0{}, 0};

    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts,
         reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(dev_at(meta_offsets.problem_sizes)),
         ps},
        {reinterpret_cast<const ArrayElementA**>(dev_at(meta_offsets.ptr_A)),
         reinterpret_cast<StrideA*>(dev_at(meta_offsets.stride_A)),
         reinterpret_cast<const ArrayElementB**>(dev_at(meta_offsets.ptr_B)),
         reinterpret_cast<StrideB*>(dev_at(meta_offsets.stride_B)),
         reinterpret_cast<const InternalElementSF**>(dev_at(meta_offsets.ptr_SFA)),
         reinterpret_cast<LayoutSFA*>(dev_at(meta_offsets.layout_SFA)),
         reinterpret_cast<const InternalElementSF**>(dev_at(meta_offsets.ptr_SFB)),
         reinterpret_cast<LayoutSFB*>(dev_at(meta_offsets.layout_SFB))},
        {fusion_args,
         reinterpret_cast<const ElementC**>(dev_at(meta_offsets.ptr_C)),
         reinterpret_cast<StrideC*>(dev_at(meta_offsets.stride_C)),
         reinterpret_cast<ElementD**>(dev_at(meta_offsets.ptr_D)),
         reinterpret_cast<StrideD*>(dev_at(meta_offsets.stride_D))},
        hw_info, scheduler
    };

    Gemm gemm;
    cutlass::Status status;

    status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS SM100 fused grouped GEMM can_implement failed: %d\n", (int)status);
        if (meta_buf != sfa_meta_host) delete[] meta_buf;
        if (sfa_offs != sfa_expert_out_offsets) delete[] sfa_offs;
        return;
    }

    status = gemm.initialize(arguments, workspace_dev, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS SM100 fused grouped GEMM initialize failed: %d\n", (int)status);
        if (meta_buf != sfa_meta_host) delete[] meta_buf;
        if (sfa_offs != sfa_expert_out_offsets) delete[] sfa_offs;
        return;
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS SM100 fused grouped GEMM run failed: %d\n", (int)status);
    }

    // Clean up heap allocations if num_experts > 64
    if (meta_buf != sfa_meta_host) delete[] meta_buf;
    if (sfa_offs != sfa_expert_out_offsets) delete[] sfa_offs;

#else
    fprintf(stderr, "SM100 fused grouped GEMM not supported on this architecture\n");
#endif
}
