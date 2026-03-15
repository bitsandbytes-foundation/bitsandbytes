from collections.abc import Sequence
from math import prod
from typing import Optional

import torch

_IS_TORCH_GTE_24 = False

if hasattr(torch.library, "register_fake"):
    _IS_TORCH_GTE_24 = True
    register_fake = torch.library.register_fake
    register_kernel = torch.library.register_kernel
else:
    # PyTorch <= 2.3
    register_fake = torch.library.impl_abstract
    register_kernel = torch.library.impl

# Int8 mixed precision matmul + dequant + bias
torch.library.define(
    "bitsandbytes::int8_mixed_scaled_mm",
    "(Tensor A, Tensor CA, Tensor CB, Tensor SCA, Tensor SCB, Tensor? outlier_cols=None, Tensor? bias=None) -> (Tensor, Tensor?)",
)


@register_fake("bitsandbytes::int8_mixed_scaled_mm")
def _(
    A: torch.Tensor,
    CA: torch.Tensor,
    CB: torch.Tensor,
    SCA: torch.Tensor,
    SCB: torch.Tensor,
    outlier_cols: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    shapeC = (*CA.shape[:-1], CB.shape[0])

    out = torch.empty(shapeC, device=A.device, dtype=A.dtype)

    outlier_cols = torch.library.get_ctx().new_dynamic_size()
    subA = A.new_empty(outlier_cols, dtype=torch.int64)

    return out, subA


# Higher level op: int8 matmul + dequant + bias
torch.library.define(
    "bitsandbytes::int8_scaled_mm",
    "(Tensor A, Tensor B, Tensor row_stats, Tensor col_stats, Tensor? bias=None, ScalarType? dtype=None) -> Tensor",
)


@register_fake("bitsandbytes::int8_scaled_mm")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    shapeC = (*A.shape[:-1], B.shape[0])
    return torch.empty(shapeC, device=A.device, dtype=dtype or torch.float16)


torch.library.define(
    "bitsandbytes::int8_linear_matmul",
    "(Tensor A, Tensor B) -> Tensor",
)


@register_fake("bitsandbytes::int8_linear_matmul")
def _(A: torch.Tensor, B: torch.Tensor):
    torch._check(A.dtype == torch.int8, lambda: "A must be int8")
    torch._check(B.dtype == torch.int8, lambda: "B must be int8")
    shapeC = (*A.shape[:-1], B.shape[0])
    return torch.empty(shapeC, device=A.device, dtype=torch.int32)


# More info on `out` overloads:
# https://github.com/pytorch/pytorch/issues/125044
torch.library.define(
    "bitsandbytes::int8_linear_matmul.out",
    "(Tensor A, Tensor B, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::int8_linear_matmul.out")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    shapeC = (*A.shape[:-1], B.shape[0])

    torch._check(A.dtype == torch.int8, lambda: "A must be int8")
    torch._check(B.dtype == torch.int8, lambda: "B must be int8")
    torch._check(out.shape == shapeC, lambda: f"Expected out.shape == {shapeC}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == torch.int32, lambda: f"Expected out.dtype == int32, got {out.dtype}")


torch.library.define(
    "bitsandbytes::int8_vectorwise_quant",
    "(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor?)",
)


@register_fake("bitsandbytes::int8_vectorwise_quant")
def _(A: torch.Tensor, threshold=0.0):
    out_row = torch.empty(A.shape, device=A.device, dtype=torch.int8)
    row_stats = torch.empty(prod(A.shape[:-1]), device=A.device, dtype=torch.float32)

    if threshold == 0.0:
        return out_row, row_stats, None

    outlier_cols = torch.library.get_ctx().new_dynamic_size()

    return out_row, row_stats, A.new_empty(outlier_cols, dtype=torch.int64)


torch.library.define("bitsandbytes::int8_vectorwise_dequant", "(Tensor A, Tensor stats) -> Tensor")


@register_fake("bitsandbytes::int8_vectorwise_dequant")
def _(A: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    torch._check(A.dtype == torch.int8, lambda: "A must be int8")
    return torch.empty_like(A, dtype=torch.float32)


# Default PyTorch-native implementation
@register_kernel("bitsandbytes::int8_vectorwise_dequant", "default")
def _(A: torch.Tensor, stats: torch.Tensor):
    # To dequantize we divide by 127, or multiply by the reciprocal.
    return A * stats.view(-1, 1) * 7.874015718698502e-3


torch.library.define(
    "bitsandbytes::int8_mm_dequant",
    "(Tensor A, Tensor row_stats, Tensor col_stats, ScalarType? dtype=None, Tensor? bias=None) -> Tensor",
)


@register_fake("bitsandbytes::int8_mm_dequant")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch._check(A.dtype == torch.int32, lambda: "A must be int32")
    return torch.empty_like(A, dtype=dtype or torch.float16)


torch.library.define(
    "bitsandbytes::int8_double_quant",
    "(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor?)",
)


@register_fake("bitsandbytes::int8_double_quant")
def _(
    A: torch.Tensor,
    threshold=0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    out_row = torch.empty_like(A, dtype=torch.int8)
    out_col = torch.empty_like(A, dtype=torch.int8)
    row_stats = torch.empty(prod(A.shape[:-1]), device=A.device, dtype=torch.float32)
    col_stats = torch.empty(A.shape[-1], device=A.device, dtype=torch.float32)
    outlier_n = torch.library.get_ctx().new_dynamic_size()
    outlier_cols = A.new_empty(outlier_n, dtype=torch.int64)
    return out_row, out_col, row_stats, col_stats, outlier_cols


torch.library.define(
    "bitsandbytes::dequantize_4bit",
    "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_4bit")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    return torch.empty(shape, dtype=dtype, device=A.device)


torch.library.define(
    "bitsandbytes::dequantize_4bit.out",
    "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::dequantize_4bit.out")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)
    torch._check(out.shape == shape, lambda: f"Expected out.shape == {shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")


torch.library.define(
    "bitsandbytes::quantize_4bit",
    "(Tensor A, int blocksize, str quant_type, ScalarType quant_storage) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::quantize_4bit")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)

    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty(((n + 1) // (quant_storage.itemsize * 2), 1), device=A.device, dtype=quant_storage)
    return out, absmax


torch.library.define(
    "bitsandbytes::dequantize_blockwise",
    "(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_blockwise")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    return torch.empty_like(A, dtype=dtype)


torch.library.define(
    "bitsandbytes::dequantize_blockwise.out",
    "(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::dequantize_blockwise.out")
def _(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor
):
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")


torch.library.define("bitsandbytes::quantize_blockwise", "(Tensor A, Tensor code, int blocksize) -> (Tensor, Tensor)")


@register_fake("bitsandbytes::quantize_blockwise")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty_like(A, dtype=torch.uint8)
    return out, absmax


torch.library.define(
    "bitsandbytes::gemv_4bit",
    "(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code, int blocksize) -> Tensor",
)


@register_fake("bitsandbytes::gemv_4bit")
def _(
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int], absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.numel() == A.size(-1), lambda: f"A must be a vector with leading dimensions of 1, got {A.shape}")
    torch._check(
        A.dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"A must be float16, bfloat16, or float32, got {A.dtype}",
    )
    torch._check(
        B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32],
        lambda: f"B must be backed by storage of type uint8, bfloat16, float16, or float32, got {B.dtype}",
    )
    shape = (*A.shape[:-1], shapeB[0])
    return torch.empty(shape, device=A.device, dtype=A.dtype)


torch.library.define(
    "bitsandbytes::gemv_4bit.out",
    "(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code, int blocksize, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::gemv_4bit.out")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)
    torch._check(A.numel() == A.size(-1), lambda: f"A must be a vector with leading dimensions of 1, got {A.shape}")
    torch._check(
        A.dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"A must be float16, bfloat16, or float32, got {A.dtype}",
    )
    torch._check(
        B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32],
        lambda: f"B must be backed by storage of type uint8, bfloat16, float16, or float32, got {B.dtype}",
    )
    torch._check(
        out.shape == (*A.shape[:-1], shapeB[0]),
        lambda: f"Expected out.shape == {(*A.shape[:-1], shapeB[0])}, got {out.shape}",
    )
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == A.dtype, lambda: f"Expected out.dtype == {A.dtype}, got {out.dtype}")


torch.library.define(
    "bitsandbytes::optimizer_update_32bit",
    "(str optimizer_name, Tensor(a0!) g, Tensor(a1!) p, Tensor(a2!) state1, Tensor(a3!)? state2, Tensor(a4!)? unorm_vec, float max_unorm, float param_norm, float beta1, float beta2, float beta3, float alpha, float eps, float weight_decay, int step, float lr, float gnorm_scale, bool skip_zeros=False) -> ()",
)


@register_fake("bitsandbytes::optimizer_update_32bit")
def _(
    optimizer_name: str,
    g: torch.Tensor,
    p: torch.Tensor,
    state1: torch.Tensor,
    state2: Optional[torch.Tensor],
    unorm_vec: Optional[torch.Tensor],
    max_unorm: float,
    param_norm: float,
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps: float,
    weight_decay: float,
    step: int,
    lr: float,
    gnorm_scale: float,
    skip_zeros=False,
) -> None:
    torch._check(
        g.numel() == p.numel(),
        lambda: f"g and p must have the same number of elements, got {g.numel()} and {p.numel()}",
    )
    compute_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    torch._check(
        g.dtype in compute_dtypes,
        lambda: f"g must be bfloat16, float16, or float32, got {g.dtype}",
    )
    torch._check(
        g.dtype == p.dtype,
        lambda: f"Expected all tensors to have the same dtype, got g.dtype={g.dtype}, p.dtype={p.dtype}",
    )


torch.library.define(
    "bitsandbytes::optimizer_update_8bit_blockwise",
    "(str optimizer_name, Tensor(a0!) g, Tensor(a1!) p, Tensor(a2!) state1, Tensor(a3!)? state2, float beta1, float beta2, float beta3, float alpha, float eps, int step, float lr, Tensor(a4!) qmap1, Tensor(a5!)? qmap2, Tensor(a6!) absmax1, Tensor(a7!)? absmax2, float weight_decay, float gnorm_scale, bool skip_zeros=False) -> ()",
)


@register_fake("bitsandbytes::optimizer_update_8bit_blockwise")
def _(
    optimizer_name: str,
    g: torch.Tensor,
    p: torch.Tensor,
    state1: torch.Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: torch.Tensor,
    qmap2: Optional[torch.Tensor],
    absmax1: torch.Tensor,
    absmax2: Optional[torch.Tensor],
    weight_decay: float,
    gnorm_scale: float,
    skip_zeros=False,
) -> None:
    torch._check(
        g.numel() == p.numel(),
        lambda: f"g and p must have the same number of elements, got {g.numel()} and {p.numel()}",
    )
    compute_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    torch._check(
        g.dtype in compute_dtypes,
        lambda: f"g must be bfloat16, float16, or float32, got {g.dtype}",
    )
    torch._check(
        g.dtype == p.dtype,
        lambda: f"Expected all tensors to have the same dtype, got g.dtype={g.dtype}, p.dtype={p.dtype}",
    )
    torch._check(
        state1.dtype == torch.uint8,
        lambda: f"state1 must be uint8, got {state1.dtype}",
    )
    torch._check(
        qmap1.dtype == absmax1.dtype == torch.float32,
        lambda: f"Expected qmap1 and absmax1 to be float32, got qmap1.dtype={qmap1.dtype}, absmax1.dtype={absmax1.dtype}",
    )
    if state2 is not None:
        torch._check(
            state2.dtype == torch.uint8,
            lambda: f"state2 must be uint8, got {state2.dtype}",
        )
        torch._check(
            qmap2.dtype == absmax2.dtype == torch.float32,
            lambda: f"Expected qmap2 and absmax2 to be float32, got qmap2.dtype={qmap2.dtype}, absmax2.dtype={absmax2.dtype}",
        )


# NVFP4 dequantization
torch.library.define(
    "bitsandbytes::dequantize_nvfp4",
    "(Tensor packed, Tensor block_scales, float tensor_scale, int numel, ScalarType dtype) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_nvfp4")
def _(
    packed: torch.Tensor, block_scales: torch.Tensor, tensor_scale: float, numel: int, dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, device=packed.device)


# CUTLASS-based fused quantize for NVFP4 (SM_120+)
# Uses QuTLASS GEMM-as-quantize approach with always-on randomized Hadamard
# rotation. The rotation is free (baked into the GEMM B operand) and improves
# quantization quality by spreading outliers across blocks.
torch.library.define(
    "bitsandbytes::cutlass_fused_quantize_nvfp4",
    "(Tensor A, float tensor_scale) -> (Tensor, Tensor, Tensor)",
)


@register_fake("bitsandbytes::cutlass_fused_quantize_nvfp4")
def _(
    A: torch.Tensor,
    tensor_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"NVFP4 requires numel divisible by 16, got {n}")
    packed = torch.empty(n // 2, dtype=torch.uint8, device=A.device)
    block_scales = torch.empty(n // 16, dtype=torch.uint8, device=A.device)
    ts_out = torch.empty(1, dtype=torch.float32, device=A.device)
    return packed, block_scales, ts_out


# Device-side quantize variant: global_scale is a device tensor (no .item() sync).
# Returns (packed, block_scales) — row-major scales without swizzling.
torch.library.define(
    "bitsandbytes::cutlass_fused_quantize_nvfp4_raw",
    "(Tensor A, Tensor global_scale_dev) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::cutlass_fused_quantize_nvfp4_raw")
def _(
    A: torch.Tensor,
    global_scale_dev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"NVFP4 requires numel divisible by 16, got {n}")
    packed = torch.empty(n // 2, dtype=torch.uint8, device=A.device)
    block_scales = torch.empty(n // 16, dtype=torch.uint8, device=A.device)
    return packed, block_scales


# Scale reordering for CUTLASS block-scaled GEMM
torch.library.define(
    "bitsandbytes::scale_to_blocked",
    "(Tensor scales, int H, int W) -> Tensor",
)


@register_fake("bitsandbytes::scale_to_blocked")
def _(scales: torch.Tensor, H: int, W: int) -> torch.Tensor:
    n_row_blocks = (H + 127) // 128
    n_col_blocks = (W + 3) // 4
    out_size = n_row_blocks * n_col_blocks * 128 * 4
    return torch.empty(out_size, dtype=torch.uint8, device=scales.device)


# Batched scale reordering for MoE: row-major → per-expert swizzled
torch.library.define(
    "bitsandbytes::scale_to_blocked_batched",
    "(Tensor scales_rowmajor, Tensor expert_row_offsets, Tensor expert_M, "
    "Tensor expert_out_offsets, int W, int num_experts, int max_row_blocks, "
    "int total_out_bytes) -> Tensor",
)


@register_fake("bitsandbytes::scale_to_blocked_batched")
def _(
    scales_rowmajor: torch.Tensor,
    expert_row_offsets: torch.Tensor,
    expert_M: torch.Tensor,
    expert_out_offsets: torch.Tensor,
    W: int,
    num_experts: int,
    max_row_blocks: int,
    total_out_bytes: int,
) -> torch.Tensor:
    return torch.empty(total_out_bytes, dtype=torch.uint8, device=scales_rowmajor.device)


# Inverse scale reordering: CUTLASS block-scaled layout → row-major
torch.library.define(
    "bitsandbytes::scale_from_blocked",
    "(Tensor blocked_scales, int H, int W) -> Tensor",
)


@register_fake("bitsandbytes::scale_from_blocked")
def _(blocked_scales: torch.Tensor, H: int, W: int) -> torch.Tensor:
    return torch.empty(H * W, dtype=torch.uint8, device=blocked_scales.device)


# MoE scatter: concatenated FP4 → padded per-expert batched FP4
torch.library.define(
    "bitsandbytes::moe_scatter_nvfp4",
    "(Tensor packed_concat, Tensor expert_offsets, int max_M, int K, int num_experts) -> Tensor",
)


@register_fake("bitsandbytes::moe_scatter_nvfp4")
def _(
    packed_concat: torch.Tensor,
    expert_offsets: torch.Tensor,
    max_M: int,
    K: int,
    num_experts: int,
) -> torch.Tensor:
    row_bytes = K // 2
    return torch.empty(num_experts * max_M * row_bytes, dtype=torch.uint8, device=packed_concat.device)


# MoE gather: padded per-expert BF16 → concatenated BF16
torch.library.define(
    "bitsandbytes::moe_gather_bf16",
    "(Tensor D_batched, Tensor expert_offsets, int max_M, int N, int num_experts, int total_tokens) -> Tensor",
)


@register_fake("bitsandbytes::moe_gather_bf16")
def _(
    D_batched: torch.Tensor,
    expert_offsets: torch.Tensor,
    max_M: int,
    N: int,
    num_experts: int,
    total_tokens: int,
) -> torch.Tensor:
    return torch.empty(total_tokens * N, dtype=torch.bfloat16, device=D_batched.device)


# NVFP4 GEMM (A @ B^T with block-scaled FP4 inputs)
torch.library.define(
    "bitsandbytes::gemm_nvfp4",
    "(Tensor A_packed, Tensor B_packed, Tensor A_scales, Tensor B_scales, "
    "float A_tensor_scale, float B_tensor_scale, int M, int N, int K) -> Tensor",
)


@register_fake("bitsandbytes::gemm_nvfp4")
def _(
    A_packed: torch.Tensor,
    B_packed: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    A_tensor_scale: float,
    B_tensor_scale: float,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    torch._check_is_size(M)
    torch._check_is_size(N)
    torch._check_is_size(K)
    return torch.empty(M, N, dtype=torch.float32, device=A_packed.device)


# Grouped NVFP4 GEMM for MoE inference
# Fuses all expert GEMMs into a single kernel launch.
# A_concat:      [total_tokens, K/2]     packed activations (all experts concatenated)
# B_all:         [num_experts * N, K/2]  packed weights (per-expert, stacked)
# SFA_concat:    swizzled activation scales (CUTLASS block-scaled layout, total_tokens rows)
# SFB_all:       swizzled weight scales (CUTLASS block-scaled layout, num_experts*N rows)
# expert_offsets: [num_experts + 1]       cumulative token offsets (int32)
# cumul_m_tiles:  [num_experts + 1]       cumulative m-tile counts (int32)
torch.library.define(
    "bitsandbytes::gemm_nvfp4_grouped",
    "(Tensor A_concat, Tensor B_all, Tensor SFA_concat, Tensor SFB_all, "
    "Tensor expert_offsets, Tensor cumul_m_tiles, "
    "float A_tensor_scale, float B_tensor_scale, "
    "int N, int K, int num_experts) -> Tensor",
)


@register_fake("bitsandbytes::gemm_nvfp4_grouped")
def _(
    A_concat: torch.Tensor,
    B_all: torch.Tensor,
    SFA_concat: torch.Tensor,
    SFB_all: torch.Tensor,
    expert_offsets: torch.Tensor,
    cumul_m_tiles: torch.Tensor,
    A_tensor_scale: float,
    B_tensor_scale: float,
    N: int,
    K: int,
    num_experts: int,
) -> torch.Tensor:
    torch._check_is_size(N)
    torch._check_is_size(K)
    # total_tokens = number of rows in A_concat = A_concat.numel() / (K/2)
    total_tokens = A_concat.numel() // (K // 2)
    return torch.empty(total_tokens, N, dtype=torch.bfloat16, device=A_concat.device)


# Batched NVFP4 GEMM for MoE inference (SM_100 datacenter Blackwell)
# All experts compute max_M rows (padded); CUDA-graph friendly.
# A_batched:   (num_experts * max_M * K // 2,) packed FP4 activations
# B_batched:   (num_experts * N * K // 2,)     packed FP4 weights
# SFA:         batched swizzled activation scales (L per-expert copies concatenated)
# SFB:         batched swizzled weight scales (L per-expert copies concatenated)
torch.library.define(
    "bitsandbytes::gemm_nvfp4_moe",
    "(Tensor A_batched, Tensor B_batched, Tensor SFA, Tensor SFB, "
    "Tensor alpha, int max_M, int N, int K, int num_experts) -> Tensor",
)


@register_fake("bitsandbytes::gemm_nvfp4_moe")
def _(
    A_batched: torch.Tensor,
    B_batched: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    alpha: torch.Tensor,
    max_M: int,
    N: int,
    K: int,
    num_experts: int,
) -> torch.Tensor:
    torch._check_is_size(max_M)
    torch._check_is_size(N)
    torch._check_is_size(K)
    torch._check_is_size(num_experts)
    return torch.empty(num_experts, max_M, N, dtype=torch.bfloat16, device=A_batched.device)


# MoE weighted gather: fused gather + scale by gating weight + FP32 accumulate + BF16 convert.
# Two-phase: atomicAdd into FP32 workspace, then convert to BF16.
# workspace_fp32 is a caller-managed scratch buffer (persistent for CUDA graphs).
torch.library.define(
    "bitsandbytes::moe_weighted_gather_bf16",
    "(Tensor D_batched, Tensor output_bf16, Tensor workspace_fp32, "
    "Tensor token_ids, Tensor expert_ids, Tensor slot_ids, Tensor weights, "
    "int num_tokens, int max_M, int N) -> Tensor",
)


@register_fake("bitsandbytes::moe_weighted_gather_bf16")
def _(
    D_batched: torch.Tensor,
    output_bf16: torch.Tensor,
    workspace_fp32: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: int,
    max_M: int,
    N: int,
) -> torch.Tensor:
    return output_bf16


# K-bit blockwise quantization (K=2..5, blocksize=32)

torch.library.define(
    "bitsandbytes::quantize_kbit",
    "(Tensor A, Tensor codebook, int k) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::quantize_kbit")
def _(A: torch.Tensor, codebook: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(codebook.numel() == (1 << k), lambda: f"codebook must have {1 << k} entries for k={k}")
    n = A.numel()
    num_blocks = -(n // -32)
    # packed: num_blocks * k int32 words + k padding words
    packed = torch.empty(num_blocks * k + k, device=A.device, dtype=torch.int32)
    absmax = torch.empty(num_blocks + 1, device=A.device, dtype=torch.uint8)
    return packed, absmax


torch.library.define(
    "bitsandbytes::dequantize_kbit",
    "(Tensor packed, Tensor codebook, Tensor absmax, int k, int n, ScalarType dtype) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_kbit")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    n: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        absmax.dtype in (torch.float32, torch.uint8),
        lambda: f"absmax must be float32 or uint8 (E4M4), got {absmax.dtype}",
    )
    num_blocks = -(n // -32)
    return torch.empty(num_blocks * 32, device=packed.device, dtype=dtype)


torch.library.define(
    "bitsandbytes::dequantize_kbit_",
    "(Tensor packed, Tensor codebook, Tensor absmax, int k, int n, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
)


@register_fake("bitsandbytes::dequantize_kbit_")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    n: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        absmax.dtype in (torch.float32, torch.uint8),
        lambda: f"absmax must be float32 or uint8 (E4M4), got {absmax.dtype}",
    )
    num_blocks = -(n // -32)
    torch._check(out.numel() >= num_blocks * 32, lambda: f"out must have at least {num_blocks * 32} elements")
    torch._check(out.dtype == dtype, lambda: f"out dtype {out.dtype} must match requested dtype {dtype}")
    return out


# K-bit dequantize from tiled layout (repack_kbit output -> flat [N, K_dim] row-major)

torch.library.define(
    "bitsandbytes::dequantize_kbit_tiled",
    "(Tensor packed, Tensor codebook, Tensor absmax, int k, int K_dim, int N, ScalarType dtype) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_kbit_tiled")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    K_dim: int,
    N: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        absmax.dtype in (torch.float32, torch.uint8, torch.float16),
        lambda: f"absmax must be float32, uint8 (E4M4), or float16, got {absmax.dtype}",
    )
    n = N * K_dim
    num_blocks = -(n // -32)
    return torch.empty(num_blocks * 32, device=packed.device, dtype=dtype)


torch.library.define(
    "bitsandbytes::dequantize_kbit_tiled_",
    "(Tensor packed, Tensor codebook, Tensor absmax, int k, int K_dim, int N, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
)


@register_fake("bitsandbytes::dequantize_kbit_tiled_")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    K_dim: int,
    N: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    n = N * K_dim
    num_blocks = -(n // -32)
    torch._check(out.numel() >= num_blocks * 32, lambda: f"out must have at least {num_blocks * 32} elements")
    torch._check(out.dtype == dtype, lambda: f"out dtype {out.dtype} must match requested dtype {dtype}")
    return out


# VQ (Vector Quantization) quantize/dequantize
#
# VQ traits helper: compute derived constants from (p, index_bits).
# Must match VQTraits<P_VAL, INDEX_BITS> in csrc/ops.cu.
_VQ_VALID_CONFIGS = {(2, 8), (2, 10), (3, 8), (3, 10), (4, 8)}


def _vq_traits(p: int, index_bits: int = 8) -> dict:
    BS = 48 if p == 3 else 32
    CB_ENTRIES = 256 if index_bits == 8 else 1024
    GROUPS = BS // p
    WORDS = (GROUPS * index_bits + 31) // 32
    TILE_K = 96 if p == 3 else 64
    TILE_N = 128
    KB_PER_TILE = TILE_K // BS
    return {
        "BS": BS,
        "CB_ENTRIES": CB_ENTRIES,
        "GROUPS": GROUPS,
        "WORDS": WORDS,
        "TILE_K": TILE_K,
        "TILE_N": TILE_N,
        "KB_PER_TILE": KB_PER_TILE,
    }


torch.library.define(
    "bitsandbytes::quantize_vq",
    "(Tensor A, Tensor codebook, int p, int index_bits=8) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::quantize_vq")
def _(A: torch.Tensor, codebook: torch.Tensor, p: int, index_bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    traits = _vq_traits(p, index_bits)
    n = A.numel()
    num_blocks = -(n // -traits["BS"])
    packed = torch.empty(num_blocks * traits["WORDS"], device=A.device, dtype=torch.int32)
    absmax = torch.empty(num_blocks, device=A.device, dtype=torch.uint8)
    return packed, absmax


torch.library.define(
    "bitsandbytes::dequantize_vq",
    "(Tensor packed, Tensor codebook, Tensor absmax, int p, int n, ScalarType dtype, int index_bits=8) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_vq")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    p: int,
    n: int,
    dtype: torch.dtype,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    BS = 48 if p == 3 else 32
    num_blocks = -(n // -BS)
    return torch.empty(num_blocks * BS, device=packed.device, dtype=dtype)


torch.library.define(
    "bitsandbytes::dequantize_vq_",
    "(Tensor packed, Tensor codebook, Tensor absmax, int p, int n, ScalarType dtype, Tensor(a!) out, "
    "int index_bits=8) -> Tensor(a!)",
)


@register_fake("bitsandbytes::dequantize_vq_")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    p: int,
    n: int,
    dtype: torch.dtype,
    out: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    return out


# VQ tiled dequantize: reads tiled VQ layout, writes flat [N, K_dim] output

torch.library.define(
    "bitsandbytes::dequantize_vq_tiled",
    "(Tensor packed_tiled, Tensor codebook, Tensor absmax_tiled, int p, int K_dim, int N, ScalarType dtype, "
    "int index_bits=8) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_vq_tiled")
def _(
    packed_tiled: torch.Tensor,
    codebook: torch.Tensor,
    absmax_tiled: torch.Tensor,
    p: int,
    K_dim: int,
    N: int,
    dtype: torch.dtype,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    return torch.empty(N * K_dim, device=packed_tiled.device, dtype=dtype)


torch.library.define(
    "bitsandbytes::dequantize_vq_tiled_",
    "(Tensor packed_tiled, Tensor codebook, Tensor absmax_tiled, int p, int K_dim, int N, ScalarType dtype, "
    "Tensor(a!) out, int index_bits=8) -> Tensor(a!)",
)


@register_fake("bitsandbytes::dequantize_vq_tiled_")
def _(
    packed_tiled: torch.Tensor,
    codebook: torch.Tensor,
    absmax_tiled: torch.Tensor,
    p: int,
    K_dim: int,
    N: int,
    dtype: torch.dtype,
    out: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    return out


# VQ scalar GEMV: codebook lookup GEMV for M=1-4

torch.library.define(
    "bitsandbytes::vq_scalar_gemv",
    "(Tensor A, Tensor B_packed, Tensor B_absmax, Tensor codebook, int K_dim, int N, int p, "
    "int index_bits=8) -> Tensor",
)


@register_fake("bitsandbytes::vq_scalar_gemv")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.shape[0] <= 4, lambda: f"vq_scalar_gemv supports M<=4, got {A.shape[0]}")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    M = A.shape[0]
    return torch.empty(M, N, device=A.device, dtype=A.dtype)


torch.library.define(
    "bitsandbytes::vq_scalar_gemv.out",
    "(Tensor A, Tensor B_packed, Tensor B_absmax, Tensor codebook, int K_dim, int N, int p, Tensor(a!) out, "
    "int index_bits=8) -> ()",
)


@register_fake("bitsandbytes::vq_scalar_gemv.out")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    out: torch.Tensor,
    index_bits: int = 8,
) -> None:
    pass


# VQ scalar GEMV with tiled B layout

torch.library.define(
    "bitsandbytes::vq_scalar_gemv_tiled",
    "(Tensor A, Tensor B_packed_tiled, Tensor B_absmax_tiled, Tensor codebook, int K_dim, int N, int p, "
    "int index_bits=8) -> Tensor",
)


@register_fake("bitsandbytes::vq_scalar_gemv_tiled")
def _(
    A: torch.Tensor,
    B_packed_tiled: torch.Tensor,
    B_absmax_tiled: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.shape[0] <= 4, lambda: f"vq_scalar_gemv_tiled supports M<=4, got {A.shape[0]}")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    M = A.shape[0]
    return torch.empty(M, N, device=A.device, dtype=A.dtype)


# VQ scalar GEMV tiled with pre-allocated output (CUDA graph compatible)

torch.library.define(
    "bitsandbytes::vq_scalar_gemv_tiled_",
    "(Tensor A, Tensor B_packed_tiled, Tensor B_absmax_tiled, Tensor codebook, int K_dim, int N, int p, "
    "Tensor(a!) out, int index_bits=8) -> Tensor(a!)",
)


@register_fake("bitsandbytes::vq_scalar_gemv_tiled_")
def _(
    A: torch.Tensor,
    B_packed_tiled: torch.Tensor,
    B_absmax_tiled: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    out: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.shape[0] <= 4, lambda: f"vq_scalar_gemv_tiled_ supports M<=4, got {A.shape[0]}")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    torch._check(out.dtype == A.dtype, lambda: f"out dtype {out.dtype} must match A dtype {A.dtype}")
    return out


# K-bit repack: flat bit-plane layout -> GEMM-tiled layout

torch.library.define(
    "bitsandbytes::repack_kbit",
    "(Tensor packed_flat, Tensor absmax_flat, int K_dim, int N, int k) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::repack_kbit")
def _(
    packed_flat: torch.Tensor, absmax_flat: torch.Tensor, K_dim: int, N: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    TILE_K, TILE_N, BLOCKSIZE = 64, 128, 32
    torch._check(N % TILE_N == 0, lambda: f"N ({N}) must be divisible by {TILE_N}")
    torch._check(K_dim % BLOCKSIZE == 0, lambda: f"K_dim ({K_dim}) must be divisible by {BLOCKSIZE}")
    K_dim_padded = ((K_dim + TILE_K - 1) // TILE_K) * TILE_K
    k_tiles = K_dim_padded // TILE_K
    n_tiles = N // TILE_N
    k_blocks_per_tile = TILE_K // BLOCKSIZE
    total_words = k_tiles * n_tiles * TILE_N * k_blocks_per_tile * k
    total_absmax = k_tiles * n_tiles * TILE_N * k_blocks_per_tile
    packed_tiled = torch.empty(total_words, device=packed_flat.device, dtype=torch.int32)
    absmax_tiled = torch.empty(total_absmax, device=packed_flat.device, dtype=torch.uint8)
    return packed_tiled, absmax_tiled


# VQ repack: flat VQ byte layout -> tiled layout

torch.library.define(
    "bitsandbytes::repack_vq",
    "(Tensor packed_flat, Tensor absmax_flat, int K_dim, int N, int p, int index_bits=8) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::repack_vq")
def _(
    packed_flat: torch.Tensor, absmax_flat: torch.Tensor, K_dim: int, N: int, p: int, index_bits: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    traits = _vq_traits(p, index_bits)
    BS = traits["BS"]
    TILE_K = traits["TILE_K"]
    TILE_N = traits["TILE_N"]
    WORDS = traits["WORDS"]
    KB_PER_TILE = traits["KB_PER_TILE"]
    torch._check(N % TILE_N == 0, lambda: f"N ({N}) must be divisible by {TILE_N}")
    torch._check(K_dim % BS == 0, lambda: f"K_dim ({K_dim}) must be divisible by {BS}")
    K_dim_padded = ((K_dim + TILE_K - 1) // TILE_K) * TILE_K
    k_tiles = K_dim_padded // TILE_K
    n_tiles = N // TILE_N
    total_words = k_tiles * n_tiles * TILE_N * KB_PER_TILE * WORDS
    total_absmax = k_tiles * n_tiles * TILE_N * KB_PER_TILE
    packed_tiled = torch.empty(total_words, device=packed_flat.device, dtype=torch.int32)
    absmax_tiled = torch.empty(total_absmax, device=packed_flat.device, dtype=torch.uint8)
    return packed_tiled, absmax_tiled


# Hadamard rotation (in-place, for kbit quantization outlier spreading)

torch.library.define(
    "bitsandbytes::hadamard_rotate_",
    "(Tensor(a!) data, int block_size, Tensor? signs) -> Tensor(a!)",
)


@register_fake("bitsandbytes::hadamard_rotate_")
def _(data: torch.Tensor, block_size: int, signs: Optional[torch.Tensor]) -> torch.Tensor:
    torch._check(
        block_size in (32, 64, 128, 256),
        lambda: f"block_size must be 32, 64, 128, or 256, got {block_size}",
    )
    torch._check(
        data.dtype in (torch.float16, torch.bfloat16),
        lambda: f"hadamard_rotate only supports float16/bfloat16, got {data.dtype}",
    )
    if signs is not None:
        torch._check(
            signs.dtype == torch.int32,
            lambda: f"signs must be int32, got {signs.dtype}",
        )
        torch._check(
            signs.numel() == block_size // 32,
            lambda: f"signs must have {block_size // 32} elements for block_size={block_size}, got {signs.numel()}",
        )
    return data


# Full-dimension Hadamard rotation (in-place, for kbit quantization outlier spreading)
# Unlike hadamard_rotate_ which uses block-diagonal Hadamard, this rotates across
# the entire last dimension of the input tensor.

torch.library.define(
    "bitsandbytes::hadamard_rotate_full_",
    "(Tensor(a!) data, int dim, Tensor? signs) -> Tensor(a!)",
)


@register_fake("bitsandbytes::hadamard_rotate_full_")
def _(data: torch.Tensor, dim: int, signs: Optional[torch.Tensor]) -> torch.Tensor:
    supported_dims = (512, 1024, 2048, 4096, 8192)
    torch._check(
        dim in supported_dims,
        lambda: f"dim must be one of {supported_dims}, got {dim}",
    )
    torch._check(
        data.numel() % dim == 0,
        lambda: f"data.numel() ({data.numel()}) must be divisible by dim ({dim})",
    )
    torch._check(
        data.dtype in (torch.float16, torch.bfloat16),
        lambda: f"hadamard_rotate_full only supports float16/bfloat16, got {data.dtype}",
    )
    if signs is not None:
        torch._check(
            signs.dtype == torch.int32,
            lambda: f"signs must be int32, got {signs.dtype}",
        )
        torch._check(
            signs.numel() == dim // 32,
            lambda: f"signs must have {dim // 32} elements for dim={dim}, got {signs.numel()}",
        )
    return data


# K-bit fused dequant + GEMM (production: fp16 + bf16)

torch.library.define(
    "bitsandbytes::kbit_gemm_prod",
    "(Tensor A, Tensor B_packed, Tensor B_absmax, Tensor codebook, int K_dim, int N, int k, int k_chunks) -> Tensor",
)


@register_fake("bitsandbytes::kbit_gemm_prod")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    k_chunks: int,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    M = A.shape[0]
    return torch.empty(M, N, device=A.device, dtype=A.dtype)


# K-bit fused dequant + GEMM with pre-allocated output and workspace (CUDA graph compatible)

torch.library.define(
    "bitsandbytes::kbit_gemm_prod_",
    "(Tensor A, Tensor B_packed, Tensor B_absmax, Tensor codebook, int K_dim, int N, int k, int k_chunks, "
    "Tensor(a!) out, Tensor C_workspace, Tensor tile_counters) -> Tensor(a!)",
)


@register_fake("bitsandbytes::kbit_gemm_prod_")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    k_chunks: int,
    out: torch.Tensor,
    C_workspace: torch.Tensor,
    tile_counters: torch.Tensor,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    M = A.shape[0]
    torch._check(out.shape == (M, N), lambda: f"out must be [{M}, {N}], got {list(out.shape)}")
    torch._check(out.dtype == A.dtype, lambda: f"out dtype {out.dtype} must match A dtype {A.dtype}")
    torch._check(C_workspace.dtype == torch.float32, lambda: f"C_workspace must be float32, got {C_workspace.dtype}")
    torch._check(tile_counters.dtype == torch.int32, lambda: f"tile_counters must be int32, got {tile_counters.dtype}")
    return out


# VQ fused dequant + MMA GEMM: codebook-based quantized matmul via tensor cores

torch.library.define(
    "bitsandbytes::vq_gemm_prod",
    "(Tensor A, Tensor B_packed, Tensor B_absmax, Tensor codebook, int K_dim, int N, int p, int k_chunks, "
    "int index_bits=8) -> Tensor",
)


@register_fake("bitsandbytes::vq_gemm_prod")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    k_chunks: int,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    M = A.shape[0]
    return torch.empty(M, N, device=A.device, dtype=A.dtype)


# VQ fused dequant + MMA GEMM with pre-allocated output and workspace (CUDA graph compatible)

torch.library.define(
    "bitsandbytes::vq_gemm_prod_",
    "(Tensor A, Tensor B_packed, Tensor B_absmax, Tensor codebook, int K_dim, int N, int p, int k_chunks, "
    "Tensor(a!) out, Tensor C_workspace, Tensor tile_counters, int index_bits=8) -> Tensor(a!)",
)


@register_fake("bitsandbytes::vq_gemm_prod_")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    k_chunks: int,
    out: torch.Tensor,
    C_workspace: torch.Tensor,
    tile_counters: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config: p={p}, index_bits={index_bits}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    M = A.shape[0]
    torch._check(out.shape == (M, N), lambda: f"out must be [{M}, {N}], got {list(out.shape)}")
    torch._check(out.dtype == A.dtype, lambda: f"out dtype {out.dtype} must match A dtype {A.dtype}")
    return out


# K-bit grouped expert GEMM: batch multiple MoE expert GEMMs into one launch

torch.library.define(
    "bitsandbytes::kbit_grouped_gemm",
    "(Tensor A_concat, Tensor B_packed_all, Tensor B_absmax_all, Tensor codebook, "
    "Tensor expert_offsets, int K_dim, int N, int k, int num_experts, int max_M) -> Tensor",
)


@register_fake("bitsandbytes::kbit_grouped_gemm")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    num_experts: int,
    max_M: int,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(A_concat.dim() == 2 and A_concat.shape[1] == K_dim, lambda: "A_concat must be [total_M, K_dim]")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A_concat.dtype}"
    )
    total_M = A_concat.shape[0]
    return torch.empty(total_M, N, device=A_concat.device, dtype=A_concat.dtype)


# K-bit grouped expert GEMM with pre-allocated output and workspace (CUDA graph compatible)

torch.library.define(
    "bitsandbytes::kbit_grouped_gemm_",
    "(Tensor A_concat, Tensor B_packed_all, Tensor B_absmax_all, Tensor codebook, "
    "Tensor expert_offsets, int K_dim, int N, int k, int num_experts, int max_M, "
    "Tensor(a!) out, Tensor C_workspace, Tensor tile_counters) -> Tensor(a!)",
)


@register_fake("bitsandbytes::kbit_grouped_gemm_")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    num_experts: int,
    max_M: int,
    out: torch.Tensor,
    C_workspace: torch.Tensor,
    tile_counters: torch.Tensor,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(A_concat.dim() == 2 and A_concat.shape[1] == K_dim, lambda: "A_concat must be [total_M, K_dim]")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A_concat.dtype}"
    )
    total_M = A_concat.shape[0]
    torch._check(out.shape == (total_M, N), lambda: f"out must be [{total_M}, {N}], got {list(out.shape)}")
    torch._check(out.dtype == A_concat.dtype, lambda: f"out dtype {out.dtype} must match A dtype {A_concat.dtype}")
    torch._check(C_workspace.dtype == torch.float32, lambda: f"C_workspace must be float32, got {C_workspace.dtype}")
    torch._check(tile_counters.dtype == torch.int32, lambda: f"tile_counters must be int32, got {tile_counters.dtype}")
    return out


# VQ Grouped expert GEMM: fused VQ codebook MoE GEMM across all experts

torch.library.define(
    "bitsandbytes::vq_grouped_gemm",
    "(Tensor A_concat, Tensor B_packed_all, Tensor B_absmax_all, Tensor codebook, "
    "Tensor expert_offsets, int K_dim, int N, int p, int num_experts, int max_M, int index_bits=8) -> Tensor",
)


@register_fake("bitsandbytes::vq_grouped_gemm")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    num_experts: int,
    max_M: int,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check(p == 2, lambda: f"VQ grouped GEMM only supports p=2, got {p}")
    torch._check(A_concat.dim() == 2 and A_concat.shape[1] == K_dim, lambda: "A_concat must be [total_M, K_dim]")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A_concat.dtype}"
    )
    total_M = A_concat.shape[0]
    return torch.empty(total_M, N, device=A_concat.device, dtype=A_concat.dtype)


# VQ Grouped expert GEMM — inplace with pre-allocated output, workspace, and tile_counters

torch.library.define(
    "bitsandbytes::vq_grouped_gemm_",
    "(Tensor A_concat, Tensor B_packed_all, Tensor B_absmax_all, Tensor codebook, "
    "Tensor expert_offsets, int K_dim, int N, int p, int num_experts, int max_M, "
    "Tensor(a!) out, Tensor C_workspace, Tensor tile_counters, int index_bits=8) -> Tensor(a!)",
)


@register_fake("bitsandbytes::vq_grouped_gemm_")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    num_experts: int,
    max_M: int,
    out: torch.Tensor,
    C_workspace: torch.Tensor,
    tile_counters: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check(p == 2, lambda: f"VQ grouped GEMM only supports p=2, got {p}")
    torch._check(A_concat.dim() == 2 and A_concat.shape[1] == K_dim, lambda: "A_concat must be [total_M, K_dim]")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A_concat.dtype}"
    )
    total_M = A_concat.shape[0]
    torch._check(out.shape == (total_M, N), lambda: f"out must be [{total_M}, {N}], got {list(out.shape)}")
    torch._check(out.dtype == A_concat.dtype, lambda: f"out dtype {out.dtype} must match A dtype {A_concat.dtype}")
    torch._check(C_workspace.dtype == torch.float32, lambda: f"C_workspace must be float32, got {C_workspace.dtype}")
    torch._check(tile_counters.dtype == torch.int32, lambda: f"tile_counters must be int32, got {tile_counters.dtype}")
    return out


# VQ Grouped Scalar GEMV: fused MoE expert scalar GEMV for M=1..4

torch.library.define(
    "bitsandbytes::vq_grouped_scalar_gemv",
    "(Tensor A_concat, Tensor B_packed_all, Tensor B_absmax_all, Tensor codebook, "
    "Tensor expert_offsets, int K_dim, int N, int p, int num_experts, int max_M, int index_bits=8) -> Tensor",
)


@register_fake("bitsandbytes::vq_grouped_scalar_gemv")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    num_experts: int,
    max_M: int,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config ({p}, {index_bits})")
    torch._check(A_concat.dim() == 2 and A_concat.shape[1] == K_dim, lambda: "A_concat must be [total_M, K_dim]")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A_concat.dtype}"
    )
    torch._check(max_M <= 4, lambda: f"vq_grouped_scalar_gemv supports max_M<=4, got {max_M}")
    total_M = A_concat.shape[0]
    return torch.empty(total_M, N, device=A_concat.device, dtype=A_concat.dtype)


# VQ Grouped Scalar GEMV — inplace with pre-allocated output (CUDA graph compatible)

torch.library.define(
    "bitsandbytes::vq_grouped_scalar_gemv_",
    "(Tensor A_concat, Tensor B_packed_all, Tensor B_absmax_all, Tensor codebook, "
    "Tensor expert_offsets, int K_dim, int N, int p, int num_experts, int max_M, "
    "Tensor(a!) out, int index_bits=8) -> Tensor(a!)",
)


@register_fake("bitsandbytes::vq_grouped_scalar_gemv_")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    num_experts: int,
    max_M: int,
    out: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check((p, index_bits) in _VQ_VALID_CONFIGS, lambda: f"Invalid VQ config ({p}, {index_bits})")
    torch._check(A_concat.dim() == 2 and A_concat.shape[1] == K_dim, lambda: "A_concat must be [total_M, K_dim]")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A_concat.dtype}"
    )
    torch._check(max_M <= 4, lambda: f"vq_grouped_scalar_gemv_ supports max_M<=4, got {max_M}")
    total_M = A_concat.shape[0]
    torch._check(out.shape == (total_M, N), lambda: f"out must be [{total_M}, {N}], got {list(out.shape)}")
    torch._check(out.dtype == A_concat.dtype, lambda: f"out dtype {out.dtype} must match A dtype {A_concat.dtype}")
    return out


# K-bit scalar GEMV: C[M,N] = A[M,K_dim] * W_kbit^T (M=1..4, scalar FMA)

torch.library.define(
    "bitsandbytes::kbit_scalar_gemv",
    "(Tensor A, Tensor B_packed, Tensor B_absmax, Tensor codebook, int K_dim, int N, int k) -> Tensor",
)

torch.library.define(
    "bitsandbytes::kbit_scalar_gemv.out",
    "(Tensor A, Tensor B_packed, Tensor B_absmax, Tensor codebook, int K_dim, int N, int k, Tensor(a!) out) -> ()",
)


@register_fake("bitsandbytes::kbit_scalar_gemv")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.shape[0] <= 4, lambda: f"kbit_scalar_gemv supports M<=4, got {A.shape[0]}")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    M = A.shape[0]
    return torch.empty(M, N, device=A.device, dtype=A.dtype)


@register_fake("bitsandbytes::kbit_scalar_gemv.out")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    out: torch.Tensor,
) -> None:
    pass


# K-bit scalar GEMV with tiled B layout (same kernel, tile-aware addressing)

torch.library.define(
    "bitsandbytes::kbit_scalar_gemv_tiled",
    "(Tensor A, Tensor B_packed_tiled, Tensor B_absmax_tiled, Tensor codebook, int K_dim, int N, int k) -> Tensor",
)


@register_fake("bitsandbytes::kbit_scalar_gemv_tiled")
def _(
    A: torch.Tensor,
    B_packed_tiled: torch.Tensor,
    B_absmax_tiled: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.shape[0] <= 4, lambda: f"kbit_scalar_gemv_tiled supports M<=4, got {A.shape[0]}")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    M = A.shape[0]
    return torch.empty(M, N, device=A.device, dtype=A.dtype)


# K-bit scalar GEMV tiled with pre-allocated output (CUDA graph compatible)

torch.library.define(
    "bitsandbytes::kbit_scalar_gemv_tiled_",
    "(Tensor A, Tensor B_packed_tiled, Tensor B_absmax_tiled, Tensor codebook, int K_dim, int N, int k, "
    "Tensor(a!) out) -> Tensor(a!)",
)


@register_fake("bitsandbytes::kbit_scalar_gemv_tiled_")
def _(
    A: torch.Tensor,
    B_packed_tiled: torch.Tensor,
    B_absmax_tiled: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    out: torch.Tensor,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.shape[0] <= 4, lambda: f"kbit_scalar_gemv_tiled_ supports M<=4, got {A.shape[0]}")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    torch._check(out.dtype == A.dtype, lambda: f"out dtype {out.dtype} must match A dtype {A.dtype}")
    return out


# K-bit scalar GEMV v2: tiled with shared memory + split-K (CUDA graph compatible)

torch.library.define(
    "bitsandbytes::kbit_scalar_gemv_v2_",
    "(Tensor A, Tensor B_packed_tiled, Tensor B_absmax_tiled, Tensor codebook, int K_dim, int N, int k, "
    "Tensor(a!) out, Tensor C_workspace, Tensor tile_counters) -> Tensor(a!)",
)


@register_fake("bitsandbytes::kbit_scalar_gemv_v2_")
def _(
    A: torch.Tensor,
    B_packed_tiled: torch.Tensor,
    B_absmax_tiled: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    out: torch.Tensor,
    C_workspace: torch.Tensor,
    tile_counters: torch.Tensor,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(A.dim() == 2 and A.shape[1] == K_dim, lambda: "A must be [M, K_dim]")
    torch._check(A.shape[0] <= 4, lambda: f"kbit_scalar_gemv_v2_ supports M<=4, got {A.shape[0]}")
    torch._check(A.dtype in (torch.float16, torch.bfloat16), lambda: f"A must be fp16 or bf16, got {A.dtype}")
    torch._check(out.dtype == A.dtype, lambda: f"out dtype {out.dtype} must match A dtype {A.dtype}")
    torch._check(C_workspace.dtype == torch.float32, lambda: f"C_workspace must be float32, got {C_workspace.dtype}")
    torch._check(tile_counters.dtype == torch.int32, lambda: f"tile_counters must be int32, got {tile_counters.dtype}")
    return out


# ============================================================================
# Training Kernels (from QLORA-2 branch)
# ============================================================================

# SwiGLU forward: h = silu(gate) * up
torch.library.define(
    "bitsandbytes::swiglu_forward",
    "(Tensor gate, Tensor up) -> Tensor",
)


@register_fake("bitsandbytes::swiglu_forward")
def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    torch._check(gate.shape == up.shape, lambda: "gate and up must have same shape")
    return torch.empty_like(gate)


# SwiGLU backward: (grad_gate, grad_up) from grad_h
torch.library.define(
    "bitsandbytes::swiglu_backward",
    "(Tensor grad_h, Tensor gate, Tensor up) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::swiglu_backward")
def _(grad_h: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(gate), torch.empty_like(up)


# RMSNorm forward: y = x * rsqrt(mean(x^2) + eps) * w, also returns rrms
torch.library.define(
    "bitsandbytes::rmsnorm_forward",
    "(Tensor x, Tensor w, float eps, bool add_unit_offset) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::rmsnorm_forward")
def _(x: torch.Tensor, w: torch.Tensor, eps: float, add_unit_offset: bool) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(x.dim() == 2, lambda: "x must be 2D [rows, cols]")
    rows = x.shape[0]
    out = torch.empty_like(x)
    rrms = torch.empty(rows, device=x.device, dtype=torch.float32)
    return out, rrms


# RMSNorm backward: (grad_x, grad_w) from grad_out
torch.library.define(
    "bitsandbytes::rmsnorm_backward",
    "(Tensor grad_out, Tensor x, Tensor w, Tensor rrms, bool add_unit_offset) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::rmsnorm_backward")
def _(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rrms: torch.Tensor,
    add_unit_offset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = torch.empty_like(x)
    grad_w = torch.empty(x.shape[1], device=x.device, dtype=torch.float32)
    return grad_x, grad_w


# RoPE forward (in-place): applies rotary embeddings to Q (or Q+K)
torch.library.define(
    "bitsandbytes::rope_forward",
    "(Tensor(a!) q, Tensor cos_cache, Tensor sin_cache, int n_heads) -> ()",
)


@register_fake("bitsandbytes::rope_forward")
def _(q: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, n_heads: int) -> None:
    pass


# Cross-Entropy Loss forward: per-sample loss + logsumexp for backward
torch.library.define(
    "bitsandbytes::cross_entropy_forward",
    "(Tensor logits, Tensor labels, int ignore_index) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::cross_entropy_forward")
def _(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(logits.dim() == 2, lambda: "logits must be 2D [N, V]")
    N = logits.shape[0]
    losses = torch.empty(N, device=logits.device, dtype=torch.float32)
    logsumexp = torch.empty(N, device=logits.device, dtype=torch.float32)
    return losses, logsumexp


# Cross-Entropy Loss backward: grad_logits from grad_output
torch.library.define(
    "bitsandbytes::cross_entropy_backward",
    "(Tensor logits, Tensor labels, Tensor grad_output, Tensor logsumexp, int ignore_index) -> Tensor",
)


@register_fake("bitsandbytes::cross_entropy_backward")
def _(
    logits: torch.Tensor,
    labels: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    return torch.empty_like(logits)


# NVFP4 quantization (E2M1 with two-level scaling: E4M3 block scales + FP32 tensor scale)
torch.library.define(
    "bitsandbytes::quantize_nvfp4",
    "(Tensor A, float? tensor_scale) -> (Tensor, Tensor, Tensor)",
)


@register_fake("bitsandbytes::quantize_nvfp4")
def _(A: torch.Tensor, tensor_scale: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"NVFP4 requires numel divisible by 16, got {n}")
    packed = torch.empty(n // 2, dtype=torch.uint8, device=A.device)
    block_scales = torch.empty(n // 16, dtype=torch.uint8, device=A.device)
    ts_out = torch.empty(1, dtype=torch.float32, device=A.device)
    return packed, block_scales, ts_out


# NVFP4 dequantization
torch.library.define(
    "bitsandbytes::dequantize_nvfp4",
    "(Tensor packed, Tensor block_scales, float tensor_scale, int numel, ScalarType dtype) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_nvfp4")
def _(
    packed: torch.Tensor, block_scales: torch.Tensor, tensor_scale: float, numel: int, dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, device=packed.device)


# NVFP4 Hadamard rotation (in-place)
torch.library.define(
    "bitsandbytes::hadamard_rotate_nvfp4",
    "(Tensor(a!) A) -> ()",
)


@register_fake("bitsandbytes::hadamard_rotate_nvfp4")
def _(A: torch.Tensor) -> None:
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"Hadamard rotation requires numel divisible by 16, got {n}")


# Fused Hadamard rotation + NVFP4 quantize
torch.library.define(
    "bitsandbytes::fused_hadamard_quantize_nvfp4",
    "(Tensor A, float? tensor_scale) -> (Tensor, Tensor, Tensor)",
)


@register_fake("bitsandbytes::fused_hadamard_quantize_nvfp4")
def _(A: torch.Tensor, tensor_scale: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"NVFP4 requires numel divisible by 16, got {n}")
    packed = torch.empty(n // 2, dtype=torch.uint8, device=A.device)
    block_scales = torch.empty(n // 16, dtype=torch.uint8, device=A.device)
    ts_out = torch.empty(1, dtype=torch.float32, device=A.device)
    return packed, block_scales, ts_out


# CUTLASS-based fused quantize for NVFP4 (SM_120+)
# Uses QuTLASS GEMM-as-quantize approach with always-on randomized Hadamard
# rotation. The rotation is free (baked into the GEMM B operand) and improves
# quantization quality by spreading outliers across blocks.
torch.library.define(
    "bitsandbytes::cutlass_fused_quantize_nvfp4",
    "(Tensor A, float tensor_scale) -> (Tensor, Tensor, Tensor)",
)


@register_fake("bitsandbytes::cutlass_fused_quantize_nvfp4")
def _(
    A: torch.Tensor,
    tensor_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"NVFP4 requires numel divisible by 16, got {n}")
    packed = torch.empty(n // 2, dtype=torch.uint8, device=A.device)
    block_scales = torch.empty(n // 16, dtype=torch.uint8, device=A.device)
    ts_out = torch.empty(1, dtype=torch.float32, device=A.device)
    return packed, block_scales, ts_out


# Scale reordering for CUTLASS block-scaled GEMM
torch.library.define(
    "bitsandbytes::scale_to_blocked",
    "(Tensor scales, int H, int W) -> Tensor",
)


@register_fake("bitsandbytes::scale_to_blocked")
def _(scales: torch.Tensor, H: int, W: int) -> torch.Tensor:
    n_row_blocks = (H + 127) // 128
    n_col_blocks = (W + 3) // 4
    out_size = n_row_blocks * n_col_blocks * 128 * 4
    return torch.empty(out_size, dtype=torch.uint8, device=scales.device)


# NVFP4 GEMM (A @ B^T with block-scaled FP4 inputs)
torch.library.define(
    "bitsandbytes::gemm_nvfp4",
    "(Tensor A_packed, Tensor B_packed, Tensor A_scales, Tensor B_scales, "
    "float A_tensor_scale, float B_tensor_scale, int M, int N, int K) -> Tensor",
)


@register_fake("bitsandbytes::gemm_nvfp4")
def _(
    A_packed: torch.Tensor,
    B_packed: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    A_tensor_scale: float,
    B_tensor_scale: float,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    torch._check_is_size(M)
    torch._check_is_size(N)
    torch._check_is_size(K)
