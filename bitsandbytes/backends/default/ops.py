from math import prod
from typing import Optional

import torch

from ..._ops import register_kernel


@register_kernel("bitsandbytes::int8_mm_dequant", "default")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch._check(A.dtype == torch.int32, lambda: f"A must be int32, got {A.dtype}")
    torch._check(row_stats.dtype == torch.float32, lambda: f"row_stats must be float32, got {row_stats.dtype}")
    torch._check(col_stats.dtype == torch.float32, lambda: f"col_stats must be float32, got {col_stats.dtype}")

    A_calc = A.view(-1, A.shape[-1])
    row_stats = row_stats.reshape(-1).unsqueeze(-1)
    col_stats = col_stats.reshape(-1).unsqueeze(0)

    out = A_calc * (row_stats * col_stats) * 6.200124e-05
    if bias is not None:
        out += bias

    return out.to(dtype or torch.float16)


@register_kernel("bitsandbytes::int8_mixed_scaled_mm", "default")
def _(
    A: torch.Tensor,
    CA: torch.Tensor,
    CB: torch.Tensor,
    SCA: torch.Tensor,
    SCB: torch.Tensor,
    outlier_cols: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    subB = None

    if outlier_cols is not None and outlier_cols.numel():
        # Extract the inputs with outliers in original precision
        subA = A[:, outlier_cols].contiguous()

        # Dequantize the corresponding weight columns
        subB = (
            torch.ops.bitsandbytes.int8_vectorwise_dequant.default(CB[:, outlier_cols].contiguous(), SCB)
            .to(A.dtype)
            .t()
        )

        # TODO: if state.has_fp16_weights: subB = B[:, outlier_cols].t()

    else:
        # Needed for torch.compile when there are no outliers.
        subA = torch.empty(0, device=A.device, dtype=A.dtype)

    # Int8 Matmul + Dequant + Bias
    output = torch.ops.bitsandbytes.int8_scaled_mm.default(CA, CB, SCA, SCB, bias=bias, dtype=A.dtype)

    if subB is not None:
        # Add the outlier columns back to the output
        output = output.addmm(subA, subB)

    return output, subA


@register_kernel("bitsandbytes::int8_scaled_mm", "default")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    out_i32 = torch.ops.bitsandbytes.int8_linear_matmul.default(A, B)
    return torch.ops.bitsandbytes.int8_mm_dequant.default(
        out_i32,
        row_stats,
        col_stats,
        dtype=dtype or torch.float16,
        bias=bias,
    )


@register_kernel("bitsandbytes::int8_linear_matmul", "default")
def _(A: torch.Tensor, B: torch.Tensor):
    return _int8_linear_matmul_impl(A, B)


@register_kernel("bitsandbytes::int8_linear_matmul.out", "default")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    torch._check(out.dtype == torch.int32)
    _int8_linear_matmul_impl(A, B, out)


def _int8_linear_matmul_impl(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None):
    # Naive implementation: perform matmul in fp32
    result = torch.matmul(A.float(), B.float().t()).to(torch.int32)
    if out is not None:
        result = out.copy_(result)
    return result


@register_kernel("bitsandbytes::int8_vectorwise_quant", "default")
def _(A: torch.Tensor, threshold=0.0):
    rows = prod(A.shape[:-1])
    outlier_cols = None

    outlier_restore = None

    if threshold > 0.0:
        outliers = A.abs() >= threshold

        if outliers.any():
            # Determine which columns contain outliers, and zero out the
            # outliers ahead of quantization. We need to keep a backup of these
            # outliers to restore them after quantization.
            outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)
            outlier_restore = A[outliers].clone()
            A[outliers] = 0
        else:
            # Needed for torch.compile support.
            outlier_cols = torch.empty(0, device=A.device, dtype=torch.int64)

    # Get absmax for each row.
    row_stats = torch.max(A.abs(), dim=1).values.float()

    # Quantize row-wise to int8.
    out_row = torch.round(A * (127.0 / row_stats.unsqueeze(-1))).to(torch.int8)

    # Zero out values from outlier columns across all rows.
    if rows > 1 and outlier_cols is not None:
        out_row[:, outlier_cols] = 0

    # Restore outliers.
    if outlier_restore is not None:
        A[outliers] = outlier_restore

    return out_row, row_stats, outlier_cols
