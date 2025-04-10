from typing import Optional

import torch

from ..._ops import register_kernel


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
