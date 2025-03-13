import ctypes as ct
from typing import Optional, Tuple

import torch

from bitsandbytes.functional import get_ptr

from ..._ops import register_kernel
from ...cextension import lib


@register_kernel("bitsandbytes::int8_linear_matmul", "cpu")
def _(A: torch.Tensor, B: torch.Tensor):
    return _int8_linear_matmul_impl(A, B)


@register_kernel("bitsandbytes::int8_linear_matmul.out", "cpu")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    torch._check(out.dtype == torch.int32)
    _int8_linear_matmul_impl(A, B, out)


def _int8_linear_matmul_impl(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None):
    # Naive implementation: perform matmul in fp32
    result = torch.matmul(A.float(), B.float().t()).to(torch.int32)
    if out is not None:
        result = out.copy_(result)
    return result


@register_kernel("bitsandbytes::int8_mm_dequant", "cpu")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype=torch.float16,
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

    return out.to(dtype)


@register_kernel("bitsandbytes::quantize_blockwise", "cpu")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> Tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.float32, lambda: f"A must be float32 on cpu, got {A.dtype}")

    n = A.numel()
    blocks = -(n // -blocksize)

    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty_like(A, dtype=torch.uint8)

    lib.cquantize_blockwise_cpu_fp32(
        get_ptr(code),
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out),
        ct.c_longlong(blocksize),
        ct.c_longlong(n),
    )

    return out, absmax


@register_kernel("bitsandbytes::dequantize_blockwise", "cpu")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(dtype == torch.float32, lambda: f"dtype must be float32 on cpu, got {dtype}")

    out = torch.empty_like(A, dtype=dtype)

    lib.cdequantize_blockwise_cpu_fp32(
        get_ptr(code),
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out),
        ct.c_longlong(blocksize),
        ct.c_longlong(A.numel()),
    )

    return out
