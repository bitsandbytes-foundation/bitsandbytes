import ctypes as ct
from typing import Optional, Tuple

import torch

from bitsandbytes.functional import get_ptr

from ..._ops import register_kernel
from ...cextension import lib


@register_kernel("bitsandbytes::int8_linear_matmul", "cpu")
def _(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    # Naive implementation: perform matmul in fp32
    result = torch.matmul(A.float(), B.float().t()).to(torch.int32)
    if out is not None:
        result = out.copy_(result)
    return result


@register_kernel("bitsandbytes::quantize_blockwise", "cpu")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> Tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.float32, "A must be float32")

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
    torch._check(dtype == torch.float32, "A must be float32 on cpu")

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
