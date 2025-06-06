from collections.abc import Sequence
import math

import torch

from bitsandbytes.utils import _reverse_4bit_compress_format

from ..._ops import register_kernel
from ..utils import GAUDI_SW_VER


@register_kernel("bitsandbytes::dequantize_4bit", "hpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4, got {quant_type}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.uint8],
        lambda: f"quant_storage supports uint8 or bfloat16, but got {A.dtype}",
    )

    # Enable non uint8 dtype
    if A.dtype != torch.uint8:
        A = A.view(torch.uint8)

    transpose = False if len(A.shape) == 2 and A.shape[0] == 1 else True

    A = A.reshape(-1)

    if GAUDI_SW_VER and (GAUDI_SW_VER.major < 1 or GAUDI_SW_VER.minor < 22):
        A = _reverse_4bit_compress_format(A)

    # HPU dequantization function for NF4 quantized tensors.
    out_dq = torch.ops.hpu.dequantize_nf4(
        A,
        absmax.to(dtype),
        blocksize,
        out_shape=(math.prod(shape),),
        out_dtype=dtype,
    )

    output = out_dq.reshape(shape)

    if transpose:
        output = output.t()

    return output
