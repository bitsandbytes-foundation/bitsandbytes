from collections.abc import Sequence
import math

import torch

from ..._ops import register_kernel
from ..utils import GAUDI_SW_VER


# convert btw standard 4-bit compression format and ipex compression format
# needed for backward compatibility with older versions of gaudi sw
def _reverse_4bit_compress_format(weight: torch.Tensor):
    out_1 = (weight & 0xF0) >> 4
    out_2 = (weight & 0xF) << 4
    out = out_1 | out_2
    return out


@register_kernel("bitsandbytes::dequantize_4bit", "hpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    if quant_type != "nf4":
        raise ValueError(f"HPU backend only supports quant_type 'nf4', got {quant_type!r}")
    if A.dtype not in (torch.bfloat16, torch.uint8):
        raise ValueError(f"HPU backend only supports uint8 or bfloat16 storage, got {A.dtype}")

    # Enable non uint8 dtype
    if A.dtype != torch.uint8:
        A = A.view(torch.uint8)

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

    return output
