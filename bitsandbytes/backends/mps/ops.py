from __future__ import annotations

import ctypes as ct
from typing import Sequence, Tuple

import torch

from ..._ops import register_kernel
from ...cextension import lib
_ALLOWED_BLOCKS = (64, 128, 256, 512, 1024, 2048, 4096)
_SUPPORTED_DTYPES = (torch.float16, torch.float32)


lib.cquantize_blockwise_fp16_nf4_tensor.argtypes = [ct.py_object, ct.py_object, ct.py_object, ct.c_int32]
lib.cquantize_blockwise_fp16_nf4_tensor.restype = None
lib.cquantize_blockwise_fp32_nf4_tensor.argtypes = [ct.py_object, ct.py_object, ct.py_object, ct.c_int32]
lib.cquantize_blockwise_fp32_nf4_tensor.restype = None
lib.cdequantize_blockwise_fp16_nf4_tensor.argtypes = [ct.py_object, ct.py_object, ct.py_object, ct.c_int32]
lib.cdequantize_blockwise_fp16_nf4_tensor.restype = None
lib.cdequantize_blockwise_fp32_nf4_tensor.argtypes = [ct.py_object, ct.py_object, ct.py_object, ct.c_int32]
lib.cdequantize_blockwise_fp32_nf4_tensor.restype = None


def _quantize_nf4(
    A: torch.Tensor, blocksize: int, quant_storage: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch._check(blocksize in _ALLOWED_BLOCKS)
    torch._check(quant_storage == torch.uint8, lambda: "Only uint8 storage is supported for NF4 on MPS.")

    A = A.contiguous()
    n = A.numel()
    blocks = -(n // -blocksize)

    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty(((n + 1) // 2, 1), device=A.device, dtype=quant_storage)

    if A.dtype == torch.float16:
        lib.cquantize_blockwise_fp16_nf4_tensor(ct.py_object(A), ct.py_object(absmax), ct.py_object(out), ct.c_int32(blocksize))
    elif A.dtype == torch.float32:
        lib.cquantize_blockwise_fp32_nf4_tensor(ct.py_object(A), ct.py_object(absmax), ct.py_object(out), ct.c_int32(blocksize))
    else:
        torch._check(False, lambda: f"NF4 quantization on MPS supports {list(_SUPPORTED_DTYPES)}, got {A.dtype}")

    return out, absmax


def _dequantize_nf4(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(blocksize in _ALLOWED_BLOCKS)

    A = A.contiguous()
    absmax = absmax.contiguous()
    torch._check(out.is_contiguous(), lambda: "Output tensor must be contiguous for NF4 dequantization on MPS.")

    if dtype == torch.float16:
        lib.cdequantize_blockwise_fp16_nf4_tensor(ct.py_object(A), ct.py_object(absmax), ct.py_object(out), ct.c_int32(blocksize))
    elif dtype == torch.float32:
        lib.cdequantize_blockwise_fp32_nf4_tensor(ct.py_object(A), ct.py_object(absmax), ct.py_object(out), ct.c_int32(blocksize))
    else:
        torch._check(False, lambda: f"NF4 dequantization on MPS supports {list(_SUPPORTED_DTYPES)}, got {dtype}")


@register_kernel("bitsandbytes::quantize_4bit", "mps")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    if quant_type != "nf4" or A.dtype not in _SUPPORTED_DTYPES:
        return torch.ops.bitsandbytes.quantize_4bit.default(A, blocksize, quant_type, quant_storage)
    return _quantize_nf4(A, blocksize, quant_storage)


@register_kernel("bitsandbytes::dequantize_4bit", "mps")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    if quant_type != "nf4" or dtype not in _SUPPORTED_DTYPES:
        return torch.ops.bitsandbytes.dequantize_4bit.default(A, absmax, blocksize, quant_type, shape, dtype)
    out = torch.empty(shape, dtype=dtype, device=A.device)
    _dequantize_nf4(A, absmax, blocksize, dtype, out)
    return out


@register_kernel("bitsandbytes::dequantize_4bit.out", "mps")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    if quant_type != "nf4" or dtype not in _SUPPORTED_DTYPES:
        torch.ops.bitsandbytes.dequantize_4bit.out.default(
            A,
            absmax,
            blocksize,
            quant_type,
            shape,
            dtype,
            out,
        )
        return

    torch._check(out.shape == tuple(shape), lambda: f"Expected out.shape == {tuple(shape)}, got {out.shape}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    _dequantize_nf4(A, absmax, blocksize, dtype, out)
