from collections.abc import Sequence
from typing import Optional

import ctypes as ct
from ctypes import _CFuncPtr
import torch

from ..._ops import register_kernel
from ...cextension import lib
from ..default.ops import _dequantize_4bit_impl, _quantize_4bit_impl
from ..utils import CODE
from .shim import MPSTensorShim#, configure_mps_blockwise_kernel


def _sync_mps_if_needed() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _check_mps_device(tensor: torch.Tensor, name: str) -> None:
    torch._check(
        tensor.device.type == "mps",
        lambda: f"{name} must live on an MPS device for the MPS backend, got {tensor.device.type}",
    )


def _supports_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.float32)


def _resolve_quant_fn(dtype: torch.dtype, quant_type: str) -> Optional[_CFuncPtr]:
    try:
        if dtype == torch.float16:
            fn = getattr(
                lib,
                "cquantize_blockwise_fp16_fp4" if quant_type == "fp4" else "cquantize_blockwise_fp16_nf4",
            )
            # configure_mps_blockwise_kernel(fn)
            return fn
        if dtype == torch.float32:
            fn = getattr(
                lib,
                "cquantize_blockwise_fp32_fp4" if quant_type == "fp4" else "cquantize_blockwise_fp32_nf4",
            )
            # configure_mps_blockwise_kernel(fn)
            return fn
    except AttributeError:
        return None
    return None


def _resolve_dequant_fn(dtype: torch.dtype, quant_type: str) -> Optional[_CFuncPtr]:
    try:
        if dtype == torch.float16:
            fn = getattr(
                lib,
                "cdequantize_blockwise_fp16_fp4" if quant_type == "fp4" else "cdequantize_blockwise_fp16_nf4",
            )
            # configure_mps_blockwise_kernel(fn)
            return fn
        if dtype == torch.float32:
            fn = getattr(
                lib,
                "cdequantize_blockwise_fp32_fp4" if quant_type == "fp4" else "cdequantize_blockwise_fp32_nf4",
            )
            # configure_mps_blockwise_kernel(fn)
            return fn
    except AttributeError:
        return None
    return None


def _quantize_4bit_native(
    A: torch.Tensor,
    blocksize: int,
    quant_type: str,
    quant_storage: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if quant_storage != torch.uint8 or not _supports_dtype(A.dtype):
        return None

    fn = _resolve_quant_fn(A.dtype, quant_type)
    if fn is None:
        return None

    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty(((n + 1) // (quant_storage.itemsize * 2), 1), device=A.device, dtype=quant_storage)

    input_shim = MPSTensorShim.from_tensor(A)
    absmax_shim = MPSTensorShim.from_tensor(absmax)
    out_shim = MPSTensorShim.from_tensor(out)

    _sync_mps_if_needed()
    fn(
        input_shim.struct,
        absmax_shim.struct,
        out_shim.struct,
        ct.c_int32(blocksize),
        ct.c_int32(n),
    )
    return out, absmax


def _dequantize_4bit_native(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> bool:
    if A.dtype != torch.uint8 or not _supports_dtype(dtype):
        return False

    _check_mps_device(absmax, "absmax")
    fn = _resolve_dequant_fn(dtype, quant_type)
    if fn is None:
        return False

    packed_shim = MPSTensorShim.from_tensor(A)
    absmax_shim = MPSTensorShim.from_tensor(absmax)
    out_shim = MPSTensorShim.from_tensor(out)

    _sync_mps_if_needed()
    fn(
        packed_shim.struct,
        absmax_shim.struct,
        out_shim.struct,
        ct.c_int32(blocksize),
        ct.c_int32(out.numel()),
    )
    return True


@register_kernel("bitsandbytes::quantize_4bit", "mps")
def _(
    A: torch.Tensor,
    blocksize: int,
    quant_type: str,
    quant_storage: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_mps_device(A, "A")
    # result = _quantize_4bit_native(A, blocksize, quant_type, quant_storage)
    # if result is not None:
    #     return result
    return _quantize_4bit_impl(A, blocksize, quant_type, quant_storage)


@register_kernel("bitsandbytes::dequantize_4bit", "mps")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    _check_mps_device(A, "A")
    _check_mps_device(absmax, "absmax")

    out = torch.empty(shape, dtype=dtype, device=A.device)
    if _dequantize_4bit_native(A, absmax, blocksize, quant_type, dtype, out):
        return out
    return _dequantize_4bit_impl(A, absmax, blocksize, quant_type, shape, dtype)


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
    _check_mps_device(A, "A")
    _check_mps_device(out, "out")
    _check_mps_device(absmax, "absmax")
    torch._check(out.shape == tuple(shape), lambda: f"Expected out.shape == {tuple(shape)}, got {out.shape}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")

    if not _dequantize_4bit_native(A, absmax, blocksize, quant_type, dtype, out):
        result = _dequantize_4bit_impl(A, absmax, blocksize, quant_type, shape, dtype)
        out.copy_(result)

