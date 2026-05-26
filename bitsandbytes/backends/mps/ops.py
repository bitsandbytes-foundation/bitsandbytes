"""MPS backend for bitsandbytes 4-bit quantization ops.

Uses Metal kernels from kernels-community/bitsandbytes-mps via the
HuggingFace Kernels Hub.
"""

from collections.abc import Sequence
from math import prod
from typing import Optional
from warnings import warn

import torch

from ..._ops import register_kernel
from ..default.ops import _gemm_4bit_default_impl

# ---------------------------------------------------------------------------
# Quant-type mapping: BnB uses strings, our Metal kernel uses ints.
# ---------------------------------------------------------------------------
_QUANT_MAP = {"fp4": 1, "nf4": 2}
_kernel = None


def _get_kernel():
    """Lazily load the bitsandbytes-mps kernel (local build or Hub)."""
    global _kernel
    if _kernel is None:
        from kernels import get_kernel

        # TODO: use kernels-community/bitsandbytes-mps when it's available
        _kernel = get_kernel("kernels-community/bitsandbytes-mps")
    return _kernel


# ============================= quantize_4bit =================================


@register_kernel("bitsandbytes::quantize_4bit", "mps")
def _(
    A: torch.Tensor,
    blocksize: int,
    quant_type: str,
    quant_storage: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if blocksize not in (64, 128, 256, 512):
        raise ValueError(f"MPS backend only supports blocksize in (64, 128, 256, 512), got {blocksize}")

    k = _get_kernel()
    packed, absmax = k.quantize_4bit(A.contiguous(), blocksize, _QUANT_MAP[quant_type])

    packed = packed.view(quant_storage).unsqueeze(1)

    return packed, absmax


# ============================ dequantize_4bit ================================


def _dequantize_4bit_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    if A.dtype != torch.uint8:
        A = A.view(torch.uint8)

    numel = prod(shape)
    k = _get_kernel()
    out = k.dequantize_4bit(A, absmax, blocksize, _QUANT_MAP[quant_type], numel, dtype)
    return out.reshape(shape)


@register_kernel("bitsandbytes::dequantize_4bit", "mps")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    if blocksize not in (64, 128, 256, 512):
        raise ValueError(f"MPS backend only supports blocksize in (64, 128, 256, 512), got {blocksize}")
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
    result = _dequantize_4bit_impl(A, absmax, blocksize, quant_type, shape, dtype)
    out.copy_(result)


# ================================ gemv_4bit ==================================


def _gemv_4bit_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    if B.dtype != torch.uint8:
        B = B.view(torch.uint8)

    quant_type_int = _QUANT_MAP["fp4"] if code[1] > 0 else _QUANT_MAP["nf4"]
    output_features = shapeB[0]

    k = _get_kernel()
    return k.gemv_4bit(A, B, absmax, output_features, blocksize, quant_type_int)


@register_kernel("bitsandbytes::gemv_4bit", "mps")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    return _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize)


@register_kernel("bitsandbytes::gemv_4bit.out", "mps")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    result = _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize)
    out.copy_(result)


@register_kernel("bitsandbytes::gemm_4bit", "mps")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    bias: Optional[torch.Tensor] = None,
    absmax_8bit: Optional[torch.Tensor] = None,
    absmax_code: Optional[torch.Tensor] = None,
    absmax_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    K = A.shape[-1]
    M = A.numel() // K
    N = shapeB[0]

    if absmax_8bit is not None:
        absmax = (
            torch.ops.bitsandbytes.dequantize_blockwise.default(absmax_8bit, absmax, absmax_code, 256, torch.float32)
            + absmax_offset
        )

    if M == 1:
        if K % blocksize == 0:
            if B.dtype != torch.uint8:
                B = B.view(torch.uint8)

            k = _get_kernel()
            result = k.gemv_4bit(A, B, absmax.view(N, -1), N, blocksize, _QUANT_MAP[quant_type])

            if bias is not None:
                result = result + bias
            return result

        warn(
            f"inner dimension ({K}) is not aligned for fast kernel "
            f"with blocksize={blocksize}, falling back to slower implementation.",
            UserWarning,
        )

    return _gemm_4bit_default_impl(
        A,
        B,
        shapeB,
        absmax,
        blocksize,
        quant_type,
        bias,
    )
