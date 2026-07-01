"""MPS backend for bitsandbytes quantization ops.

Hub kernels (kernels-community/bitsandbytes-mps) are attempted lazily on
macOS 26+. On older macOS the hub kernel path is skipped entirely and
default fallbacks or MPS-specific pure PyTorch fallbacks are used for all ops.

Note: not all ops have implementations on the Hub kernels. Those that do not are also
implemented using pure PyTorch fallbacks.
"""

from collections.abc import Sequence
from math import prod
import platform
from typing import Optional

import torch

from ..._ops import register_kernel
from ..default.ops import (
    _dequantize_4bit_compute,
    _get_4bit_quantize_bounds,
    _try_torch_compile,
)
from ..utils import _get_4bit_code

_QUANT_MAP = {"fp4": 1, "nf4": 2}

_kernel = None

_macos_major = int(platform.mac_ver()[0].split(".")[0]) if platform.mac_ver()[0] else 0

# Pre-set to True on macOS < 26 so _get_kernel() never attempts the import.
_kernel_load_failed = _macos_major < 26


def _get_kernel():
    global _kernel, _kernel_load_failed
    if _kernel_load_failed:
        return None
    if _kernel is not None:
        return _kernel
    try:
        from kernels import get_kernel

        _kernel = get_kernel("kernels-community/bitsandbytes-mps", version=1)
    except Exception:
        _kernel_load_failed = True
        return None
    return _kernel


@_try_torch_compile(dynamic=True)
def _quantize_blockwise_compute(
    A_flat: torch.Tensor, code: torch.Tensor, blocksize: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    On torch <= 2.12, torch.bucketize does not perform well.
    Implements blockwise quantization using a binary search instead of using the default.
    """
    n = A_flat.numel()
    rem = n % blocksize
    full = n - rem
    blocks = full // blocksize
    A_com = A_flat[:full].reshape(blocks, blocksize)
    absmax = A_com.abs().max(dim=-1)[0]
    scaled = torch.clamp(A_com * (1.0 / absmax.clamp(min=1e-38).view(-1, 1)), -1, 1).reshape(-1)
    if rem:
        am = A_flat[full:].abs().max().clamp(min=1e-38)
        absmax = torch.cat([absmax, am.unsqueeze(0)])
        scaled = torch.cat([scaled, torch.clamp(A_flat[full:] / am, -1, 1)])
    bounds = (code[:-1] + code[1:]) / 2
    n_bounds = bounds.shape[0]
    n_iters = n_bounds.bit_length()
    lo = torch.zeros(scaled.shape, dtype=torch.int16, device=scaled.device)
    hi = torch.full(scaled.shape, n_bounds, dtype=torch.int16, device=scaled.device)
    for _ in range(n_iters):
        mid = (lo + hi) >> 1
        val = bounds[mid.to(torch.int64)]
        lo = torch.where(val < scaled, (mid + 1).to(torch.int16), lo)
        hi = torch.where(val >= scaled, mid, hi)
    return lo.to(torch.uint8), absmax


@register_kernel("bitsandbytes::quantize_blockwise", "mps")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    q, absmax = _quantize_blockwise_compute(A.reshape(-1).float(), code.float(), blocksize)
    return q.reshape(A.shape), absmax


@_try_torch_compile(dynamic=True)
def _quantize_4bit_compute(
    A_flat: torch.Tensor,
    blocksize: int,
    bounds: torch.Tensor,
    order: torch.Tensor,
    nf4: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = A_flat.numel()
    rem = n % blocksize
    full = n - rem
    blocks = full // blocksize
    A_com = A_flat[:full].reshape(blocks, blocksize)
    absmax = A_com.abs().max(dim=-1)[0]
    scaled = torch.clamp(A_com * (1.0 / absmax.clamp(min=1e-38).view(-1, 1)), -1, 1).reshape(-1)
    if rem:
        am = A_flat[full:].abs().max().clamp(min=1e-38)
        absmax = torch.cat([absmax, am.unsqueeze(0)])
        scaled = torch.cat([scaled, torch.clamp(A_flat[full:] / am, -1, 1)])
    if scaled.numel() % 2:
        scaled = torch.nn.functional.pad(scaled, (0, 1))
    idx = torch.zeros(scaled.shape, dtype=torch.int8, device=scaled.device)
    for b in bounds:
        idx = idx + (scaled > b).to(torch.int8)
    if not nf4:
        idx = order[idx.to(torch.int32)]
    q8 = idx.to(torch.uint8)
    return (q8[::2] << 4) | q8[1::2], absmax


def _quantize_4bit_fallback(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    bounds, order = _get_4bit_quantize_bounds(quant_type, A.device)
    packed, absmax = _quantize_4bit_compute(A.reshape(-1).float(), blocksize, bounds, order, quant_type == "nf4")
    packed = packed.unsqueeze(1)
    if quant_storage != torch.uint8:
        packed = packed.squeeze().view(quant_storage).unsqueeze(1)
    return packed, absmax


@register_kernel("bitsandbytes::quantize_4bit", "mps")
def _(
    A: torch.Tensor,
    blocksize: int,
    quant_type: str,
    quant_storage: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if blocksize in (64, 128, 256, 512) and (k := _get_kernel()) is not None:
        packed, absmax = k.quantize_4bit(A.contiguous(), blocksize, _QUANT_MAP[quant_type])
        packed = packed.view(quant_storage).unsqueeze(1)
        return packed, absmax
    return _quantize_4bit_fallback(A, blocksize, quant_type, quant_storage)


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

    # Use HF Hub kernel when supported.
    if blocksize in (64, 128, 256, 512) and (k := _get_kernel()) is not None:
        numel = prod(shape)
        out = k.dequantize_4bit(A, absmax, blocksize, _QUANT_MAP[quant_type], numel, dtype)
        return out.reshape(shape)

    # Fallback to implementation from default backend.
    code = _get_4bit_code(quant_type, A.device)
    return _dequantize_4bit_compute(A.reshape(-1), absmax, code, blocksize, shape, dtype)


@register_kernel("bitsandbytes::dequantize_4bit", "mps")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
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


def _gemv_4bit_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    if blocksize in (64, 128, 256) and (k := _get_kernel()) is not None:
        if B.dtype != torch.uint8:
            B = B.view(torch.uint8)

        output_features = shapeB[0]
        quant_type_int = _QUANT_MAP["fp4"] if code[1] > 0 else _QUANT_MAP["nf4"]

        return k.gemv_4bit(A, B, absmax, output_features, blocksize, quant_type_int)

    quant_type = "fp4" if code[1] > 0 else "nf4"
    B_dq = _dequantize_4bit_impl(B, absmax, blocksize, quant_type, shapeB, A.dtype)
    return torch.nn.functional.linear(A, B_dq)


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

    # For nested absmax, we don't have a fused implementation yet.
    # Dequantize the absmax values first.
    if absmax_8bit is not None:
        absmax = (
            torch.ops.bitsandbytes.dequantize_blockwise.default(absmax_8bit, absmax, absmax_code, 256, torch.float32)
            + absmax_offset
        )

    # Use HF Hub kernel when supported for GEMV.
    if M == 1 and blocksize in (64, 128, 256) and (k := _get_kernel()) is not None:
        if B.dtype != torch.uint8:
            B = B.view(torch.uint8)
        result = k.gemv_4bit(A, B, absmax.view(N, -1), N, blocksize, _QUANT_MAP[quant_type])
        if bias is not None:
            result = result + bias
        return result

    # Fallback: dequantize + linear.
    B_dq = _dequantize_4bit_impl(B, absmax, blocksize, quant_type, shapeB, A.dtype)
    return torch.nn.functional.linear(A, B_dq, bias)
