"""MPS backend for bitsandbytes quantization ops.

Hub kernels (kernels-community/bitsandbytes-mps) are attempted lazily on
macOS 26+. On older macOS the hub kernel path is skipped entirely and
default fallbacks or MPS-specific pure PyTorch fallbacks are used for all ops.

Note: not all ops have implementations on the Hub kernels. Those that do not are also
implemented using pure PyTorch fallbacks.
"""

from collections.abc import Sequence
from math import prod
import os
import platform
from typing import Optional

import torch

from ..._ops import register_kernel
from ...cextension import get_mps_library
from ..default.ops import (
    _dequantize_4bit_compute,
    _dequantize_blockwise_compute,
    _get_4bit_quantize_bounds,
    _try_torch_compile,
)
from ..utils import _get_4bit_code

_QUANT_MAP = {"fp4": 1, "nf4": 2}

# Native hand-written Metal library (None when not built / metallib absent).
_mps_native = get_mps_library()


def _native_available() -> bool:
    """Whether the hand-written Metal quant/dequant kernels are usable on this install."""
    return _mps_native is not None


def _ensure_native_buffer(t: torch.Tensor) -> torch.Tensor:
    """Return a contiguous, storage_offset==0 tensor.

    The native dispatch treats ``tensor.data_ptr()`` as an ``id<MTLBuffer>`` and binds it
    at offset 0, which is only valid for a fresh (offset-0) allocation. A view into a
    larger buffer is cloned to guarantee that.
    """
    t = t.contiguous()
    if t.storage_offset() != 0:
        t = t.clone()
    return t


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


def _quantize_blockwise_native(
    A: torch.Tensor, code: torch.Tensor, blocksize: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route quantize_blockwise through the hand-written Metal kernel.

    Mirrors the reference math exactly (per-block absmax; reciprocal-multiply on full blocks,
    direct divide + clamped absmax on the tail block; searchsorted into the code table).
    Inputs are forced to fp32/offset-0 to match the kernel's ABI. torch.mps.synchronize()
    flushes torch's stream so the native command buffer (on a separate queue) reads
    materialized inputs; the .mm blocks on completion before return.
    """
    A_flat = _ensure_native_buffer(A.reshape(-1).to(torch.float32))
    code_f = _ensure_native_buffer(code.to(torch.float32))

    n = A_flat.numel()
    blocks = -(n // -blocksize)
    out = torch.empty(n, dtype=torch.uint8, device=A.device)
    absmax = torch.empty(blocks, dtype=torch.float32, device=A.device)

    torch.mps.synchronize()
    _mps_native.bnb_mps_quantize_blockwise(
        code_f.data_ptr(),
        A_flat.data_ptr(),
        out.data_ptr(),
        absmax.data_ptr(),
        n,
        blocksize,
    )

    return out.reshape(A.shape), absmax


@register_kernel("bitsandbytes::quantize_blockwise", "mps")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    if _native_available():
        return _quantize_blockwise_native(A, code, blocksize)
    q, absmax = _quantize_blockwise_compute(A.reshape(-1).float(), code.float(), blocksize)
    return q.reshape(A.shape), absmax


def _dequantize_blockwise_native(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype
) -> torch.Tensor:
    """Route dequantize_blockwise through the hand-written Metal kernel.

    The kernel computes out[i] = code[A[i]] * absmax[i // blocksize] in fp32; Python casts to
    the requested dtype (matching the reference's trailing .to(dtype)).
    """
    A_flat = _ensure_native_buffer(A.reshape(-1))
    if A_flat.dtype != torch.uint8:
        A_flat = _ensure_native_buffer(A_flat.view(torch.uint8))
    code_f = _ensure_native_buffer(code.to(torch.float32))
    absmax_f = _ensure_native_buffer(absmax.to(torch.float32))

    n = A_flat.numel()
    out = torch.empty(n, dtype=torch.float32, device=A.device)

    torch.mps.synchronize()
    _mps_native.bnb_mps_dequantize_blockwise(
        code_f.data_ptr(),
        A_flat.data_ptr(),
        absmax_f.data_ptr(),
        out.data_ptr(),
        n,
        blocksize,
    )

    return out.reshape(A.shape).to(dtype)


# NOTE: dequantize_blockwise was previously MISSING on the mps backend (fell through to the
# "default" pure-torch kernel). This adds a real mps registration -- native when available,
# else the same pure-torch compute the default backend uses.
@register_kernel("bitsandbytes::dequantize_blockwise", "mps")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    if _native_available():
        return _dequantize_blockwise_native(A, absmax, code, blocksize, dtype)
    return _dequantize_blockwise_compute(A.reshape(-1), absmax, code, blocksize, dtype).reshape(A.shape)


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


def _quantize_4bit_native(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route quantize_4bit (NF4/FP4) through the hand-written Metal kernel.

    `bounds` are the 15 midpoints of the sorted code; `order` remaps the searchsorted index
    back to the stored 4-bit index (identity for NF4, argsort for FP4). The kernel packs pairs
    into bytes (high nibble = even element). Storage-dtype reinterpret mirrors the reference.
    """
    bounds, order = _get_4bit_quantize_bounds(quant_type, A.device)
    bounds_f = _ensure_native_buffer(bounds.to(torch.float32))
    order_u8 = _ensure_native_buffer(order.to(torch.uint8))
    A_flat = _ensure_native_buffer(A.reshape(-1).to(torch.float32))

    n = A_flat.numel()
    blocks = -(n // -blocksize)
    n_packed = -(n // -2)  # ceil(n/2)
    out = torch.empty(n_packed, dtype=torch.uint8, device=A.device)
    absmax = torch.empty(blocks, dtype=torch.float32, device=A.device)

    torch.mps.synchronize()
    _mps_native.bnb_mps_quantize_4bit(
        bounds_f.data_ptr(),
        order_u8.data_ptr(),
        A_flat.data_ptr(),
        out.data_ptr(),
        absmax.data_ptr(),
        n,
        blocksize,
    )

    packed = out.unsqueeze(1)
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
    if _native_available():
        return _quantize_4bit_native(A, blocksize, quant_type, quant_storage)
    if blocksize in (64, 128, 256, 512) and (k := _get_kernel()) is not None:
        packed, absmax = k.quantize_4bit(A.contiguous(), blocksize, _QUANT_MAP[quant_type])
        packed = packed.view(quant_storage).unsqueeze(1)
        return packed, absmax
    return _quantize_4bit_fallback(A, blocksize, quant_type, quant_storage)


def _dequantize_4bit_native(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Route dequantize_4bit (NF4/FP4) through the hand-written Metal kernel.

    out[j] = code4[nibble_j] * absmax[j // blocksize] in fp32; Python casts to dtype and
    reshapes (matching the reference). `A` holds packed nibbles; `absmax` is the plain
    per-block scale (nested/compressed absmax is unpacked by the caller before this op).
    """
    A_flat = _ensure_native_buffer(A.reshape(-1))
    code_f = _ensure_native_buffer(_get_4bit_code(quant_type, A.device).to(torch.float32))
    absmax_f = _ensure_native_buffer(absmax.to(torch.float32))

    n = prod(shape)
    out = torch.empty(n, dtype=torch.float32, device=A.device)

    torch.mps.synchronize()
    _mps_native.bnb_mps_dequantize_4bit(
        code_f.data_ptr(),
        A_flat.data_ptr(),
        absmax_f.data_ptr(),
        out.data_ptr(),
        n,
        blocksize,
    )

    return out.reshape(shape).to(dtype)


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

    # Native hand-written Metal kernel when available (any blocksize).
    if _native_available():
        return _dequantize_4bit_native(A, absmax, blocksize, quant_type, shape, dtype)

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


def _gemv_4bit_native(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    """Route gemv_4bit (M == 1) through the fused hand-written Metal kernel.

    The kernel reads packed 4-bit B + per-block absmax + the 16-entry code table and
    computes out[n] = sum_k A[k] * dequant(B[n, k]) directly -- the dequantized B is never
    materialized. Dequantized weights are rounded to A's dtype in-kernel (reproducing the
    reference's B_dq.to(dtype)); accumulation is fp32, so only accumulation order differs
    from the oracle. A and out bind in A's own dtype (per-dtype kernel variants), so the
    steady-state call launches no torch cast kernels. Preconditions (checked by the
    caller): K % 32 == 0 and power-of-two blocksize.
    """
    N, K = int(shapeB[0]), int(shapeB[-1])

    B_flat = B if B.dtype == torch.uint8 else B.view(torch.uint8)
    B_flat = _ensure_native_buffer(B_flat.reshape(-1))
    A_flat = _ensure_native_buffer(A.reshape(-1))
    code_f = _ensure_native_buffer(code.to(torch.float32))
    absmax_f = _ensure_native_buffer(absmax.to(torch.float32))

    out = torch.empty(N, dtype=A.dtype, device=A.device)
    dtype_flag = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[A.dtype]
    bs_shift = blocksize.bit_length() - 1

    torch.mps.synchronize()
    _mps_native.bnb_mps_gemv_4bit(
        code_f.data_ptr(),
        B_flat.data_ptr(),
        absmax_f.data_ptr(),
        A_flat.data_ptr(),
        out.data_ptr(),
        K,
        N,
        bs_shift,
        dtype_flag,
    )

    return out.reshape(*A.shape[:-1], N)


def _gemv_4bit_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    # Fused native Metal kernel when available. Guards: true gemv (M == 1), 2-D shapeB
    # matching A's K, K % 32 == 0 (uint4 row loads need 16-byte-aligned rows), power-of-two
    # blocksize (the kernel indexes absmax with a shift), and a plain 16-entry code table
    # whose packed B has the expected size.
    if (
        _native_available()
        and hasattr(_mps_native._lib, "bnb_mps_gemv_4bit")  # stale dylibs predate the fused kernel
        and A.numel() == A.shape[-1]
        and len(shapeB) == 2
        and shapeB[-1] == A.shape[-1]
        and shapeB[-1] % 32 == 0
        and blocksize >= 32
        and (blocksize & (blocksize - 1)) == 0
        and code.numel() == 16
        and B.numel() * B.element_size() == (shapeB[0] * shapeB[1]) // 2
    ):
        return _gemv_4bit_native(A, B, shapeB, absmax, code, blocksize)

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


def _bf16_gemm_enabled(backward: bool = False) -> bool:
    """Whether the native bf16 gemm may run for this direction.

    Two kinds of gate. The capability symbol is the correctness one: a dylib with
    bnb_mps_gemm_4bit but built before bf16 support would take dtype_flag=2, fall through to
    fp32 element size, and read a bf16 scratch as float -- silent garbage, no crash.

    The env vars are the measurement ones, and they are per-direction on purpose. Forward and
    backward are separate phases of a training step with separate costs, so a single switch
    could only ever answer "both or neither"; these answer "what did the backward alone buy",
    in one session against one binary:
        BNB_MPS_DISABLE_BF16_GEMM=1      -- forces the fallback for BOTH directions
        BNB_MPS_DISABLE_BF16_GEMM_BWD=1  -- forces it for the backward only
    A speedup you cannot switch off is a speedup you cannot verify.
    """
    if os.environ.get("BNB_MPS_DISABLE_BF16_GEMM", "") not in ("", "0"):
        return False
    if backward and os.environ.get("BNB_MPS_DISABLE_BF16_GEMM_BWD", "") not in ("", "0"):
        return False
    return hasattr(_mps_native._lib, "bnb_mps_gemm_4bit_supports_bf16")


def _gemm_4bit_native(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Route gemm_4bit (general M) through the native Metal entry point.

    One command buffer, one commit, one blocking wait: a chunked Metal kernel dequantizes
    packed B into a private scratch MTLBuffer in A's dtype (reproducing the reference's
    B_dq.to(dtype) rounding), MPSMatrixMultiplication computes A[M,K] . B_dq[N,K]^T, and an
    optional bias epilogue adds bias[N] -- all on the same command buffer. Compared with the
    dequant + F.linear fallback this removes the torch round-trip and its second sync (the
    per-call cross-queue sync is what dominates wall-clock at small/medium M -- see the
    Phase M2 finding in MPS_STATUS.md).

    fp32/fp16 run the GEMM through MPSMatrixMultiplication; bf16 runs it through MPSGraph,
    which has a bf16 matmul where MPSMatrixMultiplication hard-asserts on anything but
    fp32/fp16/int8/int16. Same structure either way.

    `absmax` must already be the plain per-block fp32 scale: nested/compressed absmax is
    unpacked by the caller BEFORE this function. Preconditions (checked by the caller):
    A.dtype is fp32/fp16/bf16, K % 32 == 0, and power-of-two blocksize >= 32.
    """
    N, K = int(shapeB[0]), int(shapeB[-1])
    M = A.numel() // K

    B_flat = B if B.dtype == torch.uint8 else B.view(torch.uint8)
    B_flat = _ensure_native_buffer(B_flat.reshape(-1))
    A_flat = _ensure_native_buffer(A.reshape(-1))
    code_f = _ensure_native_buffer(_get_4bit_code(quant_type, A.device).to(torch.float32))
    absmax_f = _ensure_native_buffer(absmax.to(torch.float32))
    bias_ptr = None
    if bias is not None:
        bias_f = _ensure_native_buffer(bias.reshape(-1))
        bias_ptr = bias_f.data_ptr()

    out = torch.empty(M * N, dtype=A.dtype, device=A.device)
    dtype_flag = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[A.dtype]
    bs_shift = blocksize.bit_length() - 1

    torch.mps.synchronize()
    _mps_native.bnb_mps_gemm_4bit(
        code_f.data_ptr(),
        B_flat.data_ptr(),
        absmax_f.data_ptr(),
        A_flat.data_ptr(),
        bias_ptr,
        out.data_ptr(),
        M,
        K,
        N,
        bs_shift,
        dtype_flag,
    )

    return out.reshape(*A.shape[:-1], N)


def _gemm_4bit_backward_native(
    grad_output: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
) -> torch.Tensor:
    """grad_A[M, K] = grad_output[M, N] . B_dq[N, K], one command buffer / one commit / one wait.

    Structurally identical to _gemm_4bit_native, and deliberately so: the same chunked dequant
    fills the same private scratch, and only the matmul orientation differs (no transpose here --
    dequantize_4bit already emits [N, K], which is the orientation grad_A wants).

    The win is the sync, not the arithmetic. The composition this replaces runs a native dequant
    on our queue (wait), hands the result to torch, and runs a matmul on torch's queue (wait) --
    twice the cross-queue round trip, per Linear4bit, per step.

    `absmax` must already be the plain per-block fp32 scale. Preconditions are the caller's.
    """
    N, K = int(shapeB[0]), int(shapeB[-1])
    M = grad_output.numel() // N

    B_flat = B if B.dtype == torch.uint8 else B.view(torch.uint8)
    B_flat = _ensure_native_buffer(B_flat.reshape(-1))
    G_flat = _ensure_native_buffer(grad_output.reshape(-1))
    code_f = _ensure_native_buffer(_get_4bit_code(quant_type, grad_output.device).to(torch.float32))
    absmax_f = _ensure_native_buffer(absmax.to(torch.float32))

    out = torch.empty(M * K, dtype=grad_output.dtype, device=grad_output.device)
    dtype_flag = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[grad_output.dtype]
    bs_shift = blocksize.bit_length() - 1

    torch.mps.synchronize()
    _mps_native.bnb_mps_gemm_4bit_bwd(
        code_f.data_ptr(),
        B_flat.data_ptr(),
        absmax_f.data_ptr(),
        G_flat.data_ptr(),
        out.data_ptr(),
        M,
        K,
        N,
        bs_shift,
        dtype_flag,
    )

    return out.reshape(*grad_output.shape[:-1], K)


@register_kernel("bitsandbytes::gemm_4bit_backward", "mps")
def _(
    grad_output: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    absmax_8bit: Optional[torch.Tensor] = None,
    absmax_code: Optional[torch.Tensor] = None,
    absmax_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    N, K = int(shapeB[0]), int(shapeB[-1])

    if absmax_8bit is not None:
        absmax = (
            torch.ops.bitsandbytes.dequantize_blockwise.default(absmax_8bit, absmax, absmax_code, 256, torch.float32)
            + absmax_offset
        )

    # Same guards as the forward, minus bias (there is none) and with the inner dim being N.
    # K % 32 == 0 is still what the chunked dequant needs; it fills B_dq[N, K] either way.
    if (
        _native_available()
        and hasattr(_mps_native._lib, "bnb_mps_gemm_4bit_bwd")  # dylibs before the fused backward
        and (
            grad_output.dtype in (torch.float32, torch.float16)
            or (grad_output.dtype == torch.bfloat16 and _bf16_gemm_enabled(backward=True))
        )
        and len(shapeB) == 2
        and grad_output.shape[-1] == N
        and K % 32 == 0
        and blocksize >= 32
        and (blocksize & (blocksize - 1)) == 0
        and B.numel() * B.element_size() == (N * K) // 2
    ):
        return _gemm_4bit_backward_native(grad_output, B, shapeB, absmax, blocksize, quant_type)

    B_dq = _dequantize_4bit_impl(B, absmax, blocksize, quant_type, shapeB, grad_output.dtype)
    return torch.matmul(grad_output, B_dq)


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

    # Native Metal path (dequant -> scratch -> MPSMatrixMultiplication (fp32/fp16) or MPSGraph
    # (bf16) -> bias, one command buffer / one sync). Guards: 2-D shapeB matching A's K,
    # K % 32 == 0 (uint4 loads in the chunked dequant kernel), power-of-two blocksize (absmax
    # indexed with a shift), a bias matching out's dtype and width, and packed B of the
    # expected size. bf16 additionally requires the capability marker, since a dylib with
    # bnb_mps_gemm_4bit but no bf16 support would misread the scratch as fp32.
    if (
        _native_available()
        and hasattr(_mps_native._lib, "bnb_mps_gemm_4bit")  # stale dylibs predate the native gemm
        and (
            A.dtype in (torch.float32, torch.float16)
            or (A.dtype == torch.bfloat16 and _bf16_gemm_enabled())
        )
        and len(shapeB) == 2
        and shapeB[-1] == K
        and K % 32 == 0
        and blocksize >= 32
        and (blocksize & (blocksize - 1)) == 0
        and (bias is None or (bias.dtype == A.dtype and bias.numel() == N))
        and B.numel() * B.element_size() == (N * K) // 2
    ):
        return _gemm_4bit_native(A, B, shapeB, absmax, blocksize, quant_type, bias)

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
