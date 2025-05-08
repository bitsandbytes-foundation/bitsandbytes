from collections.abc import Sequence
import ctypes as ct
from typing import Optional

import torch

from bitsandbytes.functional import get_ptr

from ..._ops import register_kernel
from ...cextension import lib

# torch._int_mm for s8@s8->s32 is supported on CPU from torch 2.4+.
# However, we can overflow if we use this without AVX512_VNNI support.
# This is fixed in torch 2.6+, so we set this as the minimum to be safe.
# For more information: https://github.com/pytorch/pytorch/pull/136942
# TODO(matthewdouglas): aarch64?
if torch.__version__ >= (2, 6):

    @register_kernel("bitsandbytes::int8_linear_matmul", "cpu")
    def _(A: torch.Tensor, B: torch.Tensor):
        return torch._int_mm(
            A.reshape(-1, A.shape[-1]),
            B.t(),
        ).reshape(*A.shape[:-1], B.shape[0])


@register_kernel("bitsandbytes::int8_mm_dequant", "cpu")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
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

    return out.to(dtype or torch.float16)


@register_kernel("bitsandbytes::quantize_blockwise", "cpu")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
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


_NF4_QUANT_TABLE = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
    device="cpu",
)
_FP4_QUANT_TABLE = torch.tensor(
    [
        0.0000,
        0.0052,
        0.6667,
        1.0000,
        0.3333,
        0.5000,
        0.1667,
        0.2500,
        0.0000,
        -0.0052,
        -0.6667,
        -1.0000,
        -0.3333,
        -0.5000,
        -0.1667,
        -0.2500,
    ],
    dtype=torch.float32,
    device="cpu",
)
CODE = {"nf4": _NF4_QUANT_TABLE, "fp4": _FP4_QUANT_TABLE}


@register_kernel("bitsandbytes::quantize_4bit", "cpu")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(quant_type in ("nf4", "fp4"), lambda: f"quant_type must be nf4 or fp4 on CPU, got {quant_type}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    rem = n % blocksize
    has_rem = rem > 0

    # Scale tensor to [-1, 1]
    absmax = torch.zeros((blocks,), device=A.device, dtype=A.dtype)
    A_reshaped = A.reshape(n)
    A_com_reshaped = A_reshaped[: n - rem].reshape(n // blocksize, blocksize)
    absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
    scaled = torch.clamp(A_com_reshaped * (1 / absmax[: blocks - has_rem].view(-1, 1)), -1, 1)
    scaled = scaled.reshape(-1)
    if has_rem:
        absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
        scaled_rem = torch.clamp(A_reshaped[n - rem :] * (1 / absmax[-1]), -1, 1)
        scaled = torch.cat([scaled, scaled_rem], dim=0)
    # Quantize with the lookup table
    quant_table = CODE[quant_type]
    quantized = torch.argmin(torch.abs(scaled.view(-1, 1) - quant_table), dim=-1, keepdim=True).to(torch.uint8)

    # Pack two quantized values per byte
    packed = quantized[::2] << 4 | quantized[1::2]

    if quant_storage != torch.uint8:
        packed = packed.squeeze().view(quant_storage).unsqueeze(1)

    return packed, absmax.float()


@register_kernel("bitsandbytes::dequantize_4bit", "cpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(quant_type in ("nf4", "fp4"), lambda: f"quant_type must be nf4 or fp4 on CPU, got {quant_type}")
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )

    # Enable non uint8 dtype
    device = A.device
    if A.dtype != torch.uint8:
        bytes_value = A.cpu().numpy().tobytes()
        A = torch.frombuffer(bytes_value, dtype=torch.uint8).to(device)

    A = A.reshape(-1)
    # Map nf4 to [-1, 1]
    out_dq = torch.empty(A.size(0) * 2, dtype=torch.int32, device=A.device)
    n = out_dq.numel()
    out_dq[1::2] = A & 0xF
    out_dq[::2] = A >> 4
    # code is fp32, cast to dtype to avoid the mismatch issue
    code = CODE[quant_type].to(dtype)
    out_dq = code[out_dq]

    # Apply scales
    if out_dq.numel() != n:
        assert out_dq.numel() == n + 1
        out_dq = torch.narrow(out_dq, 0, 0, n)
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    rem = n % blocksize
    has_rem = rem > 0

    out = torch.empty(shape, dtype=dtype, device=A.device).reshape(-1)
    if has_rem:
        out[: n - rem] = (out_dq[: n - rem].view(-1, blocksize) * absmax[: blocks - has_rem].view(-1, 1)).reshape(-1)
        out[n - rem :] = out_dq[n - rem :] * absmax[-1]
    else:
        out = out_dq.view(-1, blocksize) * absmax.view(-1, 1)

    out = out.reshape(-1, *shape[1:]).to(dtype)

    return out


@register_kernel("bitsandbytes::gemv_4bit", "cpu")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    # Applied from dequantize_4bit
    B = B.view(-1, 1)
    upper = (B >> 4).to(torch.int64)
    lower = (B & 0x0F).to(torch.int64)
    blocks = torch.cat((upper, lower), dim=1).reshape(-1, blocksize)
    B_dq = code[blocks] * absmax[:, None]
    B_dq = B_dq.reshape(-1, *shapeB[1:]).to(A.dtype)

    # User called gemv with B.t(), so we need to transpose it back.
    # if B.shape[0] == 1:
    #    B_dq = B_dq.t()

    return torch.nn.functional.linear(
        A,
        B_dq,
        bias=None,
    )
