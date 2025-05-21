from collections.abc import Sequence
from math import prod
from typing import Optional

import torch

from ..._ops import register_kernel
from ..utils import CODE


@register_kernel("bitsandbytes::int8_mm_dequant", "default")
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


@register_kernel("bitsandbytes::int8_mixed_scaled_mm", "default")
def _(
    A: torch.Tensor,
    CA: torch.Tensor,
    CB: torch.Tensor,
    SCA: torch.Tensor,
    SCB: torch.Tensor,
    outlier_cols: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    subB = None

    if outlier_cols is not None and outlier_cols.numel():
        # Extract the inputs with outliers in original precision
        subA = A[:, outlier_cols].contiguous()

        # Dequantize the corresponding weight columns
        subB = (
            torch.ops.bitsandbytes.int8_vectorwise_dequant.default(CB[:, outlier_cols].contiguous(), SCB)
            .to(A.dtype)
            .t()
        )

        # TODO: if state.has_fp16_weights: subB = B[:, outlier_cols].t()

    else:
        # Needed for torch.compile when there are no outliers.
        subA = torch.empty(0, device=A.device, dtype=A.dtype)

    # Int8 Matmul + Dequant + Bias
    output = torch.ops.bitsandbytes.int8_scaled_mm.default(CA, CB, SCA, SCB, bias=bias, dtype=A.dtype)

    if subB is not None:
        # Add the outlier columns back to the output
        output = output.addmm(subA, subB)

    return output, subA


@register_kernel("bitsandbytes::int8_scaled_mm", "default")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    out_i32 = torch.ops.bitsandbytes.int8_linear_matmul.default(A, B)
    return torch.ops.bitsandbytes.int8_mm_dequant.default(
        out_i32,
        row_stats,
        col_stats,
        dtype=dtype or torch.float16,
        bias=bias,
    )


@register_kernel("bitsandbytes::int8_linear_matmul", "default")
def _(A: torch.Tensor, B: torch.Tensor):
    return _int8_linear_matmul_impl(A, B)


@register_kernel("bitsandbytes::int8_linear_matmul.out", "default")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    torch._check(out.dtype == torch.int32)
    _int8_linear_matmul_impl(A, B, out)


def _int8_linear_matmul_impl(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None):
    # Naive implementation: perform matmul in fp32
    result = torch.matmul(A.float(), B.float().t()).to(torch.int32)
    if out is not None:
        result = out.copy_(result)
    return result


@register_kernel("bitsandbytes::int8_vectorwise_quant", "default")
def _(A: torch.Tensor, threshold=0.0):
    rows = prod(A.shape[:-1])
    outlier_cols = None

    outlier_restore = None

    if threshold > 0.0:
        outliers = A.abs() >= threshold

        if outliers.any():
            # Determine which columns contain outliers, and zero out the
            # outliers ahead of quantization. We need to keep a backup of these
            # outliers to restore them after quantization.
            outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)
            outlier_restore = A[outliers].clone()
            A[outliers] = 0
        else:
            # Needed for torch.compile support.
            outlier_cols = torch.empty(0, device=A.device, dtype=torch.int64)

    # Get absmax for each row.
    row_stats = torch.max(A.abs(), dim=1).values.float()

    # Quantize row-wise to int8.
    out_row = torch.round(A * (127.0 / row_stats.unsqueeze(-1))).to(torch.int8)

    # Zero out values from outlier columns across all rows.
    if rows > 1 and outlier_cols is not None:
        out_row[:, outlier_cols] = 0

    # Restore outliers.
    if outlier_restore is not None:
        A[outliers] = outlier_restore

    return out_row, row_stats, outlier_cols


@register_kernel("bitsandbytes::quantize_blockwise", "default")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)

    n = A.numel()
    rem = n % blocksize
    has_rem = rem > 0
    blocks = n // blocksize + has_rem
    absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)
    A_reshaped = A.reshape(n)
    A_com = A_reshaped[: n - rem]
    A_com_reshaped = A_com.reshape(n // blocksize, blocksize)
    absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
    scaled_A = torch.clamp(A_com_reshaped * (1 / absmax[: blocks - has_rem].view(-1, 1)), -1, 1)
    scaled_A = scaled_A.reshape(-1)
    if has_rem:
        absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
        scaled_A_rem = torch.clamp(A_reshaped[n - rem :] * (1 / absmax[-1]), -1, 1)
        scaled_A = torch.cat([scaled_A, scaled_A_rem], dim=0)

    diff = torch.abs(scaled_A.unsqueeze(-1) - code.to(scaled_A.device))
    out = torch.argmin(diff, dim=-1).to(torch.uint8).to(scaled_A.device).reshape(A.shape)

    return out, absmax


@register_kernel("bitsandbytes::dequantize_blockwise", "default")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")

    out = code[A.reshape(-1).int()]
    blocks = out.shape[-1] // blocksize
    res = out.shape[-1] % blocksize
    if res != 0:
        out = torch.nn.functional.pad(out, (0, blocksize - res), mode="constant", value=0)
    out = (out.view(-1, blocksize) * absmax.view(-1, 1)).to(dtype).reshape(-1)
    out = out[: blocks * blocksize + res]
    out = out.reshape(A.shape)

    return out


@register_kernel("bitsandbytes::quantize_4bit", "default")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(quant_type in ("nf4", "fp4"), lambda: f"quant_type must be nf4 or fp4, got {quant_type}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()
    full_blocks = n // blocksize
    rem = n % blocksize
    blocks = full_blocks + 1 if rem else full_blocks
    absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)
    A_flattened = A.reshape(n)

    # Scale full blocks of the tensor to [-1, 1]
    A_full_blocks = A_flattened[: n - rem].reshape(n // blocksize, blocksize)
    absmax[:full_blocks] = torch.abs(A_full_blocks).max(dim=-1)[0]
    scaled = torch.clamp(A_full_blocks * (1 / absmax[:full_blocks].view(-1, 1)), -1, 1).reshape(-1)

    # Scale any partial block
    if rem:
        A_rem = A_flattened[-rem:]
        absmax[-1] = torch.abs(A_rem).max()
        scaled_rem = torch.clamp(A_rem * (1 / absmax[-1]), -1, 1)
        scaled = torch.cat([scaled, scaled_rem], dim=0)

    # Quantize with the lookup table
    code = CODE[quant_type].to(scaled.device).to(scaled.dtype)
    quantized = torch.argmin(torch.abs(scaled.view(-1, 1) - code), dim=-1, keepdim=True).to(torch.uint8)

    # Pack two quantized values per byte
    packed = quantized[::2] << 4 | quantized[1::2]

    if quant_storage != torch.uint8:
        packed = packed.squeeze().view(quant_storage).unsqueeze(1)

    return packed, absmax.float()


@register_kernel("bitsandbytes::dequantize_4bit", "default")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(quant_type in ("nf4", "fp4"), lambda: f"quant_type must be nf4 or fp4, got {quant_type}")
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )

    # Enable non uint8 dtype
    if A.dtype != torch.uint8:
        A = A.view(torch.uint8)

    A = A.reshape(-1)
    # Map nf4 to [-1, 1]
    out_dq = torch.empty(A.size(0) * 2, dtype=torch.int32, device=A.device)
    n = out_dq.numel()
    out_dq[1::2] = A & 0xF
    out_dq[::2] = A >> 4
    # code is fp32, cast to dtype to avoid the mismatch issue
    code = CODE[quant_type].to(dtype).to(A.device)
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


@register_kernel("bitsandbytes::gemv_4bit", "default")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    # Applied from dequantize_4bit
    quant_type = "fp4" if code[1] > 0 else "nf4"
    B_dq = torch.ops.bitsandbytes.dequantize_4bit.default(B, absmax, blocksize, quant_type, shapeB, A.dtype)

    return torch.nn.functional.linear(
        A,
        B_dq,
        bias=None,
    )
