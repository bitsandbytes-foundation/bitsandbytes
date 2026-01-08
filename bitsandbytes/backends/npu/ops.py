import ctypes as ct
from collections.abc import Sequence

import torch

from bitsandbytes.functional import get_ptr

from ..._ops import register_kernel
from ...cextension import lib
from ..utils import _NF4_QUANT_TABLE


@register_kernel("bitsandbytes::quantize_4bit", "npu")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 on NPU, got {quant_type}")
    n = A.numel()

    global _NF4_QUANT_TABLE
    if _NF4_QUANT_TABLE.device != A.device:
        _NF4_QUANT_TABLE = _NF4_QUANT_TABLE.to(A.device)

    # TODO: Support when weight matrix is not divisible by blocksize
    torch._check(n % blocksize == 0, lambda: f"n must be divisible by blocksize, got {n} and {blocksize}")

    # Process tensor in chunks to avoid high memory usage from large intermediate tensors
    # (e.g., during broadcasting with FP32 quant table)
    chunks_absmax = []
    chunks_out = []
    total_blocks = A.numel() // blocksize
    chunks = 8 if A.numel() > 1024 * 1024 else 1
    chunksize = (total_blocks + chunks - 1) // chunks

    for i in range(chunks):
        start = i * chunksize * blocksize
        end = min((i + 1) * chunksize * blocksize, A.numel())
        chunk_data = A.view(-1)[start:end].view(-1, blocksize)

        absmax = chunk_data.abs().max(dim=1, keepdim=True).values
        chunks_absmax.append(absmax)

        a = chunk_data / absmax.float()
        diff = torch.abs(a.unsqueeze(-1) - _NF4_QUANT_TABLE)
        out = (torch.argmin(diff, dim=-1) + 8) % 16

        out = out.reshape(-1, 2)
        # Pack 4-bit values in NPU-compatible order (low nibble first) to match NPU-specific unpacking logic;
        # differs from CUDA's packing
        out = (out[:, 0] + out[:, 1] * 16).to(torch.uint8)
        chunks_out.append(out)

    absmax = torch.cat(chunks_absmax, dim=0)
    packed = torch.cat(chunks_out, dim=0).reshape(-1, 1)
    return packed, absmax


@register_kernel("bitsandbytes::dequantize_4bit", "npu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty(shape, dtype=dtype, device=A.device)
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)
    return out


@register_kernel("bitsandbytes::dequantize_4bit.out", "npu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(out.shape == shape, lambda: f"Expected out.shape == {shape}, got {out.shape}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)


def _dequantize_4bit_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    torch._check(quant_type in ["nf4"])
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )
    if out.dtype == torch.bfloat16:
        # bf16: bf16 -> fp32 -> op -> fp32 -> bf16
        absmax = absmax.to(torch.float32)
        out_fp32 = torch.empty(out.shape, dtype=torch.float32, device=out.device)
    else:
        out_fp32 = out

    args = (
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out_fp32),
        ct.c_int(blocksize),
        ct.c_int(out.numel()),
        torch.npu.current_stream(),
    )

    if out.dtype == torch.bfloat16:
        lib.cdequantize_blockwise_fp32_nf4(*args)
        out.copy_(out_fp32.to(torch.bfloat16))
    elif out.dtype == torch.float16:
        lib.cdequantize_blockwise_fp16_nf4(*args)
    elif out.dtype == torch.float32:
        lib.cdequantize_blockwise_fp32_nf4(*args)
