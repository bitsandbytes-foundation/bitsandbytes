from collections.abc import Sequence
import ctypes as ct
import logging

import torch

from bitsandbytes.functional import get_ptr

from ..utils import CODE
from ..._ops import register_kernel
from ...cextension import ErrorHandlerMockBNBNativeLibrary, lib

logger = logging.getLogger(__name__)

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


if not isinstance(lib, ErrorHandlerMockBNBNativeLibrary):

    @register_kernel("bitsandbytes::quantize_blockwise", "cpu")
    def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
        torch._check_is_size(blocksize)

        n = A.numel()

        # Only FP32 has c++ kernrl
        if A.dtype == torch.float32:
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
        else:
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

    @register_kernel("bitsandbytes::dequantize_blockwise", "cpu")
    def _(
        A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype
    ) -> torch.Tensor:
        torch._check_is_size(blocksize)
        torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")

        out = torch.empty_like(A, dtype=dtype)
        if dtype == torch.float32:
            lib.cdequantize_blockwise_cpu_fp32(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(A.numel()),
            )
        elif dtype == torch.bfloat16:
            lib.cdequantize_blockwise_cpu_bf16(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(A.numel()),
            )
        elif dtype == torch.float16:
            lib.cdequantize_blockwise_cpu_fp16(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(A.numel()),
            )
        else:
            out = code[A.reshape(-1).int()]
            blocks = out.shape[-1] // blocksize
            res = out.shape[-1] % blocksize
            if res != 0:
                out = torch.nn.functional.pad(out, (0, blocksize - res), mode="constant", value=0)
            out = (out.view(-1, blocksize) * absmax.view(-1, 1)).to(dtype).reshape(-1)
            out = out[: blocks * blocksize + res]
            out = out.reshape(A.shape)

        return out

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
    torch._check(quant_type in ("nf4", "fp4"), lambda: f"quant_type must be nf4 or fp4, got {quant_type}")
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )
    # Enable non uint8 dtype
    if A.dtype != torch.uint8:
        A = A.view(torch.uint8)

    # TODO: support half precision absmax
    if absmax.dtype != torch.float32:
        absmax = absmax.float()

    A = A.reshape(shape[0], shape[1] // 2)
    out = torch.empty(shape, dtype=dtype, device=A.device).reshape(-1)
    if quant_type == "fp4":
        if dtype == torch.float32:
            lib.cdequantize_blockwise_cpu_fp4_fp32(
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(shape[0]),
                ct.c_longlong(shape[1]),
            )
        elif dtype == torch.bfloat16:
            lib.cdequantize_blockwise_cpu_fp4_bf16(
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(shape[0]),
                ct.c_longlong(shape[1]),
            )
        elif dtype == torch.float16:
            lib.cdequantize_blockwise_cpu_fp4_fp16(
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(shape[0]),
                ct.c_longlong(shape[1]),
            )
    elif quant_type == "nf4":
        if dtype == torch.float32:
            lib.cdequantize_blockwise_cpu_nf4_fp32(
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(shape[0]),
                ct.c_longlong(shape[1]),
            )
        elif dtype == torch.bfloat16:
            lib.cdequantize_blockwise_cpu_nf4_bf16(
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(shape[0]),
                ct.c_longlong(shape[1]),
            )
            out_2 = dequantize_nf4_test(A.reshape(-1), absmax, blocksize, quant_type, shape, dtype)
            out = out.reshape(shape)
            out_2 = out_2.reshape(shape)
            if not torch.allclose(out, out_2, rtol=1e-2, atol=5e-2):
                import pdb; pdb.set_trace()
        elif dtype == torch.float16:
            lib.cdequantize_blockwise_cpu_nf4_fp16(
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(shape[0]),
                ct.c_longlong(shape[1]),
            )
    else:
        # Map nf4 to [-1, 1]
        A = A.reshape(-1)
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

        if has_rem:
            out[: n - rem] = (out_dq[: n - rem].view(-1, blocksize) * absmax[: blocks - has_rem].view(-1, 1)).reshape(-1)
            out[n - rem :] = out_dq[n - rem :] * absmax[-1]
        else:
            out = out_dq.view(-1, blocksize) * absmax.view(-1, 1)

        out = out.reshape(-1, *shape[1:]).to(dtype)

    return out

def dequantize_nf4_test(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
):
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

    if has_rem:
        out[: n - rem] = (out_dq[: n - rem].view(-1, blocksize) * absmax[: blocks - has_rem].view(-1, 1)).reshape(-1)
        out[n - rem :] = out_dq[n - rem :] * absmax[-1]
    else:
        out = out_dq.view(-1, blocksize) * absmax.view(-1, 1)

    out = out.reshape(-1, *shape[1:]).to(dtype)

    return out


def _reverse_4bit_compress_format(weight: torch.Tensor):
    out_1 = (weight & 0xF0) >> 4
    out_2 = (weight & 0xF) << 4
    out = out_1 | out_2
    return out
