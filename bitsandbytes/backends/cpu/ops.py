from collections.abc import Sequence
import ctypes as ct
import logging

import torch

from bitsandbytes.functional import get_ptr, has_avx512bf16

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
        out = torch.empty(shape, dtype=dtype, device=A.device)
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
            raise ValueError

        return out

    if has_avx512bf16():
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
            dtype = A.dtype
            quant_type = "fp4" if code[1] > 0 else "nf4"
            # cpu fused op only support bf16 for now.
            if dtype != torch.bfloat16:
                A = A.to(torch.bfloat16)

            A = A.reshape(-1, A.shape[-1])
            out_shape = (*A.shape[:-1], shapeB[0])
            out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
            M = A.shape[0]
            N = shapeB[0]
            K = A.shape[1]
            x_strideM = A.stride(0)
            out_strideM = out.stride(0)
            if quant_type == "fp4":
                lib.gemv_4bit_inference_cpu_fp4_bf16(
                    ct.c_int64(M),
                    ct.c_int64(N),
                    ct.c_int64(K),
                    get_ptr(A),
                    get_ptr(B),
                    get_ptr(absmax),
                    get_ptr(out),
                    ct.c_int64(blocksize),
                    ct.c_int64(x_strideM),
                    ct.c_int64(out_strideM),
                )
            elif quant_type == "nf4":
                lib.gemv_4bit_inference_cpu_nf4_bf16(
                    ct.c_int64(M),
                    ct.c_int64(N),
                    ct.c_int64(K),
                    get_ptr(A),
                    get_ptr(B),
                    get_ptr(absmax),
                    get_ptr(out),
                    ct.c_int64(blocksize),
                    ct.c_int64(x_strideM),
                    ct.c_int64(out_strideM),
                )

            if dtype != torch.bfloat16:
                out = out.to(dtype)

            return out

# def unpack_weight_packed_for_cpu(packed_qweight: torch.Tensor, block_n: int = 32):
#     """
#     Inverse of convert_weight_packed_for_cpu.
#     packed_qweight: (N, K//2) uint8, each byte = (high<<4)|low, both 4-bit values in 0..15
#     returns: qweight_final (N, K) uint8 with original 4-bit values (0..15)
#     """
#     assert packed_qweight.dtype == torch.uint8
#     assert packed_qweight.dim() == 2
#     N, K_half = packed_qweight.shape
#     assert N % block_n == 0
#     BIT_COUNT = block_n  # 32
#     # reshape to rows of 32 packed bytes
#     qw = packed_qweight.reshape(-1, BIT_COUNT)           # [(N//block_n)*K_half, 32]
#     low  = (qw & 0x0F)
#     high = (qw >> 4) & 0x0F
#     # restore 64 nibbles (low first then high, matching original pack order)
#     restored = torch.cat([low, high], dim=1)             # [..., 64]
#     # reshape back (inverse of flatten)
#     restored = restored.reshape(N // block_n, K_half, block_n, 2)  # [N/block_n, K//2, block_n, 2]
#     # inverse transpose
#     restored = restored.transpose(-3, -2)                # [N/block_n, block_n, K//2, 2]
#     # final shape
#     qweight_final = restored.reshape(N, K_half * 2).to(torch.uint8)
#     return qweight_final


# _NF4_QUANT_TABLE = torch.tensor([ -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
#                                   0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0 ], dtype=torch.float32)

# def fused_matmul(x, packed_weight, scales, group_size):
#     unpacked_weight = unpack_weight_packed_for_cpu(packed_weight)
#     shape = unpacked_weight.shape
#     # original_weight = _INT4_0_TO_15_TABLE[unpacked_weight.reshape(-1).int()].reshape(shape) * scales.T.repeat_interleave(group_size, dim=1)
#     original_weight = _NF4_QUANT_TABLE[unpacked_weight.reshape(-1).int()].reshape(shape) * scales.T.repeat_interleave(group_size, dim=1)
#     res = torch.matmul(x, original_weight.T.to(x.dtype))
#     return res
