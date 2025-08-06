from collections.abc import Sequence
import warnings

import torch

from ..._ops import register_kernel
from ..utils import ipex_xpu, triton_available

# _int_mm is available in torch starting from 2.7 version,
# but currently it's don't have xpu implementation.
if ipex_xpu and torch.__version__ >= (2, 7):

    @register_kernel("bitsandbytes::int8_linear_matmul", "xpu")
    def _(A: torch.Tensor, B: torch.Tensor):
        return torch._int_mm(
            A.reshape(-1, A.shape[-1]),
            B.t(),
        ).reshape(*A.shape[:-1], B.shape[0])


# IPEX should be faster for xpu, so at first checking if it is available.
if ipex_xpu:

    @register_kernel("bitsandbytes::dequantize_nf4_ipex", "xpu")
    def _(
        A: torch.Tensor,
        absmax: torch.Tensor,
        blocksize: int,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.ops.torch_ipex.dequantize_4bit(A, "nf4", shape, absmax, None, blocksize).t().to(dtype)

    @register_kernel("bitsandbytes::dequantize_blockwise", "xpu")
    def _(
        A: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = A.shape
        out = torch.empty(A.reshape(-1).shape, dtype=dtype, device=A.device)
        # void cdequantize_blockwise_fp32(
        # float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, cudaStream_t stream)
        if dtype == torch.float16:
            ipex_xpu.xpu.bitsandbytes.cdequantize_blockwise_fp16(code, A, absmax, out, blocksize, A.numel())
        elif dtype == torch.bfloat16:
            ipex_xpu.xpu.bitsandbytes.cdequantize_blockwise_bf16(code, A, absmax, out, blocksize, A.numel())
        elif dtype == torch.float32:
            ipex_xpu.xpu.bitsandbytes.cdequantize_blockwise_fp32(code, A, absmax, out, blocksize, A.numel())
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {out.dtype}")

        return out.reshape(shape)
elif triton_available:
    from ..triton import ops as triton_ops

    register_kernel("bitsandbytes::quantize_blockwise", "xpu")(triton_ops.quantize_blockwise)
    register_kernel("bitsandbytes::dequantize_blockwise.out", "xpu")(triton_ops.dequantize_blockwise_inplace)
    register_kernel("bitsandbytes::dequantize_blockwise", "xpu")(triton_ops.dequantize_blockwise)
    register_kernel("bitsandbytes::quantize_4bit", "xpu")(triton_ops.quantize_4bit)
    register_kernel("bitsandbytes::dequantize_4bit.out", "xpu")(triton_ops.dequantize_4bit_inplace)
    register_kernel("bitsandbytes::dequantize_4bit", "xpu")(triton_ops.dequantize_4bit)
    register_kernel("bitsandbytes::gemv_4bit", "xpu")(triton_ops.gemv_4bit)
    register_kernel("bitsandbytes::optimizer_update_32bit", "xpu")(triton_ops.optimizer_update_32bit)
else:
    warnings.warn("XPU available but no ipex or triton packages found.")
