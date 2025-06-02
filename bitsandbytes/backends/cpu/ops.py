from collections.abc import Sequence
import ctypes as ct

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


@register_kernel("bitsandbytes::quantize_4bit", "cpu")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 on CPU, got {quant_type}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()

    # TODO: Support when weight matrix is not divisible by blocksize
    torch._check(n % blocksize == 0, lambda: f"n must be divisible by blocksize, got {n} and {blocksize}")

    # Divide into blocks and normalize
    blocks = A.reshape(-1, blocksize)
    absmax = blocks.abs().max(dim=1).values.float()
    scaled = blocks / absmax.unsqueeze(-1)

    # Quantize with the lookup table
    quantized = torch.argmin(torch.abs(scaled.view(-1, 1) - _NF4_QUANT_TABLE), dim=-1, keepdim=True).to(torch.uint8)

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
    torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 on CPU, got {quant_type}")
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )
    torch._check(
        A.dtype == torch.uint8,
        lambda: f"Blockwise 4bit dequantization on CPU only supports uint8 storage, got {A.dtype}",
    )

    A = A.view(-1, 1)

    # Grab upper and lower nibbles. Using int64 for indexing in the LUT.
    upper = (A >> 4).to(torch.int64)
    lower = (A & 0x0F).to(torch.int64)

    # Expand to blocks
    blocks = torch.cat((upper, lower), dim=1).reshape(-1, blocksize)

    # Dequantize
    blocks = _NF4_QUANT_TABLE[blocks] * absmax[:, None]

    # Reshape to original shape
    blocks = blocks.reshape(-1, *shape[1:])

    return blocks.to(dtype)


@register_kernel("bitsandbytes::gemv_4bit", "cpu")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    # TODO: We need to determine whether `code` is NF4, FP4, or other.
    # Right now we assume NF4, as this is the only one supported on CPU.

    B_dq = torch.ops.bitsandbytes.dequantize_4bit.default(
        B,
        absmax,
        blocksize,
        "nf4",
        shape=shapeB,
        dtype=A.dtype,
    )

    # User called gemv with B.t(), so we need to transpose it back.
    # if B.shape[0] == 1:
    #    B_dq = B_dq.t()

    return torch.nn.functional.linear(
        A,
        B_dq,
        bias=None,
    )
