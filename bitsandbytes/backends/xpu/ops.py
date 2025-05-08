from collections.abc import Sequence

import torch

import triton
import triton.language as tl

from ..._ops import register_kernel

# torch._int_mm for s8@s8->s32 is supported on CPU from torch 2.4+.
# However, we can overflow if we use this without AVX512_VNNI support.
# This is fixed in torch 2.6+, so we set this as the minimum to be safe.
# For more information: https://github.com/pytorch/pytorch/pull/136942
# TODO(matthewdouglas): aarch64?
# if torch.__version__ >= (2, 6):

#     @register_kernel("bitsandbytes::int8_linear_matmul", "xpu")
#     def _(A: torch.Tensor, B: torch.Tensor):
#         return torch._int_mm(
#             A.reshape(-1, A.shape[-1]),
#             B.t(),
#         ).reshape(*A.shape[:-1], B.shape[0])


# @triton.autotune(
#     configs=[
#         triton.Config({'SPLIT_SIZE': 64}),
#         triton.Config({'SPLIT_SIZE': 128}),
#         triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 512}),
#         triton.Config({'SPLIT_SIZE': 1024}),
#         triton.Config({'SPLIT_SIZE': 2048}),
#         triton.Config({'SPLIT_SIZE': 4096}),
#         triton.Config({'SPLIT_SIZE': 8192}),
#         triton.Config({'SPLIT_SIZE': 16384}),
#     ],
#     key=['SPLIT_SIZE'],
# )
@triton.jit
def dequant_8bit_kernel(
    a_ptr,
    c_ptr,
    quant_ptr,
    absmax_ptr,
    # bias_ptr,
    num_paired_elements,
    QUANT_BLOCK: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask)
    a = a.to(tl.uint8, bitcast=True)

    # bias = tl.load(bias_ptr)

    # apply conversion
    scaled_int8 = tl.load(quant_ptr + a, mask)

    abs_blocks_lim = (num_paired_elements // QUANT_BLOCK) * QUANT_BLOCK + num_paired_elements % QUANT_BLOCK
    abs_offsets = offsets // QUANT_BLOCK
    mask_blocked = offsets < abs_blocks_lim

    absmax = tl.load(absmax_ptr + abs_offsets, mask_blocked)
    # apply scales
    out_dq = scaled_int8 * absmax
    # out_dq = out_dq + bias

    offs = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offs < num_paired_elements
    tl.store(c_ptr + offs, out_dq, mask)


def dequant_int8_fp16(
    A_nf4: torch.Tensor,
    quant_state_code: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    quant_blocksize: int = 64,
):
    number_of_paired_elements = A_nf4.numel()
    # we assume that split_size > quant_blocksize

    SPLIT_SIZE = 256
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META["SPLIT_SIZE"]), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    dequant_8bit_kernel[grid](
        A_nf4, out, quant_state_code, absmax, number_of_paired_elements, quant_blocksize, SPLIT_SIZE
    )
    return out


def quantize_blockwise_with_code(A, blocksize, code):
    n = A.numel()
    blocks = (n + blocksize - 1) // blocksize
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)

    A_reshaped = A.reshape(-1)
    quantized = torch.zeros_like(A_reshaped, dtype=torch.uint8, device=A.device)

    for i in range(blocks):
        start = i * blocksize
        end = min(start + blocksize, n)
        block = A_reshaped[start:end]

        absmax[i] = block.abs().max()
        if absmax[i] == 0:
            continue

        block_normalized = block / absmax[i]
        block_normalized = torch.clamp(block_normalized, -1, 1)

        diff = torch.abs(block_normalized.unsqueeze(-1) - code)
        quantized[start:end] = torch.argmin(diff, dim=-1).to(torch.uint8)

    return quantized, absmax


@triton.jit
def quantize_blockwise_kernel(
    A_ptr,
    code_ptr,
    absmax_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CODE_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    thread_idx = tl.arange(0, BLOCK_SIZE)

    start_idx = block_idx * BLOCK_SIZE
    offsets = start_idx + thread_idx

    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    absmax = tl.max(tl.abs(A), axis=0)
    tl.store(absmax_ptr + block_idx, absmax)

    A_normalized = A / absmax
    A_normalized = tl.clamp(A_normalized, -1.0, 1.0)

    code = tl.load(code_ptr + tl.arange(0, CODE_SIZE))

    diff = tl.abs(A_normalized[:, None] - code[None, :])
    quantized = tl.argmin(diff, axis=1).to(tl.uint8)

    tl.store(out_ptr + offsets, quantized, mask=mask)


def quantize_blockwise_with_code_triton(A, blocksize, code):
    n = A.numel()
    blocks = (n + blocksize - 1) // blocksize # amount of blocks (with remainder)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    quantized = torch.empty_like(A, dtype=torch.uint8)

    # grid = lambda META: (triton.cdiv(number_of_paired_elements, SPLIT_SIZE), )
    grid = (blocks,)
    quantize_blockwise_kernel[grid](
        A_ptr=A,
        code_ptr=code,
        absmax_ptr=absmax,
        out_ptr=quantized,
        n_elements=n,
        BLOCK_SIZE=blocksize,
        CODE_SIZE=code.numel(),
    )

    return quantized, absmax


@register_kernel("bitsandbytes::quantize_blockwise", "xpu")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.float32, lambda: f"A must be float32 on xpu, got {A.dtype}")

    n = A.numel()
    blocks = -(n // -blocksize)

    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty_like(A, dtype=torch.uint8)

    # quant_ref(code, A, absmax, out, blocksize, n)
    out, absmax = quantize_blockwise_with_code_triton(A, blocksize, code)

    return out, absmax


@register_kernel("bitsandbytes::dequantize_blockwise", "xpu")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(dtype == torch.float32, lambda: f"dtype must be float32 on xpu, got {dtype}")

    out = torch.empty_like(A, dtype=dtype, device=A.device)
    dequant_int8_fp16(
        A,
        code,
        absmax,
        out,
        blocksize,
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
    device="xpu",
)


@register_kernel("bitsandbytes::quantize_4bit", "xpu")
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


# @triton.autotune(
#     configs=[
#         # triton.Config({'SPLIT_SIZE': 64}),
#         # triton.Config({'SPLIT_SIZE': 128}),
#         # triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 512}),
#         # triton.Config({'SPLIT_SIZE': 1024}),
#         # triton.Config({'SPLIT_SIZE': 2048}),
#         # triton.Config({'SPLIT_SIZE': 4096}),
#         # triton.Config({'SPLIT_SIZE': 8192}),
#         # triton.Config({'SPLIT_SIZE': 16384}),
#     ],
#     key=['SPLIT_SIZE'],
# )
@triton.jit
def dequant_4bit_kernel(
    a_ptr, c_ptr, quant_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr
):
    PAIRED_QUANT_BLOCK = QUANT_BLOCK // 2

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask)
    a = a.to(tl.uint8, bitcast=True)

    # higher 4bits from uint8 packed tensor
    higher = a & 0xF
    # lower 4bits
    lower = a >> 4

    # apply conversion
    higher_nf4 = tl.load(quant_ptr + higher)
    lower_nf4 = tl.load(quant_ptr + lower)

    abs_blocks_lim = (
        num_paired_elements // PAIRED_QUANT_BLOCK
    ) * PAIRED_QUANT_BLOCK + num_paired_elements % PAIRED_QUANT_BLOCK
    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    mask_blocked = offsets < abs_blocks_lim
    absmax = tl.load(absmax_ptr + abs_offsets, mask_blocked)

    # apply scales
    mul_high = higher_nf4 * absmax
    mul_low = lower_nf4 * absmax

    out_dq = tl.interleave(mul_low, mul_high)

    out_block_start = pid * SPLIT_SIZE * 2
    offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offs < num_paired_elements * 2
    tl.store(c_ptr + offs, out_dq, mask)


@register_kernel("bitsandbytes::dequantize_4bit", "xpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 on XPU, got {quant_type}")
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )
    torch._check(
        A.dtype == torch.uint8,
        lambda: f"Blockwise 4bit dequantization on XPU only supports uint8 storage, got {A.dtype}",
    )

    out = torch.empty(shape, dtype=dtype, device=A.device)

    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)
    return out


@register_kernel("bitsandbytes::dequantize_4bit.out", "xpu")
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
    # It's will be processed as an array, so
    # actual length is row * col
    # Elements are in uint8 format, so interleaved
    # so total amount of data is 2 * elem_count
    number_of_paired_elements = A.numel()
    # we assume that split_size > quant_blocksize

    SPLIT_SIZE = 512
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, SPLIT_SIZE), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    dequant_4bit_kernel[grid](A, out, _NF4_QUANT_TABLE, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)


@register_kernel("bitsandbytes::gemv_4bit", "xpu")
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

    B_dq = torch.empty(shapeB, dtype=A.dtype, device=A.device)
    _dequantize_4bit_impl(
        B,
        absmax,
        blocksize,
        "nf4",
        dtype=A.dtype,
        out=B_dq,
    )

    # User called gemv with B.t(), so we need to transpose it back.
    # if B.shape[0] == 1:
    #    B_dq = B_dq.t()

    return torch.nn.functional.linear(
        A,
        B_dq,
        bias=None,
    )
