import torch

import triton
import triton.language as tl


# Triton implementation of similar CUDA kernel to avoid loading code from csrc/kernels.cu::dQuantizeFP4
# @triton.autotune(
#     configs=[
#         triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 1}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 4}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 8}),
#     ],
#     key=["n_elements"],
# )
@triton.jit
def quantize_fp4_blockwise_kernel(
    A_ptr,
    absmax_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SPLIT_NUM_BLOCKS: tl.constexpr,
):
    PAIRED_SPLIT_NUM_BLOCKS: tl.constexpr = SPLIT_NUM_BLOCKS * 2
    block_start_idx = tl.program_id(0) * PAIRED_SPLIT_NUM_BLOCKS
    thread_idx = tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS * BLOCK_SIZE)

    offsets = block_start_idx * BLOCK_SIZE + thread_idx
    mask = offsets < n_elements

    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # To be able process several blocks -> (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE)
    A_reshaped = tl.reshape(A, (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE))

    # Calculating absamax for each block
    absmax = tl.max(tl.abs(A_reshaped), axis=1)
    tl.store(absmax_ptr + block_start_idx + tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS), absmax)

    A_normalized = A_reshaped / absmax[:, None]
    A_normalized = tl.clamp(A_normalized, -1.0, 1.0)

    sign = tl.where(A_normalized < 0, 0b1000, 0b0000)
    A_absf = tl.abs(A_normalized)

    result = tl.where(
        A_absf > 0.29166667,
        tl.where(
            A_absf > 0.583333, tl.where(A_absf > 0.8333333, 0b011, 0b010), tl.where(A_absf > 0.4166667, 0b101, 0b100)
        ),
        tl.where(
            A_absf > 0.0859375,
            tl.where(A_absf > 0.20833333, 0b0111, 0b0110),
            tl.where(A_absf > 0.00260417, 0b0001, 0b0000),
        ),
    )
    quantized = (result ^ sign).to(tl.uint8)

    quantized = quantized.reshape((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE // 2, 2))
    left, right = quantized.split()
    packed = left << 4 | (right & 0xF)

    packed_flat = tl.reshape(packed, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    out_offsets = block_start_idx * BLOCK_SIZE // 2 + tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)
    out_mask = out_offsets < n_elements // 2
    tl.store(out_ptr + out_offsets, packed_flat, mask=out_mask)


# Triton implementation of similar CUDA kernel to avoid loading code from csrc/kernels.cu::dQuantizeNF4
# @triton.autotune(
#     configs=[
#         triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 1}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 4}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 8}),
#     ],
#     key=["n_elements"],
# )
@triton.jit
def quantize_nf4_blockwise_kernel(
    A_ptr,
    absmax_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SPLIT_NUM_BLOCKS: tl.constexpr,
):
    PAIRED_SPLIT_NUM_BLOCKS: tl.constexpr = SPLIT_NUM_BLOCKS * 2
    block_start_idx = tl.program_id(0) * PAIRED_SPLIT_NUM_BLOCKS
    thread_idx = tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS * BLOCK_SIZE)

    offsets = block_start_idx * BLOCK_SIZE + thread_idx
    mask = offsets < n_elements

    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # To be able process several blocks -> (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE)
    A_reshaped = tl.reshape(A, (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE))

    # Calculating absamax for each block
    absmax = tl.max(tl.abs(A_reshaped), axis=1)
    tl.store(absmax_ptr + block_start_idx + tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS), absmax)

    A_normalized = A_reshaped / absmax[:, None]
    A_normalized = tl.clamp(A_normalized, -1.0, 1.0)

    result = tl.where(
        A_normalized > 0.03979014977812767,
        tl.where(
            A_normalized > 0.3893125355243683,
            tl.where(
                A_normalized > 0.6427869200706482,
                tl.where(A_normalized > 0.8614784181118011, 0b1111, 0b1110),
                tl.where(A_normalized > 0.5016634166240692, 0b1101, 0b1100),
            ),
            tl.where(
                A_normalized > 0.2035212516784668,
                tl.where(A_normalized > 0.2920137718319893, 0b1011, 0b1010),
                tl.where(A_normalized > 0.1202552504837513, 0b1001, 0b1000),
            ),
        ),
        tl.where(
            A_normalized > -0.33967943489551544,
            tl.where(
                A_normalized > -0.13791173323988914,
                tl.where(A_normalized > -0.045525018125772476, 0b0111, 0b0110),
                tl.where(A_normalized > -0.23460740596055984, 0b0101, 0b0100),
            ),
            tl.where(
                A_normalized > -0.6106329262256622,
                tl.where(A_normalized > -0.4599952697753906, 0b0011, 0b0010),
                tl.where(A_normalized > -0.8480964004993439, 0b0001, 0b0000),
            ),
        ),
    )
    quantized = result.to(tl.uint8)

    quantized = quantized.reshape((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE // 2, 2))

    left, right = quantized.split()
    packed = left << 4 | (right & 0xF)

    packed_flat = tl.reshape(packed, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    out_offsets = block_start_idx * BLOCK_SIZE // 2 + tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)
    out_mask = out_offsets < n_elements // 2
    tl.store(out_ptr + out_offsets, packed_flat, mask=out_mask)


def quantize_4bit_blockwise_triton(A, blocksize, quant_type, blocks, absmax, num_elements, quantized_out):
    # grid = lambda META: (triton.cdiv(blocks, META["SPLIT_NUM_BLOCKS"]),)
    split_num_blocks = 4
    grid = (triton.cdiv(blocks, split_num_blocks),)
    if quant_type == "fp4":
        quantize_fp4_blockwise_kernel[grid](
            A_ptr=A,
            absmax_ptr=absmax,
            out_ptr=quantized_out,
            n_elements=num_elements,
            BLOCK_SIZE=blocksize,
            SPLIT_NUM_BLOCKS=split_num_blocks,
        )
    else:
        quantize_nf4_blockwise_kernel[grid](
            A_ptr=A,
            absmax_ptr=absmax,
            out_ptr=quantized_out,
            n_elements=num_elements,
            BLOCK_SIZE=blocksize,
            SPLIT_NUM_BLOCKS=split_num_blocks,
        )
    return quantized_out, absmax


@triton.jit
def dequant_4bit_body_util(a, offsets, quant_ptr, absmax_ptr, n_elems, QUANT_BLOCK: tl.constexpr):
    PAIRED_QUANT_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    mask = offsets < n_elems
    higher = a & 0xF
    # lower 4bits
    lower = a >> 4

    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    absmax = tl.load(absmax_ptr + abs_offsets, mask=mask, other=1.0, eviction_policy="evict_last")

    # apply conversion
    lower_4 = tl.load(quant_ptr + lower, eviction_policy="evict_last")
    higher_4 = tl.load(quant_ptr + higher, eviction_policy="evict_last")

    mul_high = higher_4 * absmax
    mul_low = lower_4 * absmax
    out_dq = tl.interleave(mul_low, mul_high)
    return out_dq


# Triton implementation of similar CUDA kernel to avoid loading code from csrc/kernels.cu::dDequantizeFP4Tree
@triton.jit
def dequantize_fp4_tree(val, absmax):
    # val: tl.tensor (uint8)
    # absmax: tl.tensor (float32/float16)
    #  00001100  00001011  00001001  00001111
    sign = tl.where((val & 0b1000) == 0b1000, -1.0, 1.0)  # -1
    third_bit = (val & 0b0100) == 0b0100  # True
    second_bit = (val & 0b0010) == 0b0010  # False
    first_bit = (val & 0b0001) == 0b0001  # False

    branch1 = tl.where(
        second_bit,
        tl.where(first_bit, 0.25, 0.16666667),  # 1111, 1110
        tl.where(first_bit, 0.5, 0.33333333),  # 1101, 1100
    )
    branch2 = tl.where(
        second_bit,
        tl.where(first_bit, 1.0, 0.66666667),  # 1011, 1010
        tl.where(first_bit, 0.00520833, 0.0),  # 1001, 1000
    )
    out = tl.where(third_bit, branch1, branch2)
    return out * sign * absmax


@triton.jit
def dequant_fp4_body_util(a, offsets, absmax_ptr, n_elems, QUANT_BLOCK: tl.constexpr):
    PAIRED_QUANT_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    mask = offsets < n_elems
    higher = a & 0xF
    lower = a >> 4

    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    absmax = tl.load(absmax_ptr + abs_offsets, mask=mask, other=1.0, eviction_policy="evict_last")
    mul_high = dequantize_fp4_tree(higher, absmax)
    mul_low = dequantize_fp4_tree(lower, absmax)
    out_dq = tl.interleave(mul_low, mul_high)
    return out_dq


# Triton implementation of similar CUDA kernel to avoid loading code from csrc/kernels.cu::dDequantizeNF4
@triton.jit
def dequantize_nf4_tree(val):
    # val: tl.tensor (uint8)
    cond0 = (val & 0b1000) == 0b1000
    cond1 = (val & 0b0100) == 0b0100
    cond2 = (val & 0b0010) == 0b0010
    cond3 = (val & 0b0001) == 0b0001

    # Positive branch (val & 0b1000) == 8
    branch_pos = tl.where(
        cond1,
        tl.where(
            cond2,
            tl.where(cond3, 1.0, 0.7229568362236023),  # 1111, 1110
            tl.where(cond3, 0.5626170039176941, 0.44070982933044434),  # 1101, 1100
        ),
        tl.where(
            cond2,
            tl.where(cond3, 0.33791524171829224, 0.24611230194568634),  # 1011, 1010
            tl.where(cond3, 0.16093020141124725, 0.07958029955625534),  # 1001, 1000
        ),
    )

    # Negative branch (val & 0b1000) == 0
    branch_neg = tl.where(
        cond1,
        tl.where(
            cond2,
            tl.where(cond3, 0.0, -0.09105003625154495),  # 0111, 0110
            tl.where(cond3, -0.18477343022823334, -0.28444138169288635),  # 0101, 0100
        ),
        tl.where(
            cond2,
            tl.where(cond3, -0.39491748809814453, -0.5250730514526367),  # 0011, 0010
            tl.where(cond3, -0.6961928009986877, -1.0),  # 0001, 0000
        ),
    )
    return tl.where(cond0, branch_pos, branch_neg)


@triton.jit
def dequant_nf4_body_util(a, offsets, absmax_ptr, n_elems, QUANT_BLOCK: tl.constexpr):
    PAIRED_QUANT_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    mask = offsets < n_elems
    higher = a & 0xF
    # lower 4bits
    lower = a >> 4

    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    absmax = tl.load(absmax_ptr + abs_offsets, mask=mask, other=1.0, eviction_policy="evict_last")
    mul_high = dequantize_nf4_tree(higher) * absmax
    mul_low = dequantize_nf4_tree(lower) * absmax
    out_dq = tl.interleave(mul_low, mul_high)
    return out_dq


# All such kernels are similar, so maybe code can be generalised.
# @triton.autotune(
#     configs=[
# #         # triton.Config({'SPLIT_SIZE': 64}),
# #         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         triton.Config({'SPLIT_SIZE': 128}),
#         triton.Config({'SPLIT_SIZE': 128}, num_warps = 32, num_stages = 2),
# #         # triton.Config({'SPLIT_SIZE': 128}, num_warps = 4, num_stages = 4),
# #         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 256}, num_warps = 32, num_stages = 2),
#         # triton.Config({'SPLIT_SIZE': 256}, num_warps = 4, num_stages = 4),
#         triton.Config({'SPLIT_SIZE': 512}),
#         triton.Config({'SPLIT_SIZE': 512}, num_warps = 32, num_stages = 2),
#         # triton.Config({'SPLIT_SIZE': 512}, num_warps = 4, num_stages = 4),
# #         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
# #         # triton.Config({'SPLIT_SIZE': 1024}),
# #         # # triton.Config({'SPLIT_SIZE': 2048}),
# #         # # triton.Config({'SPLIT_SIZE': 4096}),
# #         # # triton.Config({'SPLIT_SIZE': 8192}),
# #         # # triton.Config({'SPLIT_SIZE': 16384}),
#     ],
#     key=['num_paired_elements'],
# )
@triton.jit
def dequant_4bit_kernel(
    a_ptr, c_ptr, quant_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask, eviction_policy="evict_first")

    out_dq = dequant_4bit_body_util(
        a=a,
        offsets=offsets,
        quant_ptr=quant_ptr,
        absmax_ptr=absmax_ptr,
        n_elems=num_paired_elements,
        QUANT_BLOCK=QUANT_BLOCK,
    )

    out_block_start = pid * SPLIT_SIZE * 2
    offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offs < num_paired_elements * 2
    tl.store(c_ptr + offs, out_dq, mask)


# @triton.autotune(
#     configs=[
#         triton.Config({'SPLIT_SIZE': 128}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 256}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 512}),
#         triton.Config({'SPLIT_SIZE': 512}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 1024}, num_warps = 32, num_stages = 2),
#     ],
#     key=['num_paired_elements'],
# )
@triton.jit
def dequant_fp4_kernel(
    a_ptr, c_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask, eviction_policy="evict_first")

    out_dq = dequant_fp4_body_util(
        a=a,
        offsets=offsets,
        absmax_ptr=absmax_ptr,
        n_elems=num_paired_elements,
        QUANT_BLOCK=QUANT_BLOCK,
    )

    out_block_start = pid * SPLIT_SIZE * 2
    offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offs < num_paired_elements * 2
    tl.store(c_ptr + offs, out_dq, mask)


# @triton.autotune(
#     configs=[
#         triton.Config({'SPLIT_SIZE': 128}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 256}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 512}),
#         triton.Config({'SPLIT_SIZE': 512}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 1024}, num_warps = 32, num_stages = 2),
#     ],
#     key=['num_paired_elements'],
# )
@triton.jit
def dequant_nf4_kernel(
    a_ptr, c_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask, eviction_policy="evict_first")

    out_dq = dequant_nf4_body_util(
        a=a,
        offsets=offsets,
        absmax_ptr=absmax_ptr,
        n_elems=num_paired_elements,
        QUANT_BLOCK=QUANT_BLOCK,
    )

    out_block_start = pid * SPLIT_SIZE * 2
    offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offs < num_paired_elements * 2
    tl.store(c_ptr + offs, out_dq, mask)


def dequantize_4bit_impl(
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

    SPLIT_SIZE = 256
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META['SPLIT_SIZE']), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    if quant_type == "fp4":
        dequant_fp4_kernel[grid](A, out, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)
    else:
        dequant_nf4_kernel[grid](A, out, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)


def dequantize_4bit_impl_passing_code(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    code: torch.Tensor,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    number_of_paired_elements = A.numel()
    # we assume that split_size > quant_blocksize

    SPLIT_SIZE = 256
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META['SPLIT_SIZE']), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    dequant_4bit_kernel[grid](A, out, code, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)


######################### Fallback dequantization functions #########################
## for debug ##


# @triton.autotune(
#     configs=[
#         # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # #
#         # triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         #
#         triton.Config({"SPLIT_NUM_BLOCKS": 2}),
#         # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "large"}, num_stages=2, num_warps=32),
#         # # triton.Config({'SPLIT_NUM_BLOCKS': 2, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=2, num_warps=32),
#         # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         # triton.Config({"SPLIT_NUM_BLOCKS": 4, "grf_mode": "large"}, num_stages=2, num_warps=32),
#         # triton.Config({"SPLIT_NUM_BLOCKS": 4, "grf_mode": "large"}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_NUM_BLOCKS': 8, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#     ],
#     key=["n_elements", "BLOCK_SIZE"],
# )
@triton.jit
def quantize_4bit_blockwise_kernel(
    A_ptr,
    code_ptr,
    absmax_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CODE_SIZE: tl.constexpr,
    SPLIT_NUM_BLOCKS: tl.constexpr,
):
    PAIRED_SPLIT_NUM_BLOCKS: tl.constexpr = SPLIT_NUM_BLOCKS * 2
    block_start_idx = tl.program_id(0) * PAIRED_SPLIT_NUM_BLOCKS
    thread_idx = tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS * BLOCK_SIZE)

    offsets = block_start_idx * BLOCK_SIZE + thread_idx
    mask = offsets < n_elements

    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # To be able process several blocks -> (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE)
    A_reshaped = tl.reshape(A, (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE))

    # Calculating absamax for each block
    absmax = tl.max(tl.abs(A_reshaped), axis=1)
    tl.store(absmax_ptr + block_start_idx + tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS), absmax)

    A_normalized = A_reshaped / absmax[:, None]
    A_normalized = tl.clamp(A_normalized, -1.0, 1.0)

    lower_pivot = tl.zeros((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE), dtype=tl.int32)
    upper_pivot = tl.full((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE), CODE_SIZE - 1, dtype=tl.int32)

    for _ in range(4):  # ceil(log2(code_size)) = 4, actually, in general case should be input parameter
        pivot = (lower_pivot + upper_pivot) // 2
        val = tl.load(code_ptr + pivot)
        is_higher = A_normalized > val  # code[pivot]
        lower_pivot = tl.where(is_higher, pivot, lower_pivot)
        upper_pivot = tl.where(is_higher, upper_pivot, pivot)

    # Choose closest level
    lower_val = tl.load(code_ptr + lower_pivot)
    upper_val = tl.load(code_ptr + upper_pivot)
    lower_dist = tl.abs(A_normalized - lower_val)
    upper_dist = tl.abs(A_normalized - upper_val)
    quantized = tl.where(lower_dist <= upper_dist, lower_pivot, upper_pivot).to(tl.uint8)

    quantized = quantized.reshape((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE // 2, 2))
    quantized = quantized.to(tl.uint8, bitcast=True)
    left, right = quantized.split()
    packed = left << 4 | (right & 0xF)

    # Reduce don't guarantee the order of the elements passed to unite_2_int4
    # packed = tl.reduce(quantized, axis=2, combine_fn=unite_2_int4)
    # packed = packed.to(tl.uint8, bitcast=True)

    packed_flat = tl.reshape(packed, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    out_offsets = block_start_idx * BLOCK_SIZE // 2 + tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)
    out_mask = out_offsets < n_elements // 2
    tl.store(out_ptr + out_offsets, packed_flat, mask=out_mask)
