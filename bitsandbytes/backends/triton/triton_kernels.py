import torch

import triton
import triton.language as tl

import warnings
if torch.xpu.is_available():
    from triton.language.extra.intel import libdevice
elif torch.cuda.is_available():
    from triton.language.extra.cuda import libdevice
else:
    warnings.warn("No supported device (XPU or CUDA) is available. optimizer32bit_triton and optimizer8bit_blockwise_triton will not be available!")


# @triton.autotune(
#     configs=[
#         # triton.Config({'SPLIT_SIZE': 64}),
#         # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 128}),
#         # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_SIZE": 256}),
#         # triton.Config({'SPLIT_SIZE': 256, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 256, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         triton.Config({"SPLIT_SIZE": 512}),
#         # triton.Config({'SPLIT_SIZE': 1024}),
#     ],
#     key=["num_paired_elements", "QUANT_BLOCK"],
# )
@triton.jit
def dequant_8bit_kernel(
    a_ptr,
    c_ptr,
    quant_ptr,
    absmax_ptr,
    num_paired_elements,
    QUANT_BLOCK: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask)
    a = a.to(tl.uint8)

    # apply conversion
    scaled_int8 = tl.load(quant_ptr + a, mask)

    abs_blocks_lim = (num_paired_elements // QUANT_BLOCK) * QUANT_BLOCK + num_paired_elements % QUANT_BLOCK
    abs_offsets = offsets // QUANT_BLOCK
    mask_blocked = offsets < abs_blocks_lim

    absmax = tl.load(absmax_ptr + abs_offsets, mask_blocked)
    # apply scales
    out_dq = scaled_int8 * absmax

    offs = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offs < num_paired_elements
    tl.store(c_ptr + offs, out_dq, mask)


def dequant_int8_blockwise(
    A_nf4: torch.Tensor,
    quant_state_code: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    quant_blocksize: int = 64,
):
    number_of_paired_elements = A_nf4.numel()

    SPLIT_SIZE = 256
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META["SPLIT_SIZE"]),)
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    dequant_8bit_kernel[grid](
        A_nf4,
        out,
        quant_state_code,
        absmax,
        number_of_paired_elements,
        quant_blocksize,
        SPLIT_SIZE,
    )
    return out


# @triton.autotune(
#     configs=[
#         triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 1}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2}),
#     ],
#     key=["n_elements"],
# )
@triton.jit
def quantize_blockwise_kernel(
    A_ptr,
    code_ptr,
    absmax_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CODE_SIZE: tl.constexpr,
    SPLIT_NUM_BLOCKS: tl.constexpr,
):
    block_start_idx = tl.program_id(0) * SPLIT_NUM_BLOCKS
    thread_idx = tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)

    offsets = block_start_idx * BLOCK_SIZE + thread_idx
    mask = offsets < n_elements

    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # To be able process several blocks -> (BLOCK_SIZE, SPLIT_NUM_BLOCKS)
    A_reshaped = tl.reshape(A, (SPLIT_NUM_BLOCKS, BLOCK_SIZE))

    # Calculating absamax for each block
    absmax = tl.max(tl.abs(A_reshaped), axis=1)
    tl.store(absmax_ptr + block_start_idx + tl.arange(0, SPLIT_NUM_BLOCKS), absmax)

    A_normalized = A_reshaped / absmax[:, None]
    A_normalized = tl.clamp(A_normalized, -1.0, 1.0)

    lower_pivot = tl.zeros((SPLIT_NUM_BLOCKS, BLOCK_SIZE), dtype=tl.int32)
    upper_pivot = tl.full((SPLIT_NUM_BLOCKS, BLOCK_SIZE), CODE_SIZE - 1, dtype=tl.int32)

    for _ in range(8):  # ceil(log2(code_size)) = 8, actually, in general case should be input parameter
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

    # too slow approach
    # diff = tl.abs(A_normalized[:, :, None] - code[None, None, :])
    # quantized = tl.argmin(diff, axis=2).to(tl.uint8)

    quantized_flat = tl.reshape(quantized, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    tl.store(out_ptr + offsets, quantized_flat, mask=mask)


def quantize_blockwise_triton(A, blocksize, code, blocks, absmax, quantized_out):
    n = A.numel()

    split_num_blocks = 1
    grid = (triton.cdiv(blocks, split_num_blocks),)
    # grid = lambda META: (triton.cdiv(blocks, META["SPLIT_NUM_BLOCKS"]),)
    quantize_blockwise_kernel[grid](
        A_ptr=A,
        code_ptr=code,
        absmax_ptr=absmax,
        out_ptr=quantized_out,
        n_elements=n,
        BLOCK_SIZE=blocksize,
        CODE_SIZE=code.numel(),
        SPLIT_NUM_BLOCKS=split_num_blocks,
    )

    return quantized_out, absmax


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

    SPLIT_SIZE = 256
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META['SPLIT_SIZE']), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    if quant_type == "fp4":
        dequant_fp4_kernel[grid](A, out, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)
    else:
        dequant_nf4_kernel[grid](A, out, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)


def _dequantize_4bit_impl_passing_code(
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


@triton.jit
def PreconditionOptimizer32bit2State_kernel(
    g_ptr, state1_ptr, state2_ptr, unorm_ptr,
    beta1, beta2, eps, step, gnorm_scale, n_elems,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elems

    # Loading gradient values
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    g = g * gnorm_scale  # Apply gradient scaling

    # Loading optimizer states
    m = tl.load(state1_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(state2_ptr + offsets, mask=mask, other=0.0)

    # Updating first and second moment estimates
    m_new = m * beta1 + (1 - beta1) * g
    v_new = v * beta2 + (1 - beta2) * (g * g)

    # Calculating bias correction
    correction1 = 1.0 - libdevice.pow(beta1, step)
    correction2 = 1.0 - libdevice.pow(beta2, step)
    m_hat = m_new / correction1
    v_hat = v_new / correction2

    # Calculating update and its square
    update = m_hat / (libdevice.sqrt(v_hat) + eps)
    update_sq = update * update

    # Reducing the sum of squared updates within the block
    block_sum = tl.sum(update_sq, axis=0)
    tl.atomic_add(unorm_ptr, block_sum)  # Atomically add to global unorm


@triton.jit
def Optimizer32bit2State_kernel(
    g_ptr, p_ptr, state1_ptr, state2_ptr, unorm_ptr, 
    max_unorm, param_norm, beta1, beta2, beta3, alpha, 
    eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n_elems,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elems

    # Calculate the update scaling factor
    update_scale = 1.0
    if max_unorm > 0.0 and unorm_ptr is not None:
        unorm_val = libdevice.sqrt(tl.load(unorm_ptr))  # Calculate actual unorm
        clip_val = max_unorm * param_norm
        update_scale = tl.where(unorm_val > clip_val, clip_val / unorm_val, 1.0)

    # Loading data
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(state1_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(state2_ptr + offsets, mask=mask, other=0.0)
    g_scaled = g * gnorm_scale

    # Calculating bias correction
    correction1 = 1.0 - libdevice.pow(beta1, step)
    correction2 = libdevice.sqrt(1.0 - libdevice.pow(beta2, step))

    # Handling skip_zeros flag
    non_zero_mask = mask
    if skip_zeros:
        non_zero_mask = mask & (g != 0.0)

    # Updating first and second moment estimates
    m_new = tl.where(non_zero_mask,
                    m * beta1 + (1 - beta1) * g_scaled,
                    m)
    v_new = tl.where(non_zero_mask,
                    v * beta2 + (1 - beta2) * (g_scaled * g_scaled),
                    v)

    # Calculating update
    denom = libdevice.sqrt(v_new) + eps * correction2
    step_size = -lr * correction2 / correction1
    update = update_scale * step_size * (m_new / denom)

    # Applying update and weight decay
    p_new = tl.where(non_zero_mask, p + update, p)
    if weight_decay > 0.0:
        p_new = tl.where(non_zero_mask, p_new * (1.0 - lr * weight_decay), p_new)

    # Storing results (original states)
    tl.store(p_ptr + offsets, p_new, mask=mask)
    tl.store(state1_ptr + offsets, m_new, mask=mask)
    tl.store(state2_ptr + offsets, v_new, mask=mask)


@triton.jit
def OptimizerStatic8bit2StateBlockwise_kernel(
    p_ptr, g_ptr, state1_ptr, state2_ptr, absmax1_ptr, absmax2_ptr, qmap1_ptr, qmap2_ptr,
    beta1, beta2, eps, step, lr, weight_decay, gnorm_scale, skip_zeros,
    n_elements, code_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID and offsets for block-wise processing
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load gradient and parameter
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)

    # Apply gradient norm scaling
    g = g * gnorm_scale

    # Skip updates if gradient is zero and skip_zeros is True
    non_zero_mask = mask
    if skip_zeros:
        non_zero_mask = mask & (g != 0.0)

    # Load and dequantize state1 (m)
    state1 = tl.load(state1_ptr + offsets, mask=mask, other=0).to(tl.uint8)
    absmax1 = tl.load(absmax1_ptr + pid)
    m = tl.load(qmap1_ptr + state1, mask=mask, other=0.0) * absmax1

    # Load and dequantize state2 (v)
    state2 = tl.load(state2_ptr + offsets, mask=mask, other=0).to(tl.uint8)
    absmax2 = tl.load(absmax2_ptr + pid)
    v = tl.load(qmap2_ptr + state2, mask=mask, other=0.0) * absmax2

    # Adam update step
    m_new = tl.where(non_zero_mask, beta1 * m + (1 - beta1) * g, m)
    v_new = tl.where(non_zero_mask, beta2 * v + (1 - beta2) * (g * g), v)

    # Bias correction
    m_hat = m_new / (1 - libdevice.pow(beta1, step))
    v_hat = v_new / (1 - libdevice.pow(beta2, step))

    # Update parameter
    p_new = tl.where(non_zero_mask, p - lr * m_hat / (libdevice.sqrt(v_hat) + eps), p)

    # Apply weight decay
    if weight_decay > 0.0:
        p_new = tl.where(non_zero_mask, p_new * (1.0 - lr * weight_decay), p_new)

    # Quantize m_new
    absmax_m = tl.max(libdevice.abs(m_new), axis=0)
    m_normalized = m_new / absmax_m
    m_normalized = tl.clamp(m_normalized, -1.0, 1.0)
    lower_pivot = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    upper_pivot = tl.full((BLOCK_SIZE,), code_size - 1, dtype=tl.int32)
    for _ in range(8):  # ceil(log2(code_size)) = 8 for code_size=256
        pivot = (lower_pivot + upper_pivot) // 2
        val = tl.load(qmap1_ptr + pivot, mask=mask, other=0.0)
        is_higher = m_normalized > val
        lower_pivot = tl.where(is_higher, pivot, lower_pivot)
        upper_pivot = tl.where(is_higher, upper_pivot, pivot)
    lower_val = tl.load(qmap1_ptr + lower_pivot, mask=mask, other=0.0)
    upper_val = tl.load(qmap1_ptr + upper_pivot, mask=mask, other=0.0)
    lower_dist = libdevice.abs(m_normalized - lower_val)
    upper_dist = libdevice.abs(m_normalized - upper_val)
    quantized_m = tl.where(lower_dist <= upper_dist, lower_pivot, upper_pivot).to(tl.uint8)
    quantized_m = tl.where(non_zero_mask, quantized_m, state1)

    # Quantize v_new
    absmax_v = tl.max(libdevice.abs(v_new), axis=0)
    v_normalized = v_new / absmax_v
    v_normalized = tl.clamp(v_normalized, -1.0, 1.0)
    lower_pivot = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    upper_pivot = tl.full((BLOCK_SIZE,), code_size - 1, dtype=tl.int32)
    for _ in range(8):  # ceil(log2(code_size)) = 8 for code_size=256
        pivot = (lower_pivot + upper_pivot) // 2
        val = tl.load(qmap2_ptr + pivot, mask=mask, other=0.0)
        is_higher = v_normalized > val
        lower_pivot = tl.where(is_higher, pivot, lower_pivot)
        upper_pivot = tl.where(is_higher, upper_pivot, pivot)
    lower_val = tl.load(qmap2_ptr + lower_pivot, mask=mask, other=0.0)
    upper_val = tl.load(qmap2_ptr + upper_pivot, mask=mask, other=0.0)
    lower_dist = libdevice.abs(v_normalized - lower_val)
    upper_dist = libdevice.abs(v_normalized - upper_val)
    quantized_v = tl.where(lower_dist <= upper_dist, lower_pivot, upper_pivot).to(tl.uint8)
    quantized_v = tl.where(non_zero_mask, quantized_v, state2)

    # Store results
    tl.store(p_ptr + offsets, p_new, mask=mask)
    tl.store(state1_ptr + offsets, quantized_m, mask=mask)
    tl.store(state2_ptr + offsets, quantized_v, mask=mask)
    tl.store(absmax1_ptr + pid, absmax_m, mask=True)
    tl.store(absmax2_ptr + pid, absmax_v, mask=True)
