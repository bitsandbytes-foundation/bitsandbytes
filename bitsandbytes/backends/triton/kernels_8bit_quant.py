import torch

import triton
import triton.language as tl


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
    out_ptr,
    code_ptr,
    absmax_ptr,
    n,
    QUANT_BLOCK: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < n
    out_dq = dequant_8bit_blockwise_kernel_util(a_ptr, offsets, code_ptr, absmax_ptr, mask, QUANT_BLOCK)
    tl.store(out_ptr + offsets, out_dq, mask)


def dequant_8bit_blockwise(
    a: torch.Tensor,
    absmax: torch.Tensor,
    quant_state_code: torch.Tensor,
    quant_blocksize: int = 64,
    dtype: torch.dtype = None,
    out: torch.Tensor = None,
):
    n = a.numel()
    if out is None:
        if dtype is None:
            raise ValueError("If out is None, dtype must be specified")
        out = torch.empty_like(a, dtype=dtype, device=a.device)

    SPLIT_SIZE = 256
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META["SPLIT_SIZE"]),)
    grid = (triton.cdiv(n, SPLIT_SIZE),)
    dequant_8bit_kernel[grid](
        a,
        out,
        quant_state_code,
        absmax,
        n,
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
def quantize_8bit_blockwise_kernel(
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

    quantized, absmax = quantize_8bit_blockwise_kernel_util(A, code_ptr, CODE_SIZE, BLOCK_SIZE, SPLIT_NUM_BLOCKS)
    tl.store(absmax_ptr + block_start_idx + tl.arange(0, SPLIT_NUM_BLOCKS), absmax)
    tl.store(out_ptr + offsets, quantized, mask=mask)


def quantize_blockwise_triton(A, code, blocksize, absmax=None, out=None):
    n = A.numel()
    blocks = -(n // -blocksize)

    if absmax is None:
        absmax = torch.empty((blocks,), device=A.device, dtype=A.dtype)
    if out is None:
        out = torch.empty_like(A.flatten(), dtype=torch.uint8)

    split_num_blocks = 1
    grid = (triton.cdiv(blocks, split_num_blocks),)
    # grid = lambda META: (triton.cdiv(blocks, META["SPLIT_NUM_BLOCKS"]),)
    quantize_8bit_blockwise_kernel[grid](
        A_ptr=A,
        code_ptr=code,
        absmax_ptr=absmax,
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=blocksize,
        CODE_SIZE=code.numel(),
        SPLIT_NUM_BLOCKS=split_num_blocks,
        # num_warps=1,
        # num_stages=2,
    )
    out = out.reshape(A.shape)

    return out, absmax


@triton.jit
def quantize_8bit_blockwise_kernel_util(
    a,
    code_ptr,
    CODE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_PER_TH: tl.constexpr,
):
    # To be able process several blocks -> (BLOCK_SIZE, SPLIT_NUM_BLOCKS)
    a_reshaped = tl.reshape(a, (N_PER_TH, BLOCK_SIZE))

    # Calculating absmax for each block
    absmax = tl.max(tl.abs(a_reshaped), axis=1)

    a_normalized = a_reshaped / absmax[:, None]
    a_normalized = tl.clamp(a_normalized, -1.0, 1.0)

    lower_pivot = tl.zeros((N_PER_TH, BLOCK_SIZE), dtype=tl.int32)
    upper_pivot = tl.full((N_PER_TH, BLOCK_SIZE), CODE_SIZE - 1, dtype=tl.int32)

    # ceil(log2(code_size)) = 8, actually, in general case should be input parameter
    for _ in range(8):
        pivot = (lower_pivot + upper_pivot) // 2
        val = tl.load(code_ptr + pivot)
        is_higher = a_normalized > val  # code[pivot]
        lower_pivot = tl.where(is_higher, pivot, lower_pivot)
        upper_pivot = tl.where(is_higher, upper_pivot, pivot)

    # Choose closest level
    lower_val = tl.load(code_ptr + lower_pivot)
    upper_val = tl.load(code_ptr + upper_pivot)
    lower_dist = tl.abs(a_normalized - lower_val)
    upper_dist = tl.abs(a_normalized - upper_val)
    quantized = tl.where(lower_dist <= upper_dist, lower_pivot, upper_pivot).to(tl.uint8)

    # too slow approach
    # diff = tl.abs(A_normalized[:, :, None] - code[None, None, :])
    # quantized = tl.argmin(diff, axis=2).to(tl.uint8)

    quantized_flat = tl.reshape(quantized, (BLOCK_SIZE * N_PER_TH,))
    return quantized_flat, absmax


@triton.jit
def dequant_8bit_blockwise_kernel_util(
    a_ptr,
    offsets,
    code_ptr,
    absmax_ptr,
    mask,
    BLOCK_SIZE: tl.constexpr,
):
    a = tl.load(a_ptr + offsets, mask, other=0).to(tl.uint8)
    scaled_int8 = tl.load(code_ptr + a, mask)
    # Load scales
    absmax_offsets = offsets // BLOCK_SIZE
    absmax = tl.load(absmax_ptr + absmax_offsets, mask=mask, other=0.0, eviction_policy="evict_last")
    # Apply scales
    out_dq = scaled_int8 * absmax
    return out_dq
