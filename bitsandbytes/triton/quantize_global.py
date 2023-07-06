import math
import torch
import time
from bitsandbytes.triton.triton_utils import is_triton_available

if not is_triton_available():
    def quantize_global_transpose(input): return None
    def quantize_global(x: torch.Tensor): return None
else:

    import triton
    import triton.language as tl
    from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

    # global quantize
    @triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE': 1024,}, num_warps=4),
                triton.Config({'BLOCK_SIZE': 2048,}, num_stages=1),

            ],
            key=['n_elements']
    )
    @triton.jit
    def _quantize_global(
        x_ptr,
        absmax_inv_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        absmax_inv = tl.load(absmax_inv_ptr)
        output = tl.libdevice.llrint(127. * (x * absmax_inv))
        tl.store(output_ptr + offsets, output, mask=mask)

    def quantize_global(x: torch.Tensor):
        absmax = x.abs().max().unsqueeze(0)
        absmax_inv = 1./ absmax
        output = torch.empty(*x.shape, device='cuda', dtype=torch.int8)
        assert x.is_cuda and output.is_cuda
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _quantize_global[grid](x, absmax_inv, output, n_elements)
        return output, absmax


    # global quantize and transpose
    @triton.autotune(
            configs=[
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=4),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=4),

                # ...
            ],
            key=['M', 'N']
    )
    @triton.jit
    def _quantize_global_transpose(A, absmax_inv_ptr, B, stride_am, stride_an, stride_bn, stride_bm, M, N, 
                          BLOCK_M : tl.constexpr, 
                          BLOCK_N : tl.constexpr, 
                          GROUP_M : tl.constexpr):
        pid = tl.program_id(0)
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
        
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        a = tl.load(A, mask=mask)
        absmax_inv = tl.load(absmax_inv_ptr)
        
        # rematerialize to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        B = B + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]

        output = tl.libdevice.llrint(127. * (a * absmax_inv))

        tl.store(B, output, mask=mask)

    def quantize_global_transpose(input):
        absmax = input.abs().max().unsqueeze(0)
        absmax_inv = 1./ absmax
        M, N = input.shape
        out = torch.empty(N, M, device='cuda', dtype=torch.int8)
        
        assert out.size(0) == N and out.size(1) == M
        assert input.stride(0) == 1 or input.stride(1) == 1
        assert out.stride(0) == 1 or out.stride(1) == 1
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        _quantize_global_transpose[grid](input, absmax_inv, out, input.stride(0), input.stride(1), out.stride(0), out.stride(1), M, N)
        return out, absmax

