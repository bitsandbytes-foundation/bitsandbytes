import math
import torch
import time
from bitsandbytes.triton.triton_utils import is_triton_available

if not is_triton_available():
    def quantize_columnwise_and_transpose(x: torch.Tensor): return None
else:

    import triton
    import triton.language as tl
    from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

    # This kernel does fused columnwise quantization and transpose.

    # TODO: autotune this better.
    @triton.autotune(
            configs=[
                triton.Config({}, num_stages=1),
                triton.Config({}, num_stages=2),
                triton.Config({}, num_stages=4),
                triton.Config({}, num_stages=8),
                triton.Config({}, num_stages=16),
                triton.Config({}, num_stages=1, num_warps=8),
                triton.Config({}, num_stages=2, num_warps=8),
                triton.Config({}, num_stages=4, num_warps=8),
                triton.Config({}, num_stages=8, num_warps=8),
                triton.Config({}, num_stages=16, num_warps=8),
                triton.Config({}, num_warps=1),
                triton.Config({}, num_warps=2),
                triton.Config({}, num_warps=4),
                triton.Config({}, num_warps=8),
            ],
            key=['n_elements']
    )
    @triton.jit
    def _quantize_columnwise_and_transpose(
        x_ptr,
        output_ptr,
        output_maxs,
        n_elements,
        M : tl.constexpr, N : tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        P2: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid
        p2_arange = tl.arange(0, P2)
        p2_arange_mask = p2_arange < M
        arange =  p2_arange * N
        offsets = block_start + arange
        x = tl.load(x_ptr + offsets, mask=p2_arange_mask)
        abs_x = tl.abs(x)
        max_val = tl.max(tl.where(p2_arange_mask, abs_x, 0), axis=0)
        output = tl.libdevice.llrint(127. * (x / max_val))

        new_start = pid * M 
        new_offsets = new_start + p2_arange
        tl.store(output_ptr + new_offsets, output, mask=p2_arange_mask)
        tl.store(output_maxs + pid, max_val)

    def quantize_columnwise_and_transpose(x: torch.Tensor):
        M, N = x.shape
        output = torch.empty(N, M, device=x.device, dtype=torch.int8)
        output_maxs = torch.empty(x.shape[1], device=x.device, dtype=torch.float16)

        P2 = int(2 ** (math.ceil(math.log2(M))))

        assert x.is_cuda and output.is_cuda
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _quantize_columnwise_and_transpose[grid](x, output, output_maxs, n_elements, M, N, BLOCK_SIZE=M, P2=P2)
        return output, output_maxs

