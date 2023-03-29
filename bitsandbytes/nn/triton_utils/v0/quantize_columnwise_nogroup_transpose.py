import math
import torch
import time
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

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
def _quantize_columnwise_nogroup_transpose(
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

def quantize_columnwise_nogroup_transpose(x: torch.Tensor):
    M, N = x.shape
    output = torch.empty(N, M, device=x.device, dtype=torch.int8)
    output_maxs = torch.empty(x.shape[1], device=x.device, dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(M))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _quantize_columnwise_nogroup_transpose[grid](x, output, output_maxs, n_elements, M, N, BLOCK_SIZE=M, P2=P2)
    return output, output_maxs



if __name__ == '__main__':
    torch.manual_seed(0)

    x = torch.randn(1280, 768).cuda().to(torch.float16)
    out = quantize_columnwise_nogroup_transpose(x)


    x_real = x.t().float()
    x_real_int8 = (127. * x_real / x_real.abs().max(dim=1, keepdim=True)[0]).round().to(torch.int8)
    maxs = x_real.abs().max(dim=1, keepdim=True)[0].half()

    #print(out[0][2,:])

    print((out[0] == x_real_int8).float().mean())
    print((out[1] == maxs[:, 0]).float().mean())

    # print(out[0])
    # print(out[1])

    # print(out[0][2,:])
    # print(x_real[2, :])

    # print((out[0] != x_real).nonzero())

    #import pdb; pdb.set_trace()
    # repeat = 16

    # for _ in range(8):
    #     out = quantize_columnwise_nogroup_transpose(x)

    # triton_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(triton_graph):
    #     out = quantize_columnwise_nogroup_transpose(x)

    # triton_graph.replay()

    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     triton_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()

    # print(out[0])
    # print(out[1])
    # print(x / x.abs().max(dim=0, keepdim=True)[0])
    # x_real = (127 * (x / x.abs().max(dim=0, keepdim=True)[0])).round().to(torch.int8)
    # max1 = out[1]
    # max2 = x.abs().max(0)[0]
    # print(max1, max2)
    # import pdb; pdb.set_trace()
    # print(torch.allclose(max1, max2))

    # print(f"time: {(end - start) / repeat * 1000:.3f} ms")
