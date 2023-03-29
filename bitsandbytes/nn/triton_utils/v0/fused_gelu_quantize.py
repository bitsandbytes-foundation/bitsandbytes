import math
import torch
import time
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

tl.libdevice

# TODO: autotune this better.
@triton.autotune(
        configs=[
            triton.Config({}, num_stages=1, num_warps=8),
            triton.Config({}, num_stages=2, num_warps=8),
            triton.Config({}, num_stages=4, num_warps=8),
            triton.Config({}, num_stages=8, num_warps=8),
            triton.Config({}, num_stages=1),
            triton.Config({}, num_stages=2),
            triton.Config({}, num_stages=4),
            triton.Config({}, num_stages=8),
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=['n_elements']
)
@triton.jit
def _quantize_rowwise_nogroup_gelu(
    x_ptr,
    output_ptr,
    output_maxs,
    output_fp16,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)

    cdf = 0.5 * (1.0 + tl.libdevice.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
    x_new = x * cdf
    
    tl.store(output_fp16 + offsets, x_new, mask=row_mask)

    abs_x = tl.abs(x_new)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    output = tl.libdevice.llrint(127. * (x_new / max_val))
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_maxs + pid, max_val)

def quantize_rowwise_nogroup_gelu(x: torch.Tensor):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_fp16 = torch.empty(*x.shape, device=x.device, dtype=torch.float16)
    output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    _quantize_rowwise_nogroup_gelu[grid](x, output, output_maxs, output_fp16, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output, output_maxs, output_fp16



# TODO: autotune this better.
@triton.autotune(
        configs=[
            triton.Config({}, num_stages=1, num_warps=8),
            triton.Config({}, num_stages=2, num_warps=8),
            triton.Config({}, num_stages=4, num_warps=8),
            triton.Config({}, num_stages=8, num_warps=8),
            triton.Config({}, num_stages=1),
            triton.Config({}, num_stages=2),
            triton.Config({}, num_stages=4),
            triton.Config({}, num_stages=8),
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=['n_elements']
)
@triton.jit
def _quantize_rowwise_nogroup_back_gelu(
    x_ptr,
    in_ptr,
    output_ptr,
    output_maxs,
    output_fp16,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x_out = tl.load(x_ptr + offsets, mask=row_mask)
    x_in = tl.load(in_ptr + offsets, mask=row_mask)

    cdf = 0.5 * (1.0 + tl.libdevice.tanh(x_in * 0.7978845608 * (1 + 0.044715 * x_in * x_in)))
    intermediate = tl.libdevice.tanh(x_in * 0.7978845608 * (1 + 0.044715 * x_in * x_in))
    dcdf = 0.5 * (0.7978845608 + 0.1070322243 * x_in * x_in) * (1 - intermediate * intermediate)
    x = x_out * (cdf + x_in * dcdf)
    
    tl.store(output_fp16 + offsets, x, mask=row_mask)

    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    output = tl.libdevice.llrint(127. * (x / max_val))
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_maxs + pid, max_val)

def quantize_rowwise_nogroup_back_gelu(x: torch.Tensor, y : torch.Tensor):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_fp16 = torch.empty(*x.shape, device=x.device, dtype=torch.float16)
    output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    _quantize_rowwise_nogroup_back_gelu[grid](x, y, output, output_maxs, output_fp16, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output, output_maxs, output_fp16



# if __name__ == '__main__':
#     torch.manual_seed(0)

#     x = torch.randn(1280, 768).cuda().to(torch.float16)
#     out = quantize_rowwise_nogroup(x)

#     x_real = (127 * x.float() / x.abs().max(dim=1, keepdim=True)[0]).round().to(torch.int8)
#     max2 = x.abs().max(1)[0]

#     print(torch.allclose(out[1], max2))
#     print( (x_real == out[0]).float().mean() )

#     # for i in range(x.shape[0]):
#     #     print( (x_real[i, :] == out[0][i, :]).float().mean() )

#     # print(out[0])
#     # print(x_real)
#     # import pdb; pdb.set_trace()
#     # print(out[2])
#     # print(out[2][:10])
#     sums = x.sum(dim=0)
#     #print(sums[:10])
#     #print( (sums == out[2]).float().mean() )

#     import pdb; pdb.set_trace()
#     # import pdb; pdb.set_trace()
#     # exit()

#     # repeat = 16

#     # for _ in range(8):
#     #     out = quantize_rowwise_nogroup(x)

#     # triton_graph = torch.cuda.CUDAGraph()
#     # with torch.cuda.graph(triton_graph):
#     #     out = quantize_rowwise_nogroup(x)

#     # triton_graph.replay()

#     # torch.cuda.synchronize()
#     # start = time.time()
#     # for _ in range(repeat):
#     #     triton_graph.replay()
#     # torch.cuda.synchronize()
#     # end = time.time()

#     # print(out[0])
#     # print(out[1])
#     # print(x / x.abs().max(dim=1, keepdim=True)[0])
#     # max1 = out[1]
#     # max2 = x.abs().max(1)[0]
#     # print(max1, max2)
#     # print(torch.allclose(max1, max2))

#     #print(f"time: {(end - start) / repeat * 1000:.3f} ms")
