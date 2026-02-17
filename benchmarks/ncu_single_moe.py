"""Single MoE kernel invocation for detailed NCU profiling."""
import torch, sys
sys.path.insert(0, ".")
import bitsandbytes
from bitsandbytes.functional import quantize_kbit, create_normal_float_codebook

torch.manual_seed(42)
k, K_dim, N, num_experts, M = 4, 2048, 512, 8, 512
codebook = create_normal_float_codebook(k, device="cuda")

packed_list, absmax_list = [], []
for e in range(num_experts):
    W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
    packed, absmax, _ = quantize_kbit(W, k, codebook=codebook, absmax_format="fp32")
    packed_list.append(packed)
    absmax_list.append(absmax)

B_packed_list, B_absmax_list = [], []
for e in range(num_experts):
    bp, ba = torch.ops.bitsandbytes.repack_kbit(packed_list[e], absmax_list[e], K_dim, N, k)
    B_packed_list.append(bp)
    B_absmax_list.append(ba)

B_packed_all = torch.cat(B_packed_list)
B_absmax_all = torch.cat(B_absmax_list)

A_list, offsets = [], [0]
for e in range(num_experts):
    A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
    A_list.append(A)
    offsets.append(offsets[-1] + M)
A_concat = torch.cat(A_list, dim=0)
eo = torch.tensor(offsets, dtype=torch.int32, device="cuda")

# Warmup
for _ in range(3):
    C = torch.ops.bitsandbytes.kbit_grouped_gemm(
        A_concat, B_packed_all, B_absmax_all, codebook, eo, K_dim, N, k, num_experts, M)
torch.cuda.synchronize()

# Profiled call
C = torch.ops.bitsandbytes.kbit_grouped_gemm(
    A_concat, B_packed_all, B_absmax_all, codebook, eo, K_dim, N, k, num_experts, M)
torch.cuda.synchronize()
