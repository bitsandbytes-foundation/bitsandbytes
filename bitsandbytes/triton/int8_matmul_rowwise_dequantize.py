import torch

from bitsandbytes.triton.triton_utils import is_triton_available

if not is_triton_available():

    def int8_matmul_rowwise_dequantize(a, b, state_x, state_w, bias):
        return None
else:
    import triton
    import triton.language as tl
    from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

    # This is a matmul kernel based on triton.ops.matmul
    # It is modified to support rowwise quantized input and columnwise quantized weight
    # It's purpose is fused matmul then dequantize
    # It does support bias.

    def init_to_zero(name):
        return lambda nargs: nargs[name].zero_()

    def get_configs_io_bound():
        configs = []
        for num_stages in [2, 3, 4, 5, 6]:
            for block_m in [16, 32]:
                for block_k in [32, 64]:
                    for block_n in [32, 64, 128, 256]:
                        num_warps = 2 if block_n <= 64 else 4
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": 1},
                                num_stages=num_stages,
                                num_warps=num_warps,
                            ),
                        )
                        # split_k
                        for split_k in [2, 4, 8, 16]:
                            configs.append(
                                triton.Config(
                                    {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": split_k},
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                    pre_hook=init_to_zero("C"),
                                ),
                            )
        return configs

    @triton.autotune(
        configs=[
            # basic configs for compute-bound matmuls
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=5, num_warps=2),
            # good for int8
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=5, num_warps=2),
            *get_configs_io_bound(),
        ],
        key=["M", "N", "K"],
        prune_configs_by={"early_config_prune": early_config_prune, "perf_model": estimate_matmul_time, "top_k": 10},
    )
    @triton.heuristics(
        {
            "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
        },
    )
    @triton.jit
    def _int8_matmul_rowwise_dequantize(
        A,
        B,
        C,
        bias,
        state_x_ptr,
        state_w_ptr,
        M,
        N,
        K,
        divfactor,
        has_bias: tl.constexpr,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        ACC_TYPE: tl.constexpr,
    ):
        # matrix multiplication
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)
        # do matrix multiplication
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
        # pointers
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        w_factor = tl.load(state_w_ptr + rbn)[None, :]
        x_factor = tl.load(state_x_ptr + ram)[:, None]

        # acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                k_remaining = K - k * (BLOCK_K * SPLIT_K)
                a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
                b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
            acc += tl.dot(a, b)
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk

        acc = w_factor * (x_factor * (acc * divfactor))
        acc = acc.to(C.dtype.element_ty)

        if has_bias:
            bias = tl.load(bias + rn).to(C.dtype.element_ty)
            acc = acc + bias[None, :]

        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        # handles write-back with reduction-splitting
        if SPLIT_K == 1:
            tl.store(C, acc, mask=mask)
        else:
            tl.atomic_add(C, acc, mask=mask)

    def int8_matmul_rowwise_dequantize(a, b, state_x, state_w, bias):
        divfactor = 1.0 / (127.0 * 127.0)

        has_bias = 0 if bias is None else 1

        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        c = torch.empty((M, N), device=device, dtype=torch.float16)
        # accumulator types
        ACC_TYPE = tl.float32  # if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # launch int8_matmul_rowwise_dequantize kernel
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), META["SPLIT_K"])
        _int8_matmul_rowwise_dequantize[grid](
            a,
            b,
            c,
            bias,
            state_x,
            state_w,
            M,
            N,
            K,
            divfactor,
            has_bias,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            GROUP_M=8,
            ACC_TYPE=ACC_TYPE,
        )
        return c
