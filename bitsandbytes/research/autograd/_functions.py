from functools import reduce  # Required in Python 3
import operator
from typing import Optional
import warnings

import torch

from bitsandbytes.autograd._functions import GlobalOutlierPooler, MatmulLtState
import bitsandbytes.functional as F


# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class MatMulFP8Mixed(torch.autograd.Function):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, fw_code=None, bw_code=None, bsz=1024, bsz2=1024):
        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B

            B_shape = B.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

        # 1. Dequantize
        # 2. MatmulnN
        cA, state = F.quantize_blockwise(A, code=fw_code, blocksize=bsz)
        fp8A = F.dequantize_blockwise(cA, state, blocksize=bsz).to(A.dtype)

        cB, state = F.quantize(B.float(), code=fw_code)
        fp8B = F.dequantize(cB, state).to(B.dtype)

        output = torch.matmul(fp8A, fp8B)

        # output is half

        # 3. Save state
        ctx.fw_code = fw_code
        ctx.bw_code = bw_code
        ctx.bsz = bsz
        ctx.bsz2 = bsz2
        ctx.dtype_A, ctx.dtype_B = A.dtype, B.dtype

        if any(ctx.needs_input_grad[:2]):
            # NOTE: we send back A, and re-quant.
            ctx.tensors = (A, fp8B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, None, None, None, None

        req_gradA, req_gradB, _, _, _, _, _ = ctx.needs_input_grad
        A, B = ctx.tensors

        grad_A, grad_B = None, None

        # TODO: Fix blocksize to be output_dim
        cgrad_out, state = F.quantize_blockwise(grad_output, code=ctx.bw_code, blocksize=ctx.bsz2)
        fp8out = F.dequantize_blockwise(cgrad_out, state, blocksize=ctx.bsz2).to(grad_output.dtype)

        # cgrad_output_2, state_2 = F.quantize(grad_output.float(), code=ctx.bw_code)
        # fp8out_2 = F.dequantize(cgrad_output_2, state_2).to(grad_output.dtype)

        # grad_output_reshape = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        # fp8grad_transpose, stategrad_transpose = F.vectorwise_quant(grad_output_reshape, dim=0, quant_type='vector')
        # fp8out_transpose = (fp8grad_transpose / 7) * stategrad_transpose
        # fp8out_transpose = fp8out_transpose.view(grad_output.shape[0], grad_output.shape[1], grad_output.shape[2])

        # not supported by PyTorch. TODO: create work-around
        if req_gradA:
            grad_A = torch.matmul(fp8out, B.t().to(fp8out.dtype)).to(A.dtype)

        if req_gradB:
            if len(A.shape) == 3:
                At = A.transpose(2, 1).contiguous()
            else:
                At = A.transpose(1, 0).contiguous()
            # cA, state = F.quantize(At.float(), code=ctx.fw_code)
            # fp8At = F.dequantize(cA, state).to(A.dtype)
            grad_B = torch.matmul(At.to(grad_output.dtype), grad_output).to(B.dtype)

        return grad_A, grad_B, None, None, None, None, None


class MatMulFP8Global(torch.autograd.Function):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, fw_code=None, bw_code=None, bsz=1024, bsz2=1024):
        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B

            B_shape = B.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

        # 1. Dequantize
        # 2. MatmulnN
        cA, state = F.quantize(A.float(), code=fw_code)
        fp8A = F.dequantize(cA, state).to(A.dtype)

        cB, state = F.quantize(B.float(), code=fw_code)
        fp8B = F.dequantize(cB, state).to(B.dtype)

        output = torch.matmul(fp8A, fp8B)

        # output is half

        # 3. Save state
        ctx.fw_code = fw_code
        ctx.bw_code = bw_code
        ctx.bsz = bsz
        ctx.bsz2 = bsz2
        ctx.dtype_A, ctx.dtype_B = A.dtype, B.dtype

        if any(ctx.needs_input_grad[:2]):
            # NOTE: we send back A, and re-quant.
            ctx.tensors = (A, fp8B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, None, None, None, None

        req_gradA, req_gradB, _, _, _, _, _ = ctx.needs_input_grad
        A, B = ctx.tensors

        grad_A, grad_B = None, None

        # TODO: Fix blocksize to be output_dim
        cgrad_out, state = F.quantize(grad_output.float(), code=ctx.bw_code)
        fp8out = F.dequantize(cgrad_out, state).to(grad_output.dtype)

        # cgrad_output_2, state_2 = F.quantize(grad_output.float(), code=ctx.bw_code)
        # fp8out_2 = F.dequantize(cgrad_output_2, state_2).to(grad_output.dtype)

        # grad_output_reshape = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        # fp8grad_transpose, stategrad_transpose = F.vectorwise_quant(grad_output_reshape, dim=0, quant_type='vector')
        # fp8out_transpose = (fp8grad_transpose / 7) * stategrad_transpose
        # fp8out_transpose = fp8out_transpose.view(grad_output.shape[0], grad_output.shape[1], grad_output.shape[2])

        # not supported by PyTorch. TODO: create work-around
        if req_gradA:
            grad_A = torch.matmul(fp8out, B.t().to(fp8out.dtype)).to(A.dtype)

        if req_gradB:
            if len(A.shape) == 3:
                At = A.transpose(2, 1).contiguous()
            else:
                At = A.transpose(1, 0).contiguous()
            cA, state = F.quantize(At.float(), code=ctx.fw_code)
            fp8At = F.dequantize(cA, state).to(A.dtype)
            grad_B = torch.matmul(fp8At.to(fp8out.dtype), fp8out).to(B.dtype)

        return grad_A, grad_B, None, None, None, None, None


class SwitchBackBnb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, state: Optional[MatmulLtState] = None):
        state = state or MatmulLtState()

        # default to pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            if A.shape[-1] == B.shape[0]:
                return torch.empty(A.shape[:-1] + B.shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B.shape[:1], dtype=A.dtype, device=A.device)

        # 1. Quantize A
        # 2. Quantize B
        # 3. Matmul
        # 4. Mixed-precision decomposition matmul
        # 5. Save state
        input_shape = A.shape
        if state.outlier_pool is None:
            state.outlier_pool = GlobalOutlierPooler.get_instance()

        # Cast A to fp16
        if A.dtype != torch.float16:
            warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")

        # 1. Quantize A
        if len(A.shape) == 3:
            A = A.view(-1, A.shape[-1]).contiguous()
        CA, CAt, SCA, SCAt, outlier_cols = F.int8_double_quant(A.to(torch.float16), threshold=state.threshold)

        if state.threshold > 0.0 and outlier_cols is not None:
            if state.has_fp16_weights:
                idx = outlier_cols
                CA[:, idx] = 0
                subA = A[:, idx]
                state.subB = B[:, idx].t().contiguous()
                state.idx = idx
            else:
                if state.SB is None:
                    state.SB = (state.CB.shape, "row")
        else:
            if not state.has_fp16_weights and state.SB is None:
                state.SB = (state.CB.shape, "row")
            subA = None

        # 2. Quantize B
        if state.has_fp16_weights:
            # print('B shape', B.shape)
            has_grad = True if (getattr(B, "grad", None) is not None) else False
            is_transposed = not B.is_contiguous() and B.shape[0] == B.stride(1)
            if is_transposed:
                B = B.contiguous()

            if (state.is_training and not has_grad) or state.SB is None:
                state.reset_grads()
                (
                    state.CB,
                    state.CBt,
                    state.SCB,
                    state.SCBt,
                    _,
                ) = F.int8_double_quant(B.to(torch.float16))
                state.SB = (state.CB.shape, "row")
        else:
            has_grad = False

        if outlier_cols is not None and not state.has_fp16_weights:
            # extract outliers
            state.idx = outlier_cols
            outliers = state.CB[:, state.idx.long()].clone()
            state.subB = (outliers * state.SCB.view(-1, 1) / 127.0).t().contiguous().to(A.dtype)
            CA[:, state.idx.long()] = 0

            subA = A[:, state.idx.long()]

        shapeB = state.SB[0]

        if len(input_shape) == 3:
            output_shape = (input_shape[0], input_shape[1], shapeB[0])
        else:
            output_shape = (input_shape[0], shapeB[0])

        # 3. Matmul
        out32 = F.int8_linear_matmul(CA, state.CB)
        # we apply the fused bias here

        if bias is None or bias.dtype == torch.float16:
            output = F.int8_mm_dequant(out32, SCA, state.SCB, bias=bias).to(A.dtype)
        else:  # apply bias separately
            output = F.int8_mm_dequant(out32, SCA, state.SCB, bias=None).to(A.dtype)
            output.add_(bias)

        # 4. Mixed-precision decomposition matmul
        if outlier_cols is not None and subA is not None:
            output += torch.matmul(subA, state.subB)

        # 5. Save state
        ctx.state = state

        ctx.grad_shape = input_shape
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (CAt, subA, A)
            ctx.tensor_states = (SCAt, state.idx)
        else:
            ctx.tensors = [None, None, None]
            ctx.tensor_states = (None, None)
            ctx.save_for_backward(None, None)

        clone_func = torch.clone if len(output_shape) == 3 else lambda x: x
        return clone_func(output.view(output_shape))

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

        req_gradA, req_gradB, _, req_gradBias, _ = ctx.needs_input_grad
        CAt, subA, A = ctx.tensors
        SCAt, idx = ctx.tensor_states
        state = ctx.state
        grad_A = grad_B = grad_bias = None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # Cast grad_output to fp16
        if len(grad_output.shape) == 3:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        Cgrad, Cgradt, SCgrad, SCgradt, outlier_cols = F.int8_double_quant(grad_output.to(torch.float16))

        if req_gradB:
            # print('back A shape', A.shape)
            # print('grad output t shape', grad_output.t().shape)
            grad_B = torch.matmul(grad_output.t(), A)

        if req_gradA:
            if state.CB is not None:
                CB = state.CB.to(ctx.dtype_A, copy=True).mul_(state.SCB.unsqueeze(1).mul(1.0 / 127.0))
                grad_A = torch.matmul(grad_output, CB).view(ctx.grad_shape).to(ctx.dtype_A)
            else:
                raise Exception("State must contain either CBt or CB matrix for backward")

        return grad_A, grad_B, None, grad_bias, None


def get_block_sizes(input_matrix, weight_matrix):
    input_features = input_matrix.shape[-1]
    output_features = weight_matrix.shape[0] if weight_matrix.shape[1] == input_features else weight_matrix.shape[1]
    array = [4096, 2048, 1024, 512, 256, 128, 64, 0]
    bsz, bsz2 = 1024, 1024
    for i, k in enumerate(array):
        if input_features > array[i + 1]:
            bsz = k
            break
    for i, k in enumerate(array):
        if output_features > array[i + 1]:
            bsz2 = k
            break

    return bsz, bsz2


def matmul_fp8_global(
    A: torch.Tensor,
    B: torch.Tensor,
    fw_code: torch.Tensor,
    bw_code: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bsz: int = -1,
    bsz2: int = -1,
):
    if bsz == -1 or bsz2 == -1:
        bsz, bsz2 = get_block_sizes(A, B)
    return MatMulFP8Global.apply(A, B, out, fw_code, bw_code, bsz, bsz2)


def matmul_fp8_mixed(
    A: torch.Tensor,
    B: torch.Tensor,
    fw_code: torch.Tensor,
    bw_code: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bsz: int = -1,
    bsz2: int = -1,
):
    if bsz == -1 or bsz2 == -1:
        bsz, bsz2 = get_block_sizes(A, B)
    return MatMulFP8Mixed.apply(A, B, out, fw_code, bw_code, bsz, bsz2)


def switchback_bnb(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    state: Optional[MatmulLtState] = None,
    threshold=0.0,
    bias=None,
):
    state = state or MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold
    return SwitchBackBnb.apply(A, B, out, bias, state)
