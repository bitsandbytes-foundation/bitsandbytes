from dataclasses import dataclass
from math import prod
from typing import Callable, Optional
import warnings
from warnings import warn

import torch
from typing_extensions import deprecated

import bitsandbytes.functional as F

# The inverse transformation for the colTuring and colAmpere format were contributed by Alex Borzunov:
# https://github.com/bigscience-workshop/petals/blob/main/src/petals/utils/linear8bitlt_patch.py


"""
    This class pools outlier dimensions across layers.
    This is particularly important for small models where outlier features
    are less systematic and occur with low frequency.
"""


class GlobalOutlierPooler:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.outliers = set()
        self.model_dim = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def add_outliers(self, outlier_idx, feature_dim):
        if self.model_dim is None:
            self.model_dim = feature_dim
        if feature_dim != self.model_dim:
            return  # we do not encode outliers for the 2nd FFN layer

        self.outliers.update(outlier_idx.tolist())

    def get_current_outlier_idx(self):
        return torch.Tensor(list(self.outliers)).to(torch.int64)


@deprecated(
    "This function is deprecated and will be removed in a future release.",
    category=FutureWarning,
)
def get_inverse_transform_indices(
    transform_tile: Callable[[torch.Tensor], torch.Tensor],
    tile_size: tuple[int, int],
):
    """
    Compute a permutation of indices that invert the specified (tiled) matrix transformation

    :param transform_tile: a function that applies forward transform to a tensor of shape [dim1, dim2]
    :param tile_size: higher-level tile dimensions, i.e. (8, 32) for Turing and (32, 32) for Ampere
    :note: we assume that tile_transform applies to a cpu-based int8 tensor of shape tile_size
    :example: transform_tile function for the turing layout (bitsandbytes.functional as F)
    :returns: indices
    """
    d1, d2 = tile_size
    assert 0 < d1 * d2 < 2**64
    tile_indices = torch.arange(d1 * d2, dtype=torch.int64).view(d1, d2)
    # encode each position in tile as a tuple of <= 8 unique bytes
    permuted_tile_indices = torch.zeros_like(tile_indices)
    for i in range(8):
        # select i-th byte, apply transformation and trace where each index ended up
        ith_dim_indices = torch.div(tile_indices, 256**i, rounding_mode="trunc") % 256
        sample_tile_i = (ith_dim_indices - 128).to(torch.int8).contiguous()
        assert torch.all(sample_tile_i.int() + 128 == ith_dim_indices), "int overflow"
        permuted_tile_i = transform_tile(sample_tile_i)
        ith_permuted_indices = permuted_tile_i.to(tile_indices.dtype) + 128
        permuted_tile_indices += ith_permuted_indices * (256**i)
        if d1 * d2 < 256**i:
            break  # if all indices fit in i bytes, stop early
    return permuted_tile_indices


_is_compiling = torch.compiler.is_compiling


@deprecated(
    "This function is deprecated and will be removed in a future release.",
    category=FutureWarning,
)
def undo_layout(permuted_tensor: torch.Tensor, tile_indices: torch.LongTensor) -> torch.Tensor:
    """
    Undo a tiled permutation such as turing or ampere layout

    :param permuted_tensor: torch tensor in a permuted layout
    :param tile_indices: reverse transformation indices, from get_inverse_transform_indices
    :return: contiguous row-major tensor
    """
    (rows, cols), (tile_rows, tile_cols) = permuted_tensor.shape, tile_indices.shape
    assert rows % tile_rows == cols % tile_cols == 0, "tensor must contain a whole number of tiles"
    tensor = permuted_tensor.reshape(-1, tile_indices.numel()).t()
    outputs = torch.empty_like(tensor)  # note: not using .index_copy because it was slower on cuda
    outputs[tile_indices.flatten()] = tensor
    outputs = outputs.reshape(tile_rows, tile_cols, cols // tile_cols, rows // tile_rows)
    outputs = outputs.permute(3, 0, 2, 1)  # (rows // tile_rows, tile_rows), (cols // tile_cols, tile_cols)
    return outputs.reshape(rows, cols).contiguous()


@dataclass
class MatmulLtState:
    _tile_indices: Optional[torch.Tensor] = None  # TODO: remove

    force_no_igemmlt: bool = False

    CB: Optional[torch.Tensor] = None
    CxB: Optional[torch.Tensor] = None  # TODO: Deprecate/remove
    SB: Optional[torch.Tensor] = None
    SCB: Optional[torch.Tensor] = None

    CxBt: Optional[torch.Tensor] = None  # TODO: Deprecate/remove
    SBt: Optional[torch.Tensor] = None
    CBt: Optional[torch.Tensor] = None

    subB: Optional[torch.Tensor] = None

    outlier_pool: Optional[GlobalOutlierPooler] = None
    has_accumulated_gradients = False
    threshold = 0.0
    idx: Optional[torch.Tensor] = None
    is_training = True
    has_fp16_weights = True
    use_pool = False
    formatB = "row"  # TODO: Deprecate/remove

    def reset_grads(self):
        self.CB = None
        self.CxB = None
        self.SB = None
        self.SCB = None

        self.CxBt = None
        self.SBt = None
        self.CBt = None

    @property
    def tile_indices(self):
        raise ValueError("tile_indices is no longer supported.")


class MatMul8bitLt(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        state: Optional[MatmulLtState] = None,
    ):
        state = state or MatmulLtState()

        # default of pytorch behavior if inputs are empty
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

        input_shape = A.shape

        # Cast A to fp16
        if A.dtype != torch.float16 and not _is_compiling():
            warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")

        if len(A.shape) == 3:
            A = A.reshape(-1, A.shape[-1])

        # 1. Quantize A. Note that as a side-effect, outliers are suppressed in CA/CAt.
        if ctx.needs_input_grad[1]:
            # Slower path
            CA, CAt, SCA, SCAt, outlier_cols = F.int8_double_quant(A.to(torch.float16), threshold=state.threshold)
        else:
            # Fast path
            CA, SCA, outlier_cols = F.int8_vectorwise_quant(A.to(torch.float16), threshold=state.threshold)
            CAt = SCAt = None

        has_grad = False

        if state.has_fp16_weights or state.CB is None:
            has_grad = getattr(B, "grad", None) is not None
            is_transposed = not B.is_contiguous() and B.shape[0] == B.stride(1)
            if is_transposed:
                B = B.contiguous()

            if (state.is_training and not has_grad) or state.CB is None or state.SCB is None:
                state.reset_grads()

                # 2. Quantize B
                state.CB, state.SCB, _ = F.int8_vectorwise_quant(B.to(torch.float16))

        # Handle sparse decomposition
        if state.threshold > 0.0:
            state.idx = outlier_cols

            # Mixed Int8 Matmul + Dequant + Bias
            output, subA = torch.ops.bitsandbytes.int8_mixed_scaled_mm(
                A,
                CA,
                state.CB,
                SCA,
                state.SCB,
                outlier_cols,
                bias,
            )

        else:
            # Int8 Matmul + Dequant + Bias
            output = torch.ops.bitsandbytes.int8_scaled_mm.default(
                CA, state.CB, SCA, state.SCB, bias=bias, dtype=A.dtype
            )
            subA = None

        # 5. Save state
        ctx.state = state

        ctx.grad_shape = input_shape
        ctx.dtype_A = A.dtype
        ctx.dtype_bias = None if bias is None else bias.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (CAt, subA, A)
            ctx.tensor_states = (SCAt, state.idx)
        else:
            ctx.tensors = [None, None, None]
            ctx.tensor_states = (None, None)
            ctx.save_for_backward(None, None)

        output_shape = (*input_shape[:-1], state.CB.shape[0])

        if len(input_shape) == 3:
            return output.reshape(output_shape)

        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

        req_gradA, req_gradB, _, req_gradBias, _ = ctx.needs_input_grad
        CAt, subA, A = ctx.tensors
        SCAt, idx = ctx.tensor_states
        state: MatmulLtState = ctx.state
        grad_A = grad_B = grad_bias = None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # Cast grad_output to fp16
        if len(grad_output.shape) == 3:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        if req_gradB:
            Cgrad, _, _, SCgradt, _ = F.int8_double_quant(grad_output.to(torch.float16))

            grad_B = torch.ops.bitsandbytes.int8_scaled_mm.default(
                Cgrad.t().contiguous(),
                CAt.t(),
                SCgradt,
                SCAt,
                dtype=torch.float16,
            )

            if state.threshold > 0.0 and subA is not None and subA.numel() > 0:
                grad_B[:, idx] += torch.matmul(grad_output.t(), subA)

        if req_gradA:
            if state.CB is not None:
                CB = state.CB.to(ctx.dtype_A, copy=True).mul_(state.SCB.unsqueeze(1).mul(1.0 / 127.0))
                grad_A = torch.matmul(grad_output.to(ctx.dtype_A), CB).view(ctx.grad_shape)
            else:
                raise Exception("State must contain CB matrix for backward")

        return grad_A, grad_B, None, grad_bias, None


class MatMul8bitFp(torch.autograd.Function):
    # For Intel CPU and XPU MatMul8bitFp is much faster (~3x) than MatMul8bitLt in finetune.
    # Because the MatMul8bitLt has more mechanisms in computing grad.
    # We don't have fast kernel for quant/dequant 8bit in CPU/XPU, so it's very slow.
    # We'd like to use dequant + matmul to run finetune with good performance.

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, state=MatmulLtState):
        if state.has_fp16_weights or state.CB is None:
            has_grad = getattr(B, "grad", None) is not None
            is_transposed = not B.is_contiguous() and B.shape[0] == B.stride(1)
            if is_transposed:
                B = B.contiguous()

            if (state.is_training and not has_grad) or state.CB is None or state.SCB is None:
                state.reset_grads()
                state.CB, state.SCB, _ = F.int8_vectorwise_quant(B.to(torch.float16))
                B = state.CB

        CB = state.CB.data.to(A.dtype).mul_(state.SCB.unsqueeze(1).mul(1.0 / 127.0))
        output = torch.nn.functional.linear(A, CB, bias)
        ctx.state = state
        ctx.dtype_A = A.dtype
        ctx.grad_shape = A.shape
        ctx.A = A
        ctx.dtype_bias = None if bias is None else bias.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        req_gradA, req_gradB, _, req_gradBias, _ = ctx.needs_input_grad
        A = ctx.A
        state = ctx.state
        grad_A = grad_B = grad_bias = None
        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # Cast grad_output to fp16
        if len(grad_output.shape) == 3:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        if req_gradB:
            grad_B = torch.matmul(A.t(), grad_output).t()

        if req_gradA:
            if state.CB is not None:
                CB = state.CB.to(ctx.dtype_A, copy=True).mul_(state.SCB.unsqueeze(1).mul(1.0 / 127.0))
                grad_A = torch.matmul(grad_output.to(ctx.dtype_A), CB).view(ctx.grad_shape)
            else:
                raise Exception("State must contain CB matrix for backward")

        return grad_A, grad_B, None, grad_bias, None


class MatMul4Bit(torch.autograd.Function):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state: Optional[F.QuantState] = None):
        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            B_shape = quant_state.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

        # 1. Dequantize
        # 2. MatmulnN
        output = torch.nn.functional.linear(A, F.dequantize_4bit(B, quant_state).to(A.dtype).t(), bias)

        # 3. Save state
        ctx.state = quant_state
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (None, B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

        req_gradA, _, _, req_gradBias, _ = ctx.needs_input_grad
        _, B = ctx.tensors

        grad_A, grad_B, grad_bias = None, None, None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # not supported by PyTorch. TODO: create work-around
        # if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        if req_gradA:
            grad_A = torch.matmul(grad_output, F.dequantize_4bit(B, ctx.state).to(grad_output.dtype).t())

        return grad_A, grad_B, None, grad_bias, None


def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    state: Optional[MatmulLtState] = None,
    threshold=0.0,
    bias: Optional[torch.Tensor] = None,
):
    state = state or MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold
    # MatMul8bitLt is slower because no fast kernel for quant/dequant 8bit in CPU/XPU
    if state.is_training:
        if A.device.type in ("cpu", "xpu"):
            return MatMul8bitFp.apply(A, B, out, bias, state)
    return MatMul8bitLt.apply(A, B, out, bias, state)


def matmul_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    quant_state: F.QuantState,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
):
    assert quant_state is not None

    if A.numel() == A.shape[-1] and A.requires_grad == False and A.device.type != "hpu":
        if A.shape[-1] % quant_state.blocksize != 0:
            warn(
                f"Some matrices hidden dimension is not a multiple of {quant_state.blocksize} and efficient inference kernels are not supported for these (slow). Matrix input size found: {A.shape}",
            )
            return MatMul4Bit.apply(A, B, out, bias, quant_state)
        else:
            out = F.gemv_4bit(A, B.t(), out, state=quant_state)
            if bias is not None:
                out += bias
            return out
    else:
        return MatMul4Bit.apply(A, B, out, bias, quant_state)
