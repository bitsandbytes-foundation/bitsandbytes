import torch
import torch.nn as nn
import time
from functools import partial

from .triton_utils.v0.dequantize_rowwise import dequantize_rowwise
from .triton_utils.v0.quantize_rowwise import quantize_rowwise
from .triton_utils.v0.quantize_columnwise_and_transpose import quantize_columnwise_and_transpose
from .triton_utils.v0.int8_matmul_rowwise_dequantize import int8_matmul_rowwise_dequantize
from .triton_utils.v0.quantize_global import quantize_global, quantize_global_transpose
from .triton_utils.v0.int8_matmul_mixed_dequanitze import int8_matmul_mixed_dequanitze


class _switchback_global(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_3D, W, bias):
        # reshape input to [N * L, D]
        X = X_3D.view(-1, X_3D.size(-1))

        # rowwise quantize for X, global quantize for W
        X_int8, state_X = quantize_rowwise(X)
        W_int8, state_W = quantize_global(W)

        # save for backward.
        ctx.save_for_backward = X, W

        # matmult, fused dequant and add bias
        # call "mixed" because we are mixing rowwise quantized and global quantized
        return int8_matmul_mixed_dequanitze(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D.size()[:-1], -1)

    @staticmethod
    def backward(ctx, G_3D):
        # reshape input to [N_out * L, D]
        G = G_3D.reshape(-1, G_3D.size(-1))

        grad_X = grad_W = grad_bias = None

        X, W = ctx.save_for_backward
        if ctx.needs_input_grad[0]:
            # rowwise quantize for G, global quantize for W
            # for W, we also fuse the transpose operation because only A @ B^T is supported
            # so we transpose once then call .t() in the matmul
            G_int8, state_G = quantize_rowwise(G)
            W_int8, state_W = quantize_global_transpose(W)
            grad_X = int8_matmul_mixed_dequanitze(G_int8, W_int8.t(), state_G, state_W, None).view(
                *G_3D.size()[:-1], -1
            )
        if ctx.needs_input_grad[1]:
            # backward pass uses standard weight grad
            grad_W = torch.matmul(G.t(), X.to(G.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias
    
class _switchback_vectorrize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_3D, W, bias):
        # reshape input to [N * L, D]
        X = X_3D.view(-1, X_3D.size(-1))

        ctx.save_for_backward = X, W
        # rowwise quantize for X
        # columnwise quantize for W (first rowwise, transpose later)
        X_int8, state_X = quantize_rowwise(X)
        W_int8, state_W = quantize_rowwise(W)

        # matmult, fused dequant and add bias
        # call kernel which expects rowwise quantized X and W
        return int8_matmul_rowwise_dequantize(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D.size()[:-1], -1)
    
    @staticmethod
    def backward(ctx, G_3D):
        X, W = ctx.save_for_backward

        G = G_3D.reshape(-1, G_3D.size(-1))

        grad_X = grad_W = grad_bias = None

        if ctx.needs_input_grad[0]:
            # rowwise quantize for G, columnwise quantize for W and fused transpose
            # we call .t() for weight later because only A @ B^T is supported
            G_int8, state_G = quantize_rowwise(G)
            W_int8, state_W = quantize_columnwise_and_transpose(W)
            grad_X = int8_matmul_rowwise_dequantize(G_int8, W_int8.t(), state_G, state_W, None).view(
                *G_3D.size()[:-1], -1
            )
        if ctx.needs_input_grad[1]:
            # backward pass uses standard weight grad
            grad_W = torch.matmul(G.t(), X.to(G.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias
    
class _switchback_global_mem_efficient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_3D, W, bias):
        # reshape input to [N * L, D]
        X = X_3D.view(-1, X_3D.size(-1))
        X_3D_sz = X_3D.size()

        # rowwise quantize for X, global quantize for W
        X_int8, state_X = quantize_rowwise(X)
        del X
        W_int8, state_W = quantize_global(W)

        print('in mem eff backward.')

        # save for backward.
        ctx.save_for_backward = X_int8, state_X, W_int8, state_W

        # matmult, fused dequant and add bias
        # call "mixed" because we are mixing rowwise quantized and global quantized
        return int8_matmul_mixed_dequanitze(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D_sz[:-1], -1)

    @staticmethod
    def backward(ctx, G_3D):
        # reshape input to [N_out * L, D]
        G = G_3D.reshape(-1, G_3D.size(-1))
        G_3D_sz = G_3D.size()

        grad_X = grad_W = grad_bias = None

        X_int8, state_X, W_int8, state_W = ctx.save_for_backward
        if ctx.needs_input_grad[1]:
            real_X = dequantize_rowwise(X_int8, state_X)
            del X_int8
            grad_W = torch.matmul(G.t(), real_X.to(G.dtype))
            del real_X
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)
        if ctx.needs_input_grad[0]:
            G_int8, state_G = quantize_rowwise(G)
            del G
            W_int8 = W_int8.t().contiguous()
            grad_X = int8_matmul_mixed_dequanitze(G_int8, W_int8.t(), state_G, state_W, None).view(
                *G_3D_sz[:-1], -1
            )

        return grad_X, grad_W, grad_bias

class SwitchBackLinear(nn.Linear):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            bias: bool = True,
            device=None, 
            dtype=None,
            vectorize: bool = False,
            mem_efficient : bool = False,
        ):
        super().__init__(in_features, out_features, bias, device, dtype)

        # By default, we use the global quantization.
        self.vectorize = vectorize
        if self.vectorize:
            self._fn = _switchback_vectorrize
            if mem_efficient:
                print('mem efficient is not supported for vectorize mode.')
                exit(1)
        else:
            if mem_efficient:
                self._fn = _switchback_global_mem_efficient
            else:
                self._fn = _switchback_global

    def prepare_for_eval(self):
        # If we just want to do eval, we can pre-quantize the weights instead of doing it on the forward pass.
        # Note this is experimental and not tested thoroughly.
        # Note this needs to be explicitly called with something like
        # def cond_prepare(m):
        #     if hasattr(m, "prepare_for_eval"):
        #         m.prepare_for_eval()
        # model.apply(cond_prepare)
        print('=> preparing for eval.')
        if self.vectorize:
            W_int8, state_W = quantize_rowwise(self.weight)
        else:
            W_int8, state_W = quantize_global(self.weight)
        
        self.register_buffer("W_int8", W_int8)
        self.register_buffer("state_W", state_W)

        del self.weight

    def forward(self, x):
        if self.training:
            return self._fn.apply(x, self.weight, self.bias)
        else:
            # If it hasn't been "prepared for eval", run the standard forward pass.
            if not hasattr(self, "W_int8"):
                return self._fn.apply(x, self.weight, self.bias)
            
            # Otherwise, use pre-computed weights.
            X = x.view(-1, x.size(-1))
            X_int8, state_X = quantize_rowwise(X)

            if self.vectorize:
                return int8_matmul_rowwise_dequantize(
                    X_int8, self.W_int8.t(), state_X, self.state_W, self.bias
                ).view(*x.size()[:-1], -1)
            else:
                return int8_matmul_mixed_dequanitze(
                    X_int8, self.W_int8.t(), state_X, self.state_W, self.bias
                ).view(*x.size()[:-1], -1)

SwitchBackLinearGlobal = partial(SwitchBackLinear, vectorize=False)
SwitchBackLinearGlobalMemEfficient = partial(SwitchBackLinear, vectorize=False, mem_efficient=True)
SwitchBackLinearVectorized = partial(SwitchBackLinear, vectorize=True)

# This is just the standard linear function.
class StandardLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        X = input.view(-1, input.size(-1))

        ctx.save_for_backward(X, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output.view(*input.size()[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output_3D):
        input, weight, bias = ctx.saved_tensors

        grad_output = grad_output_3D.reshape(-1, grad_output_3D.size(-1))

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight.to(grad_output.dtype)).view(*grad_output_3D.size()[:-1], -1)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input.to(grad_output.dtype))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class StandardLinear(nn.Linear):

    def forward(self, x):
        return StandardLinearFunction.apply(x, self.weight, self.bias)
    
