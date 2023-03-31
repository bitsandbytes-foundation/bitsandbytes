import torch
import torch.nn as nn
import time

from .triton_utils.v0.quantize_rowwise_nogroup import quantize_rowwise_nogroup
from .triton_utils.v0.quantize_columnwise_nogroup_transpose import quantize_columnwise_nogroup_transpose
from .triton_utils.v0.int8_matmul_rowwise_dequantize_bias import int8_matmul_rowwise_dequantize_bias
from .triton_utils.v0.int8_matmul_rowwise_dequantize import int8_matmul_rowwise_dequantize
from .triton_utils.v0.quantize_global import quantize_global, quantize_global_transpose
from .triton_utils.v0.int8_matmul_mixed_dequanitze import int8_matmul_mixed_dequanitze, int8_matmul_mixed_dequanitze_bias
from .triton_utils.v0.fused_gelu_quantize import quantize_rowwise_nogroup_gelu, quantize_rowwise_nogroup_back_gelu

class _switchback(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_3D, W, bias):

        X = X_3D.view(-1, X_3D.size(-1))

        ctx.save_for_backward = X, W
        X_int8, state_X = quantize_rowwise_nogroup(X)
        W_int8, state_W = quantize_rowwise_nogroup(W)
        return int8_matmul_rowwise_dequantize_bias(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D.size()[:-1], -1)
    
    @staticmethod
    def backward(ctx, G_3D):
        X, W = ctx.save_for_backward

        G = G_3D.reshape(-1, G_3D.size(-1))

        grad_X = grad_W = grad_bias = None

        if ctx.needs_input_grad[0]:
            G_int8, state_G = quantize_rowwise_nogroup(G)
            W_int8, state_W = quantize_columnwise_nogroup_transpose(W)
            grad_X = int8_matmul_rowwise_dequantize(G_int8, W_int8.t(), state_G, state_W).view(
                *G_3D.size()[:-1], -1
            )
        if ctx.needs_input_grad[1]:
            grad_W = torch.matmul(G.t(), X.to(G.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias

class SwitchBackLinear(nn.Linear):

    def prepare_for_eval(self):
        state_W = self.weight.abs().max(dim=1, keepdim=True)[0]
        W_int8 = (127 * self.weight.float() / state_W).round().to(torch.int8)
        state_W = state_W.squeeze()
        
        self.register_buffer("W_int8", W_int8)
        self.register_buffer("state_W", state_W)

        del self.weight

    def forward(self, x):
        if self.training:
            return _switchback.apply(x, self.weight, self.bias)
        else:
            if not hasattr(self, "state_W"):
                self.prepare_for_eval()
            X = x.view(-1, x.size(-1))
            X_int8, state_X = quantize_rowwise_nogroup(X)
            return int8_matmul_rowwise_dequantize_bias(
                X_int8, self.W_int8.t(), state_X, self.state_W, self.bias
            ).view(*x.size()[:-1], -1)
    

class _switchback_global(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_3D, W, bias):

        X = X_3D.view(-1, X_3D.size(-1))

        X_int8, state_X = quantize_rowwise_nogroup(X)
        W_int8, state_W = quantize_global(W)
        ctx.save_for_backward = X, W
        return int8_matmul_mixed_dequanitze_bias(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D.size()[:-1], -1)

    @staticmethod
    def backward(ctx, G_3D):

        G = G_3D.reshape(-1, G_3D.size(-1))

        grad_X = grad_W = grad_bias = None

        X, W = ctx.save_for_backward
        if ctx.needs_input_grad[0]:
            G_int8, state_G = quantize_rowwise_nogroup(G)
            W_int8, state_W = quantize_global_transpose(W)
            grad_X = int8_matmul_mixed_dequanitze(G_int8, W_int8.t(), state_G, state_W).view(
                *G_3D.size()[:-1], -1
            )
        if ctx.needs_input_grad[1]:
            grad_W = torch.matmul(G.t(), X.to(G.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias
    


class SwitchBackGlobalLinear(nn.Linear):

    def prepare_for_eval(self):
        state_W = self.weight.abs().max()
        W_int8 = (127 * self.weight.float() / state_W).round().to(torch.int8)
        
        self.register_buffer("W_int8", W_int8)
        self.register_buffer("state_W", state_W)

        del self.weight

    def forward(self, x):
        if self.training:
            return _switchback_global.apply(x, self.weight, self.bias)
        else:
            if not hasattr(self, "state_W"):
                self.prepare_for_eval()
            X = x.view(-1, x.size(-1))
            X_int8, state_X = quantize_rowwise_nogroup(X)
            return int8_matmul_mixed_dequanitze_bias(
                X_int8, self.W_int8.t(), state_X, self.state_W, self.bias
            ).view(*x.size()[:-1], -1)
        



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
    
