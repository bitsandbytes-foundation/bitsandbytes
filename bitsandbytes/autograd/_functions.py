import torch
import math
import bitsandbytes.functional as F
from scipy.stats import norm

def print_overflow(flowed, gap, s, name):
    if flowed > 0.5:
        if torch.rand(1) < 0.05:
            print(flowed.item(), gap, s, name)
    elif flowed > 0.3:
        if torch.rand(1) < 0.01:
            print(flowed.item(), gap, s, name)
    elif flowed > 0.07:
        if torch.rand(1) < 0.001:
            print(flowed.item(), gap, s, name)
    else:
        if torch.rand(1) < 0.0001:
            print(flowed.item(), gap, s, name)

def post_scale(iout, s, idx, p, scale_mode, name):
    if scale_mode == 'none':
        pass
    elif scale_mode == 'last':
        iout2 = ((iout.float()/s[idx]).clamp(-127, 127)).int()

        n = iout.numel()
        count = ((iout2 == -127) +  (iout2 == 127)).float().sum()
        flowed = count/n
        count2 = ((iout2 < -100) + (iout2 > 100)).float().sum()
        gap = count2/n

        if s is not None and flowed < 0.0005:
            iout = (iout2.float()*s[idx]).int()
            #if torch.rand(1) < 0.001:
            #    print('s', name)
        else:
            pass
            #if torch.rand(1) < 0.001:
            #    print('b', name)

        print_overflow(flowed, gap.item(),  s[idx].item(), name)

        p_flowed = 0.0
        loss = 0.0
        if flowed > 0:
            loss += -10
        if gap < p and loss < 1e-7:
            if loss == 0 and gap == 0:
                loss += -0.5*(gap-p)*10000
            else:
                loss += -0.5*(gap-p)*1000
        s.data[idx] -= loss

    return iout

class MatMul8bit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, out=None, quant_type='vector', precision=[8, 8, 8], index=None, s=None, scale_mode='last', p=0.01):

        if precision[0] != 8:
            with torch.no_grad():
                output = torch.matmul(A, B)
        else:
            if len(B.shape) == 2: dim = 0
            else: dim = 1
            qA, SA = F.vectorwise_quant(A, dim=-1, quant_type=quant_type)
            qB, SB = F.vectorwise_quant(B, dim=dim, quant_type=quant_type)
            iout = F.igemm(qA, qB)

            iout = post_scale(iout, s, 0, p, scale_mode, 'fw')

            output = F.vectorwise_mm_dequant(iout, SA, SB, A.dtype, quant_type)

        if A.requires_grad or B.requires_grad:
            ctx.save_for_backward(A, B)

        ctx.quant_type = quant_type
        ctx.precision = precision
        ctx.s = s
        ctx.p = p
        ctx.scale_mode = scale_mode

        return output

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        s, p, scale_mode = ctx.s, ctx.p, ctx.scale_mode
        quant_type = ctx.quant_type
        precision = ctx.precision
        grad_A = grad_B = None
        grad_s = getattr(ctx, 'grad_s', None)
        #print(grad_s, 'backprop')

        if B.requires_grad:
            if len(A.shape) == 3:
                dims = [0, 1]
                # bsi -> ibs
                permute_dim = [0, 2, 1]
            else:
                dims = [0]
                # bs -> sb
                permute_dim = [1, 0]

            if precision[1] != 8:
                with torch.no_grad():
                    grad_B = torch.matmul(A.permute(permute_dim), grad_output)
            else:
                if len(B.shape) == 2 and len(A.shape) == 3:
                    grad_output = grad_output.contiguous()
                    if not grad_output.is_contiguous(): grad_output.contiguous()
                    if quant_type == 'row':
                        qgrad_output, S1 = F.vectorwise_quant(grad_output.view(-1, grad_output.shape[2]), dim=0, quant_type='linear')
                    else:
                        qgrad_output, S1 = F.vectorwise_quant(grad_output.view(-1, grad_output.shape[2]), dim=0, quant_type=quant_type)
                    if not A.is_contiguous(): A = A.contiguous()
                    qA, S2 = F.vectorwise_quant(A.view(-1, A.shape[2]), dim=0, quant_type=quant_type)
                    igrad_B = F.igemm(qA.t(), qgrad_output)
                    #print(igrad_B.shape, 'b')
                    igrad_B = post_scale(igrad_B, s, 1, p, scale_mode, 'wgrad')
                    grad_B = F.vectorwise_mm_dequant(igrad_B, S2.t(), S1, grad_output.dtype, quant_type)
                else:
                    if quant_type == 'row':
                        qgrad_output, S1 = F.vectorwise_quant(grad_output, dim=dims, quant_type='linear')
                    else:
                        qgrad_output, S1 = F.vectorwise_quant(grad_output, dim=dims, quant_type=quant_type)
                    qA, S2 = F.vectorwise_quant(A, dim=dims, quant_type=quant_type)
                    igrad_B = F.igemm(qA.permute(permute_dim), qgrad_output)
                    grad_B = F.vectorwise_mm_dequant(igrad_B, S2.permute(permute_dim), S1, grad_output.dtype, quant_type)

        if A.requires_grad:
            if len(grad_output.shape) == 3: dims = [2]
            else: dims = [1]

            if len(B.shape) == 3:
                # bio -> boi
                permute_dim = [0, 2, 1]
                dim_B = dims
            else:
                # io -> oi
                permute_dim = [1, 0]
                dim_B = [1]

            if precision[2] != 8:
                with torch.no_grad():
                    grad_A = torch.matmul(grad_output, B.permute(permute_dim))
            else:
                qgrad_output, S1 = F.vectorwise_quant(grad_output, dim=dims, quant_type=quant_type)
                if quant_type == 'row':
                    qB, S3 = F.vectorwise_quant(B, dim=dim_B, quant_type='linear')
                    igrad_A = F.igemm(qgrad_output, qB.permute(permute_dim))
                    grad_A = F.vectorwise_mm_dequant(igrad_A, S1, S3, grad_output.dtype, quant_type)
                else:
                    qB, S3 = F.vectorwise_quant(B, dim=dim_B, quant_type=quant_type)
                    igrad_A = F.igemm(qgrad_output, qB.permute(permute_dim))
                    #print(igrad_A.shape, 'a')
                    #igrad_A = post_scale(igrad_A, s, 2, p, scale_mode, 'bw')
                    grad_A = F.vectorwise_mm_dequant(igrad_A, S1, S3.permute(permute_dim), grad_output.dtype, quant_type)
                    #if torch.rand(1) < 0.001:
                    #    overflows_grad_B = (igrad_A.float()/(127))*S1
                    #    print(F.estimate_quantiles(overflows_grad_B))
                    #    #print(overflows_grad_B.flatten()[:10])
                    #    n = overflows_grad_B.numel()
                    #    print(overflows_grad_B.min(), overflows_grad_B.max(), overflows_grad_B.mean(), overflows_grad_B.median())
                    #    overflows = (overflows_grad_B > 127).sum().item() + (overflows_grad_B < -127).sum().item()
                    #    print(overflows/n)

        return grad_A, grad_B, None, None, None, None, None, None, None


mm = MatMul8bit.apply
bmm = MatMul8bit.apply
matmul = MatMul8bit.apply

