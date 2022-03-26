import torch
import math
import bitsandbytes.functional as F
from scipy.stats import norm
import torch.distributed as dist

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

def post_scale(A, B, iout, s, idx, args, absmax, name):
    p = args.scale_p
    scale_mode = getattr(args, 'scale_mode', 'none')
    offset = args.scale_offset
    beta1 = args.scale_beta1
    beta2 = args.scale_beta2
    thresh = args.scale_threshold
    absmax1 = absmax[0]
    absmax2 = absmax[1]
    iter_val = 0

    if scale_mode == 'none':
        pass
    elif scale_mode == 'last':
        iout2 = ((iout.float()/s[idx]).clamp(-127, 127)).int()

        n = iout.numel()
        count = ((iout2 == -127) +  (iout2 == 127)).float().sum()
        flowed = count/n
        count2 = ((iout2 < -127+offset) + (iout2 > 127-offset)).float().sum()
        gap = count2/n

        if s is not None and flowed < args.scale_flow_thresh:
            iout = (iout2.float()*s[idx]).int()
            if torch.rand(1) < 0.00001:
                print('s', name)
        else:
            if torch.rand(1) < 0.00001:
                print('b', name)

        print_overflow(flowed, gap.item(),  s[idx].item(), name)

        p_flowed = 0.0
        loss = 0.0
        if flowed > 0:
            loss += -0.5*(flowed-p_flowed)*beta1
        if gap < p and loss < thresh:
            if loss == 0 and gap == 0:
                loss += -0.5*(gap-p)*beta1
            else:
                loss += -0.5*(gap-p)*beta2
        s.data[idx] -= loss
    elif scale_mode == 'absmax':
        #assert args.scale_layers == 'grad'
        #absmax_scale = s.max(1).values*127
        #iout2 = ((iout.float()/absmax_scale).clamp(-127, 127)).int()

        #n = iout.numel()
        #count = ((iout2 == -127) +  (iout2 == 127)).float().sum()
        #flowed = count/n

        #if s is not None and flowed < args.scale_flow_thresh:
        #    iout = (iout2.float()*absmax_scale).int()
        #    if torch.rand(1) < 0.01:
        #        print('s', name)
        #else:
        #    if torch.rand(1) < 0.01:
        #        print('b', name)

        #print_overflow(flowed, 0,  absmax_scale.max().item(), name)

        #absmax = torch.abs(iout).max(0).values
        #s[:, iter_val] = absmax
        #print(absmax[:3], absmax_scale[:3]/127, flowed)


        if s[0, 0].item() == 4:
            if not dist.is_initialized() or dist.get_rank() == 0:
                #print(torch.abs(iout.float()).view(-1, iout.shape[-1]).max(0).values[:5], name, s[0, 0].item(), 'n', iout.shape)
                #print(torch.abs(iout.float()).view(-1, iout.shape[-1]).max(1).values[:5], name, s[0, 0].item(), 't')
                #print(iout.float().view(-1, iout.shape[-1]).std(0)[:5], name, s[0, 0].item())
                #print(torch.abs(iout).float().view(-1, iout.shape[-1]).mean(0)[:5], name, s[0, 0].item())
                dim = (0 if name == 'wgrad' else 1)
                maxval = torch.abs(iout.float()).view(-1, iout.shape[-1]).max(dim).values[:5]
                maxtotal = torch.abs(iout.float()).view(-1, iout.shape[-1]).max().item()
                stdval1 = A.view(-1, A.shape[-1]).float().std(1)[:5]
                stdval2 = B.float().std(0).mean()
                idx = 0
                quant = F.estimate_quantiles(torch.abs(iout.view(-1, iout.shape[-1]))[idx, :].float(), offset=0)
                val1 = 8e5
                val2 = 8.2e5
                #if maxval[idx] < val2 and maxval[idx] > val1:
                #if stdval1[idx].int() == 20:
                if maxval[idx].int() > 1e6:
                    print(absmax1.view(-1)[idx].item(), maxval[idx].int().item(), stdval1[idx].int().item(), (stdval1[idx]*stdval2).int().item(), quant[-2].int().item(), maxtotal)
                idx = 1
                val1 = 6e5
                val2 = 6.5e5
                quant = F.estimate_quantiles(torch.abs(iout.view(-1, iout.shape[-1]))[idx, :].float(), offset=0)
                if maxval[idx].int() < 6e5 and maxval[idx].int() > 4e5:
                #if maxval[idx] < val2 and maxval[idx] > val1:
                    print(absmax1.view(-1)[idx].item(), maxval[idx].int().item(), stdval1[idx].int().item(), (stdval1[idx]*stdval2).int().item(), quant[-2].int().item())
                #print(absmax1.view(-1)[:5], absmax2.view(-1)[:5], absmax1.max(), absmax2.max(), 'max')
                #stdval = iout.float().view(-1, iout.shape[-1]).std(0)[:5]
                #meanval = torch.abs(iout).float().view(-1, iout.shape[-1]).mean(0)[:5]
                #for m, s in zip(maxval, stdval1):
                    #print(m.int().item(), s.int().item(), (s*stdval2).int().item(), name)
                #print(absmax1.view(-1)[:5], name, s[0].item())
                #print(absmax1.view(-1)[:5]*absmax2.mean(), name, s[0].item())

    return iout

class MatMul8bit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, out=None, quant_type='vector', precision=[8, 8, 8], index=None, s=None, args=None):
        gemm_func = torch.matmul if quant_type == 'zeropoint' else F.igemm

        if precision[0] != 8:
            with torch.no_grad():
                output = torch.matmul(A, B)
        else:
            if len(B.shape) == 2: dim = 0
            else: dim = 1
            if quant_type == 'row':
                qA, SA = F.vectorwise_quant(A, dim=-1, quant_type='linear')
            else:
                qA, SA = F.vectorwise_quant(A, dim=-1, quant_type=quant_type)
            qB, SB = F.vectorwise_quant(B, dim=dim, quant_type=quant_type)
            iout = gemm_func(qA, qB)

            output = F.vectorwise_mm_dequant(iout, SA, SB, A.dtype, quant_type)

        if A.requires_grad or B.requires_grad:
            ctx.save_for_backward(A, B)

        ctx.quant_type = quant_type
        ctx.precision = precision
        ctx.s = s
        ctx.args = args

        return output

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        s, args = ctx.s, ctx.args
        quant_type = ctx.quant_type
        precision = ctx.precision
        grad_A = grad_B = None
        grad_s = getattr(ctx, 'grad_s', None)
        gemm_func = torch.matmul if quant_type == 'zeropoint' else F.igemm

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
                    igrad_B = gemm_func(qA.t(), qgrad_output)
                    grad_B = F.vectorwise_mm_dequant(igrad_B, S2.t(), S1, grad_output.dtype, quant_type)
                else:
                    if quant_type == 'row':
                        qgrad_output, S1 = F.vectorwise_quant(grad_output, dim=dims, quant_type='linear')
                    else:
                        qgrad_output, S1 = F.vectorwise_quant(grad_output, dim=dims, quant_type=quant_type)
                    qA, S2 = F.vectorwise_quant(A, dim=dims, quant_type=quant_type)
                    igrad_B = gemm_func(qA.permute(permute_dim), qgrad_output)
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
                    igrad_A = gemm_func(qgrad_output, qB.permute(permute_dim))
                    grad_A = F.vectorwise_mm_dequant(igrad_A, S1, S3, grad_output.dtype, quant_type)
                else:
                    qB, S3 = F.vectorwise_quant(B, dim=dim_B, quant_type=quant_type)
                    igrad_A = gemm_func(qgrad_output, qB.permute(permute_dim))
                    if quant_type in ['linear', 'zeropoint']:
                        grad_A = F.vectorwise_mm_dequant(igrad_A, S1, S3, grad_output.dtype, quant_type)
                    else:
                        grad_A = F.vectorwise_mm_dequant(igrad_A, S1, S3.permute(permute_dim), grad_output.dtype, quant_type)

        return grad_A, grad_B, None, None, None, None, None, None


mm = MatMul8bit.apply
bmm = MatMul8bit.apply
matmul = MatMul8bit.apply

