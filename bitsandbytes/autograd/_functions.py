import torch
import math
import bitsandbytes.functional as F
from scipy.stats import norm

class MatMul8bit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, out=None, quant_type='vector', precision=[8, 8, 8], index=None):

        if precision[0] != 8:
            with torch.no_grad():
                output = torch.matmul(A, B)
        else:
            if len(B.shape) == 2: dim = 0
            else: dim = 1
            qA, SA = F.vectorwise_quant(A, dim=-1, quant_type=quant_type)
            qB, SB = F.vectorwise_quant(B, dim=dim, quant_type=quant_type)
            iout = F.igemm(qA, qB)
            output = F.vectorwise_mm_dequant(iout, SA, SB, A.dtype, quant_type)

        if A.requires_grad or B.requires_grad:
            ctx.save_for_backward(A, B)

        #if torch.rand(1) < 0.01 and A.shape[-1] == 8192:
        #if A.shape[-1] == 8192 and index == 3:
        if torch.rand(1) < 0.01:
            stda = A.std([2]).view(-1)
            stdb = B.std([0])
            ma = A.mean([2])
            mb = B.mean([1])
            #n = math.sqrt(A.shape[-1])
            n = A.shape[-1]
            n = math.sqrt(n)
            #if n == 8192:
            #    n = math.sqrt(n*(1-(2/math.pi)))
            #else:
            #    n = math.sqrt(n)
            stdc = (n*stda*stdb.mean()).view(-1)
            stdout = output.std([2]).view(-1)
            C = output.view(-1, output.shape[-1])
            val = norm.ppf(0.99, loc=0, scale=stdout[0].item())
            #print(C.shape, stdc.shape, stdout.shape, stda.shape, stdb.shape)
            #print(index)
            #print(stda[:10])
            #print(stdb[:10])
            #print(stdb.mean())
            #print(stdc[:10])
            #print(stdout[:10])
            print(val, 'val', stdout[0].item(),(torch.rand(1).cuda()-0.5)/0.5*0.15)
            cutoff = (stdout+(((torch.rand(1).cuda()-0.5)/0.5*0.15)*stdout))*3
            #cutoff = stdout*3
            quants = F.estimate_quantiles(C[37])
            values = []
            for i, row in enumerate(C):
                flowed = ((row>cutoff[i]).float() + (row<-cutoff[i]).float()).sum().item()
                values.append((flowed/C.shape[1]))
            print(index, stdout[:10].mean().item(),stdc[:10].mean().item(), stda[:10].mean().item(), stdb.mean().item(), mb[:10].mean().item(), sum(values)/len(values), C.shape[-1])
            #if sum(values)/len(values) > 0.5:
            #    print(quants)
            #    print(C[37])
            #    print(index, quants[-2], stdout[37].item(), stda[37], sum(values)/len(values))

        #if torch.rand(1) < 0.001:
        #    overflows_grad_B = (iout.float()/(127))*SA
        #    print(F.estimate_quantiles(overflows_grad_B))
        #    #print(overflows_grad_B.flatten()[:10])
        #    n = overflows_grad_B.numel()
        #    print(overflows_grad_B.min(), overflows_grad_B.max(), overflows_grad_B.mean(), overflows_grad_B.median())
        #    overflows = (overflows_grad_B > 127).sum().item() + (overflows_grad_B < -127).sum().item()
        #    print(overflows/n)
        ctx.quant_type = quant_type
        ctx.precision = precision

        return output

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        quant_type = ctx.quant_type
        precision = ctx.precision
        grad_A = grad_B = None

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
                    #if torch.rand(1) < 0.001:
                    #    overflows_grad_B = (igrad_B.float()/(127*127))*S2.t()
                    #    print(F.estimate_quantiles(overflows_grad_B))
                    #    #print(overflows_grad_B.flatten()[:10])
                    #    n = overflows_grad_B.numel()
                    #    print(overflows_grad_B.min(), overflows_grad_B.max(), overflows_grad_B.mean(), overflows_grad_B.median())
                    #    overflows = (overflows_grad_B > 127).sum().item() + (overflows_grad_B < -127).sum().item()
                    #    print(overflows/n)
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
                    grad_A = F.vectorwise_mm_dequant(igrad_A, S1, S3.permute(permute_dim), grad_output.dtype, quant_type)
                    #if torch.rand(1) < 0.001:
                    #    overflows_grad_B = (igrad_A.float()/(127))*S1
                    #    print(F.estimate_quantiles(overflows_grad_B))
                    #    #print(overflows_grad_B.flatten()[:10])
                    #    n = overflows_grad_B.numel()
                    #    print(overflows_grad_B.min(), overflows_grad_B.max(), overflows_grad_B.mean(), overflows_grad_B.median())
                    #    overflows = (overflows_grad_B > 127).sum().item() + (overflows_grad_B < -127).sum().item()
                    #    print(overflows/n)

        return grad_A, grad_B, None, None, None, None


mm = MatMul8bit.apply
bmm = MatMul8bit.apply
matmul = MatMul8bit.apply

