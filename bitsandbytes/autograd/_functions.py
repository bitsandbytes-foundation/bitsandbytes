import torch
import bitsandbytes.functional as F

class MatMul8bit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, out=None, quant_type='vector'):
        if len(B.shape) == 2: dim = 0
        else: dim = 1
        qA, SA = F.vectorwise_quant(A, dim=-1, quant_type=quant_type)
        qB, SB = F.vectorwise_quant(B, dim=dim, quant_type=quant_type)
        iout = F.igemm(qA, qB)
        output = F.vectorwise_mm_dequant(iout, SA, SB, A.dtype, quant_type)

        #if A.requires_grad or B.requires_grad:
        ctx.save_for_backward(A, B)
        ## in some rare cases, the following cases can save some memory
        #elif A.requires_grad and not B.requires_grad:
        #    ctx.save_for_backward(None, B)
        #elif not A.requires_grad and B.requires_grad:
        #    ctx.save_for_backward(A, None)
        #else:
        #    ctx.save_for_backward(None, None)
        #if torch.rand(1) < 0.01:
        #    with torch.no_grad():
        #        out2 = torch.matmul(A, B)
        #        err = torch.abs(output-out2)
        #        print(err.mean().item(), 'output')

        ctx.quant_type = quant_type

        return output

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        quant_type = ctx.quant_type
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
            if len(B.shape) == 2 and len(A.shape) == 3:
                grad_output = grad_output.contiguous()
                qgrad_output, S1 = F.vectorwise_quant(grad_output.view(-1, grad_output.shape[2]), dim=0, quant_type=quant_type)
                qA, S2 = F.vectorwise_quant(A.view(-1, A.shape[2]), dim=0, quant_type=quant_type)
                igrad_B = F.igemm(qA.t(), qgrad_output)
                grad_B = F.vectorwise_mm_dequant(igrad_B, S2.t(), S1, grad_output.dtype, quant_type)
            else:
                #print(grad_output.shape, A.shape, B.shape, 'bw1')
                qgrad_output, S1 = F.vectorwise_quant(grad_output, dim=dims, quant_type=quant_type)
                qA, S2 = F.vectorwise_quant(A, dim=dims, quant_type=quant_type)
                igrad_B = F.igemm(qA.permute(permute_dim), qgrad_output)
                grad_B = F.vectorwise_mm_dequant(igrad_B, S2.permute(permute_dim), S1, grad_output.dtype, quant_type)

            #if torch.rand(1) < 0.01:
            #    with torch.no_grad():
            #        grad_B2 = torch.matmul(A.permute(permute_dim), grad_output)
            #        err = torch.abs(grad_B-grad_B2)
            #        print(err.mean().item(), 'grad_B')
            #        del grad_B2, err

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

            #if len(A.shape) == 3 and len(B.shape) == 3:
                #print(grad_output.shape, A.shape, B.shape, 'bw2')
            qgrad_output, S1 = F.vectorwise_quant(grad_output, dim=dims, quant_type=quant_type)
            qB, S3 = F.vectorwise_quant(B, dim=dim_B, quant_type=quant_type)
            igrad_A = F.igemm(qgrad_output, qB.permute(permute_dim))
            grad_A = F.vectorwise_mm_dequant(igrad_A, S1, S3.permute(permute_dim), grad_output.dtype, quant_type)

            #if torch.rand(1) < 0.01:
            #    with torch.no_grad():
            #        grad_A2 = torch.matmul(grad_output, B.permute(permute_dim))
            #        err = torch.abs(grad_A-grad_A2)
            #        print(err.mean().item(), 'grad_A')
            #        del grad_A2, err

        return grad_A, grad_B, None, None


mm = MatMul8bit.apply
bmm = MatMul8bit.apply
matmul = MatMul8bit.apply

