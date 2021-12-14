import torch
import bitsandbytes.functional as F

class MatMul8bit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, out=None, quant_type='vector', precision=[8, 8, 8]):

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
                    qgrad_output, S1 = F.vectorwise_quant(grad_output.view(-1, grad_output.shape[2]), dim=0, quant_type=quant_type)
                    if not A.is_contiguous(): A = A.contiguous()
                    qA, S2 = F.vectorwise_quant(A.view(-1, A.shape[2]), dim=0, quant_type=quant_type)
                    igrad_B = F.igemm(qA.t(), qgrad_output)
                    grad_B = F.vectorwise_mm_dequant(igrad_B, S2.t(), S1, grad_output.dtype, quant_type)
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
                qB, S3 = F.vectorwise_quant(B, dim=dim_B, quant_type=quant_type)
                igrad_A = F.igemm(qgrad_output, qB.permute(permute_dim))
                grad_A = F.vectorwise_mm_dequant(igrad_A, S1, S3.permute(permute_dim), grad_output.dtype, quant_type)

        return grad_A, grad_B, None, None, None


mm = MatMul8bit.apply
bmm = MatMul8bit.apply
matmul = MatMul8bit.apply



class MLP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w1, w2):
        assert len(x.shape)==3

        #outa = torch.matmul(x, w1.t())
        # 16-bit: 0.136 / 0.25 -> for all -> 0.023 / 0.042 for one layer
        # raw: 0.0175
        x8, scale_x = F.vectorwise_quant(x, dim=[-1]) # +0.01
        w18, scale_w1 = F.vectorwise_quant(w1, dim=[-1]) # +0.01

        # -0.02 / -0.02 -> use empty instead of zero -> total: 0.02 / 0.03 (zeros) and  0.01 / 0.015 (empty)
        x8, Sx = F.transform(x8, 'col32') # +0.003 / +0.004
        w18, Sw1 = F.transform(w18, 'col_turing') # +0.005 / +0.008
        out1, Sout32 = F.get_transform_buffer((x.shape[0], x.shape[1], w1.shape[0]), torch.int32, x8.device, 'col32') # 0.002 / 0.007


        F.igemmlt(x8, w18, out1, Sx, Sw1, Sout32) # 0.012 / 0.023

        out132, Sout132 = F.transform(out1, state=Sout32, to_order='row') # 0.015 / 0.038
        out1h = F.vectorwise_mm_dequant(out132, scale_x, scale_w1.t()) # 0.051 / 0.033
        torch.relu_(out1h)

        w28, scale_w2 = F.vectorwise_quant(w2, dim=[-1]) # +0.02 / +0.04
        out18, scale_out18 = F.vectorwise_quant(out1h, dim=[-1])
        out18col, Sout18 = F.transform(out18, 'col32')
        w28, Sw2 = F.transform(w28, 'col_turing') # +0.01 / +0.013
        out2, Sout2 = F.get_transform_buffer((x.shape[0], x.shape[1], w2.shape[0]), torch.int32, x8.device, 'col32')

        F.igemmlt(out18col, w28, out2, Sout18, Sw2, Sout2)

        out232, Sout232 = F.transform(out2, state=Sout2, to_order='row')
        out2 = F.vectorwise_mm_dequant(out232, scale_out18, scale_w2.t())


        ctx.grad_input1 = x
        ctx.grad_input2 = out1h
        ctx.w1 = w1
        ctx.w2 = w2

        return x


    def backward(ctx, grad_output):
        #return x_grad, w1_grad, w2_grad
        seq, bsz = grad_output.shape[0], grad_output.shape[1]
        grad_input1 = ctx.grad_input1
        grad_input2 = ctx.grad_input2
        w1 = ctx.w1
        w2 = ctx.w2

        # weight grad = grad_out^t @ grad_in^t(^t)
        # [oh] = [obs, hbs^t]
        grad_out8, scale_grad_out = F.vectorwise_quant(grad_output.view(seq*bsz, -1).t().contiguous(), dim=[-1])
        grad_in8, scale_grad_in = F.vectorwise_quant(grad_input2.view(seq*bsz, -1).t().contiguous(), dim=[-1])
        grad_out8, Sgrad_out = F.transform(grad_out8, 'col32')
        grad_in8, Sgrad_in = F.transform(grad_in8, 'col_turing')
        w2grad, Sw2grad = F.get_transform_buffer(w2.shape, torch.int32, grad_output.device, 'col32')

        F.igemmlt(grad_out8, grad_in8, w2grad, Sgrad_out, Sgrad_in, Sw2grad)

        w2grad, Sw2grad = F.transform(w2grad, state=Sw2grad, to_order='row')
        w2grad = F.vectorwise_mm_dequant(w2grad, scale_grad_out, scale_grad_in.t())

        # generating output_grad2
        # grad_output2 = (grad_output @ w^t(^t))*relu_mask
        # [bsh] = [bso, ho^t]*[bsh]
        grad_out8, scale_grad_out = F.vectorwise_quant(grad_output, dim=[-1])
        w28, scale_w2 = F.vectorwise_quant(w2.t().contiguous(), dim=[-1])
        grad_out8, Sgrad_out = F.transform(grad_out8, 'col32')
        w28, Sw2 = F.transform(w28, 'col_turing')
        grad_out2, Sgrad_out2 = F.get_transform_buffer((seq, bsz, w1.shape[0]), torch.int32, grad_output.device, 'col32')

        F.igemmlt(grad_out8, w28, grad_out2, Sgrad_out, Sw2, Sgrad_out2)

        grad_out2, Sgrad_out2 = F.transform(grad_out2, state=Sgrad_out2, to_order='row')
        grad_output2 = F.vectorwise_mm_dequant(grad_out2, scale_grad_out, scale_w2.t())
        grad_output2.mul_((grad_input2>0).to(grad_out2.dtype))


        # weight grad = grad_out^t @ grad_in^t(^t)
        # [oh] = [obs, hbs^t] 
        grad_out8, scale_grad_out = F.vectorwise_quant(grad_output2.view(-1, grad_output2.shape[2]).t().contiguous(), dim=[-1])
        grad_in8, scale_grad_in = F.vectorwise_quant(grad_input1.view(-1, grad_input1.shape[2]).t().contiguous(), dim=[-1])
        grad_out8, Sgrad_out = F.transform(grad_out8, 'col32')
        grad_in8, Sgrad_in = F.transform(grad_in8, 'col_turing')
        w1grad, Sw1grad = F.get_transform_buffer(w1.shape, torch.int32, grad_output.device, 'col32')

        F.igemmlt(grad_out8, grad_in8, w1grad, Sgrad_out, Sgrad_in, Sw1grad)

        w1grad, Sw1grad = F.transform(w1grad, state=Sw1grad, to_order='row')
        w1grad = F.vectorwise_mm_dequant(w1grad, scale_grad_out, scale_grad_in.t())


        return None, w1grad, w2grad
