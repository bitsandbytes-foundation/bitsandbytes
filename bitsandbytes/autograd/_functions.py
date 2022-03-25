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

class MatMul8bitLt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, out=None, CB=None, return_CB=False):
        formatB = F.get_special_format_str()
        input_shape = A.shape

        if len(A.shape) == 3:
            output_shape = (A.shape[0], A.shape[1], B.shape[-1])
            A = A.view(-1, A.shape[-1]).contiguous()
        else:
            output_shape = (A.shape[0], B.shape[-1])


        CA, CAt, SCA, SCAt, coo_tensor = F.double_quant(A)
        C32A, SA = F.transform(CA, 'col32')

        if CB is not None:
            CxB, SB, SCB = CB
            CBt = SCBt = None
        else:
            CB, CBt, SCB, SCBt, coo_tensor = F.double_quant(B.t())
            CxB, SB = F.transform(CB, to_order=formatB)

        out32, Sout32 = F.igemmlt(C32A, CxB, SA, SB)
        output = F.mm_dequant(out32, Sout32, SCA, SCB)

        if A.requires_grad or B.requires_grad:
            ctx.tensors = (A, B, CAt, CBt)
            ctx.tensor_states = (SCAt, SCBt)
            ctx.formatB = formatB
            ctx.grad_shape = input_shape
        else:
            ctx.save_for_backward(None, None)

        if return_CB:
            return (output.view(output_shape).clone(), CxB, SCB)
        else:
            return output.view(output_shape).clone()

    @staticmethod
    def backward(ctx, grad_output, CxB=None, SCB=None):
        A, B, CAt, CBt = ctx.tensors
        SCAt, SCBt = ctx.tensor_states
        formatB = ctx.formatB

        if len(grad_output.shape) == 3:
            grad_output = grad_output.view(-1, grad_output.shape[-1]).contiguous()

        grad_A = grad_B = None

        Cgrad, Cgradt, SCgrad, SCgradt, coo_tensor = F.double_quant(grad_output)
        if B.requires_grad:
            CxAt, SAt = F.transform(CAt, formatB, transpose=True)
            C32grad, Sgrad = F.transform(Cgradt, 'col32', transpose=True)
            gradB32, SgradB32 = F.igemmlt(C32grad, CxAt, Sgrad, SAt)
            grad_B = F.mm_dequant(gradB32, SgradB32, SCgradt, SCAt).t()

        if A.requires_grad:
            C32grad, Sgrad = F.transform(Cgrad, 'col32')
            CxBt, SBt = F.transform(CBt, to_order=formatB, transpose=True)
            gradA32, SgradA32 = F.igemmlt(C32grad, CxBt, Sgrad, SBt)
            grad_A = F.mm_dequant(gradA32, SgradA32, SCgrad, SCBt).view(ctx.grad_shape)

        return grad_A, grad_B, None, None, None, None


mmlt = MatMul8bitLt.apply
matmullt = MatMul8bitLt.apply
