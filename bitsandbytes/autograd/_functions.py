import torch
import bitsandbytes.functional as F

tensor = torch.Tensor

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


mm_cublas = MatMul8bit.apply
bmm_cublas = MatMul8bit.apply
matmul_cublas = MatMul8bit.apply

class MatMul8bitLt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, out=None, decomp_threshold=0.0, CB=None, return_CB=False):
        requires_gradA = A.requires_grad
        requires_gradB = B.requires_grad
        formatB = F.get_special_format_str()
        input_shape = A.shape

        if len(A.shape) == 3:
            output_shape = (A.shape[0], A.shape[1], B.shape[0])
            A = A.view(-1, A.shape[-1]).contiguous()
        else:
            output_shape = (A.shape[0], B.shape[0])

        CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(A, threshold=decomp_threshold)
        idx = None
        if decomp_threshold > 0.0 and coo_tensorA is not None:
            idx = torch.unique(coo_tensorA.colidx).long()
            CA[:, idx] = 0
            CAt[:, idx] = 0
            subA = A[:, idx]
        else:
            subA = None

        C32A, SA = F.transform(CA, 'col32')

        is_transposed = not B.is_contiguous() and B.shape[0] == B.stride(1)
        if is_transposed: B = B.contiguous()

        if CB is not None:
            CxB, SB, SCB, subB = CB
            CBt = SCBt = None
        else:
            CB, CBt, SCB, SCBt, coo_tensorB = F.double_quant(B)
            CxB, SB = F.transform(CB, to_order=formatB)
            subB = None

        out32, Sout32 = F.igemmlt(C32A, CxB, SA, SB)
        output = F.mm_dequant(out32, Sout32, SCA, SCB)

        if decomp_threshold > 0.0 and coo_tensorA is not None:
            output += torch.matmul(subA, B[:, idx].t())

        ctx.formatB = formatB
        ctx.grad_shape = input_shape
        ctx.req_grads = [requires_gradA, requires_gradB]
        ctx.decomp_threshold = decomp_threshold

        if requires_gradA or requires_gradB:
            ctx.tensors = (CAt, CBt, subA)
            ctx.tensor_states = (SCAt, SCBt, idx)
        else:
            ctx.tensors = [None, None]
            ctx.tensor_states = (None, None)
            ctx.save_for_backward(None, None)

        clone_func = torch.clone if len(output_shape) == 3 else lambda x : x
        if return_CB:
            return (clone_func(output.view(output_shape)), CxB, SCB, subB or torch.empty(1))
        else:
            return clone_func(output.view(output_shape))

    @staticmethod
    def backward(ctx, grad_output, CxB=None, SCB=None, subB=None):
        req_gradA, req_gradB = ctx.req_grads
        CAt, CBt, subA = ctx.tensors
        SCAt, SCBt, idx = ctx.tensor_states
        formatB = ctx.formatB
        decomp_threshold = ctx.decomp_threshold

        if len(grad_output.shape) == 3:
            grad_output = grad_output.view(-1, grad_output.shape[-1]).contiguous()

        grad_A = grad_B = None

        Cgrad, Cgradt, SCgrad, SCgradt, coo_tensor = F.double_quant(grad_output)
        if req_gradB:
            CxAt, SAt = F.transform(CAt, formatB, transpose=True)
            C32grad, Sgrad = F.transform(Cgradt, 'col32', transpose=True)
            gradB32, SgradB32 = F.igemmlt(C32grad, CxAt, Sgrad, SAt)
            grad_B = F.mm_dequant(gradB32, SgradB32, SCgradt, SCAt)
            if decomp_threshold > 0.0 and subA is not None:
                grad_B[:, idx] += torch.matmul(grad_output.t(), subA)

        if req_gradA:
            C32grad, Sgrad = F.transform(Cgrad, 'col32')
            CxBt, SBt = F.transform(CBt, to_order=formatB, transpose=True)
            gradA32, SgradA32 = F.igemmlt(C32grad, CxBt, Sgrad, SBt)
            grad_A = F.mm_dequant(gradA32, SgradA32, SCgrad, SCBt).view(ctx.grad_shape)


        return grad_A, grad_B, None, None, None, None, None


matmul = MatMul8bitLt.apply


def matmul(A : tensor, B : tensor, out : tensor=None, threshold : int=0.0, CB : tensor=None, return_CB : bool=False):
    return MatMul8bitLt.apply(A, B, out, threshold, CB, return_CB)

