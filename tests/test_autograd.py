import pytest

import torch
import bitsandbytes as bnb

from itertools import product

n = 1
k = 25
dim1 = torch.randint(1,64, size=(n,)).tolist()
dim2 = torch.randint(32,128, size=(n,)).tolist()
dim3 = torch.randint(32,256, size=(n,)).tolist()
dim4 = torch.randint(32,256, size=(n,)).tolist()
funcs = [(torch.mm, bnb.mm), (torch.bmm, bnb.bmm), (torch.matmul, bnb.matmul)]
str_funcs = ['mm', 'bmm', 'matmul']
req_grad = [(False, False), (True, False), (True, True), (False, True)]
req_grad_str = ['FF', 'TF', 'TT', 'FT']
dtype = [torch.float32, torch.float16]
values = list(product(dim1,dim2,dim3,dim4,funcs, dtype, req_grad))
str_values = list(product(dim1,dim2,dim3,dim4,str_funcs, dtype, req_grad_str))
names = ['dim1_{0}_dim2_{1}_dim3_{2}_dim4_{3}_func_{4}_dtype_{5}_requires_grad_{6}'.format(*vals) for vals in str_values]
@pytest.mark.parametrize("dim1, dim2, dim3, dim4, funcs, dtype, req_grad", values, ids=names)
def test_matmul(dim1, dim2, dim3, dim4, funcs, dtype, req_grad):
    dim2 = dim2 - (dim2 % 16)
    dim3 = dim3 - (dim3 % 16)
    dim4 = dim4 - (dim4 % 16)
    for i in range(k):

        # normal multiply
        if funcs[0] in [torch.mm, torch.matmul]:
            A = torch.randn(size=(dim2, dim3), device='cuda', requires_grad=req_grad[0])
            B = torch.randn(size=(dim3, dim4), device='cuda', requires_grad=req_grad[1])
            target = torch.randn(size=(dim2, dim4), device='cuda', requires_grad=req_grad[1])
            torch.nn.init.xavier_uniform_(B)

            out_torch = funcs[0](A, B)
            out_bnb = funcs[1](A, B)

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx==0).sum().item() < n*0.0175
            idx = torch.isclose(out_bnb, out_torch, atol=0.035, rtol=0.2)
            assert (idx==0).sum().item() < n*0.001

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_allclose(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx==0).sum().item() < n*0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx==0).sum().item() < n*0.02
                torch.testing.assert_allclose(gradB1, gradB2, atol=0.18, rtol=0.3)

        # batched matrix multiply
        if funcs[0] in [torch.bmm, torch.matmul]:
            A = torch.randn(size=(dim1, dim2, dim3), device='cuda', requires_grad=req_grad[0])
            B = torch.randn(size=(dim1, dim3, dim4), device='cuda', requires_grad=req_grad[1])
            target = torch.randn(size=(dim1, dim2, dim4), device='cuda', requires_grad=req_grad[1])
            torch.nn.init.xavier_uniform_(B)

            out_torch = funcs[0](A, B)
            out_bnb = funcs[1](A, B)

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx==0).sum().item() < n*0.01
            torch.testing.assert_allclose(out_bnb, out_torch, atol=0.027, rtol=0.2)

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_allclose(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx==0).sum().item() < n*0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx==0).sum().item() < n*0.02

        if funcs[0] in [torch.matmul]:
            A = torch.randn(size=(dim1, dim2, dim3), device='cuda', requires_grad=req_grad[0])
            B = torch.randn(size=(dim3, dim4), device='cuda', requires_grad=req_grad[1])
            target = torch.randn(size=(dim1, dim2, dim4), device='cuda', requires_grad=req_grad[1])
            torch.nn.init.xavier_uniform_(B)

            out_torch = funcs[0](A, B)
            out_bnb = funcs[1](A, B)

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx==0).sum().item() < n*0.0175
            idx = torch.isclose(out_bnb, out_torch, atol=0.035, rtol=0.2)
            assert (idx==0).sum().item() < n*0.001

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_allclose(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx==0).sum().item() < n*0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx==0).sum().item() < n*0.02
