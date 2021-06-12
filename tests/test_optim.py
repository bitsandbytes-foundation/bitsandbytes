import os
import time
import shutil
import uuid
import pytest
import torch
import bitsandbytes as bnb

from os.path import join
from itertools import product


def get_temp_dir():
    path = '/tmp/autoswap/{0}'.format(str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    return path

def rm_path(path):
    shutil.rmtree(path)

dim1 = [1024]
dim2 = [32, 1024, 4097]
gtype = [torch.float32, torch.float16]
values = list(product(dim1,dim2, gtype))
names = ['dim1_{0}_dim2_{1}_gtype_{2}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, gtype", values, ids=names)
def test_adam32bit(dim1, dim2, gtype):
    if dim1 == 1 and dim2 == 1: return
    p1 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1
    p2 = p1.clone()
    p1 = p1.float()
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8


    adam1 = torch.optim.Adam([p1], lr, (beta1, beta2), eps)
    adam2 = bnb.optim.Adam([p2], lr, (beta1, beta2), eps)

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    else:
        atol, rtol = 1e-4, 1e-3


    for i in range(50):
        g = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.01
        p1.grad = g.float()
        p2.grad = g.clone()

        adam2.step()
        adam1.step()

        torch.testing.assert_allclose(adam1.state[p1]['exp_avg'], adam2.state[p2]['state1'], atol=atol, rtol=rtol)
        torch.testing.assert_allclose(adam1.state[p1]['exp_avg_sq'], adam2.state[p2]['state2'], atol=atol, rtol=rtol)

        if i % 10 == 0 and i > 0:
            path = get_temp_dir()
            torch.save(adam2.state_dict(),join(path, 'opt.pt'))
            adam2.load_state_dict(torch.load(join(path, 'opt.pt')))
            rm_path(path)
            torch.testing.assert_allclose(p1, p2.float(), atol=atol, rtol=rtol)
            torch.testing.assert_allclose(adam1.state[p1]['exp_avg'], adam2.state[p2]['state1'], atol=atol, rtol=rtol)
            torch.testing.assert_allclose(adam1.state[p1]['exp_avg_sq'], adam2.state[p2]['state2'], atol=atol, rtol=rtol)

        if gtype == torch.float16:
            # the adam buffers should also be close because they are 32-bit
            # but the paramters can diverge because they are 16-bit
            # the difference grow larger and larger with each update
            # --> copy the state to keep weights close
            p1.data = p1.data.half().float()
            p2.copy_(p1.data)
            torch.testing.assert_allclose(p1.half(), p2)

dim1 = [1024]
dim2 = [32, 1024, 4097]
gtype = [torch.float32, torch.float16]
values = list(product(dim1,dim2, gtype))
names = ['dim1_{0}_dim2_{1}_gtype_{2}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, gtype", values, ids=names)
def test_global_config(dim1, dim2, gtype):
    if dim1 == 1 and dim2 == 1: return
    p1 = torch.randn(dim1,dim2, device='cpu', dtype=gtype)*0.1
    p2 = torch.randn(dim1,dim2, device='cpu', dtype=gtype)*0.1
    mask = torch.rand_like(p2) < 0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8

    bnb.optim.GlobalOptimManager.get_instance().initialize()
    bnb.optim.GlobalOptimManager.get_instance().override_config(p2, 'is_sparse', True)

    bnb.optim.GlobalOptimManager.get_instance().register_parameters([p1, p2])
    p1 = p1.cuda()
    p2 = p2.cuda()

    adam2 = bnb.optim.Adam([p1, p2], lr, (beta1, beta2), eps)

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    else:
        atol, rtol = 1e-4, 1e-3

    original_p2 = p2[mask].clone()

    for i in range(50):
        g1 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1 + 0.001
        g2 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1 + 0.001
        p1.grad = g1
        p2.grad = g2

        if i > 30 and i % 10 == 0:
            g1.data[mask] = 0.0
            g2.data[mask] = 0.0
            p1.grad = g1
            p2.grad = g2
            original_p1 = p1[mask].clone()
            original_p2 = p2[mask].clone()
            og_s1 = adam2.state[p2]['state1'][mask].clone()
            og_s2 = adam2.state[p2]['state2'][mask].clone()
            og_s11 = adam2.state[p1]['state1'][mask].clone()
            og_s21 = adam2.state[p1]['state2'][mask].clone()

        adam2.step()

        if i > 30 and i % 10 == 0:
            torch.testing.assert_allclose(original_p2, p2[mask])
            torch.testing.assert_allclose(adam2.state[p2]['state1'][mask], og_s1)
            torch.testing.assert_allclose(adam2.state[p2]['state2'][mask], og_s2)
            assert ((p1[mask]- original_p1)==0.0).sum() < p1.numel()
            assert ((adam2.state[p1]['state1'][mask]- og_s11)==0.0).sum() == 0.0
            assert ((adam2.state[p1]['state2'][mask]- og_s21)==0.0).sum() == 0.0

