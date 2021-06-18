import os
import time
import shutil
import uuid
import pytest
import ctypes
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F

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
            del adam2
            adam2 = None
            adam2 = bnb.optim.Adam([p2])
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
    p3 = torch.randn(dim1,dim2, device='cpu', dtype=gtype)*0.1
    mask = torch.rand_like(p2) < 0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8

    bnb.optim.GlobalOptimManager.get_instance().initialize()
    bnb.optim.GlobalOptimManager.get_instance().override_config(p2, 'is_sparse', True)
    bnb.optim.GlobalOptimManager.get_instance().override_config(p3, 'optim_bits', 8)

    bnb.optim.GlobalOptimManager.get_instance().register_parameters([p1, p2, p3])
    p1 = p1.cuda()
    p2 = p2.cuda()
    p3 = p3.cuda()

    adam2 = bnb.optim.Adam([p1, p2, p3], lr, (beta1, beta2), eps)

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    else:
        atol, rtol = 1e-4, 1e-3

    original_p2 = p2[mask].clone()

    for i in range(50):
        g1 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1 + 0.001
        g2 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1 + 0.001
        g3 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1 + 0.001
        p1.grad = g1
        p2.grad = g2
        p3.grad = g3

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

        assert adam2.state[p3]['state1'].dtype == torch.uint8
        assert adam2.state[p3]['state2'].dtype == torch.uint8

        if i > 30 and i % 10 == 0:
            torch.testing.assert_allclose(original_p2, p2[mask])
            torch.testing.assert_allclose(adam2.state[p2]['state1'][mask], og_s1)
            torch.testing.assert_allclose(adam2.state[p2]['state2'][mask], og_s2)
            assert ((p1[mask]- original_p1)==0.0).sum() < p1.numel()
            assert ((adam2.state[p1]['state1'][mask]- og_s11)==0.0).sum() == 0.0
            assert ((adam2.state[p1]['state2'][mask]- og_s21)==0.0).sum() == 0.0



dim1 = [1024]
dim2 = [32, 1024, 4097]
gtype = [torch.float32, torch.float16]
values = list(product(dim1,dim2, gtype))
names = ['dim1_{0}_dim2_{1}_gtype_{2}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, gtype", values, ids=names)
def test_adam8bit(dim1, dim2, gtype):
    if dim1 == 1 and dim2 == 1: return
    p1 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1
    p2 = p1.clone()
    p1 = p1.float()
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8


    adam1 = torch.optim.Adam([p1], lr, (beta1, beta2), eps)
    adam2 = bnb.optim.Adam8bit([p2], lr, (beta1, beta2), eps)

    if gtype == torch.float32:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3

    else:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3


    for i in range(50):
        g = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.01
        p1.grad = g.float()
        p2.grad = g.clone()

        adam2.step()
        adam1.step()

        s1 = F.dequantize_with_absmax(adam2.state[p2]['qmap1'], adam2.state[p2]['max1'], adam2.state[p2]['state1'])
        s2 = F.dequantize_with_absmax(adam2.state[p2]['qmap2'], adam2.state[p2]['max2'], adam2.state[p2]['state2'])
        torch.testing.assert_allclose(adam1.state[p1]['exp_avg'], s1, atol=atol, rtol=rtol)
        torch.testing.assert_allclose(adam1.state[p1]['exp_avg_sq'], s2, atol=atol, rtol=rtol)
        torch.testing.assert_allclose(p1, p2.float(), atol=patol, rtol=prtol)

        err  = torch.abs(p1-p2)
        relerr = err/torch.abs(p1)
        assert err.mean() < 0.0001
        assert relerr.mean() < 0.001

        if i % 10 == 0 and i > 0:
            s1cpy = s1.clone()
            s2cpy = s2.clone()
            raws1cpy = adam2.state[p2]['state1'].clone()
            raws2cpy = adam2.state[p2]['state2'].clone()
            qmap1 = adam2.state[p2]['qmap1'].clone()
            qmap2 = adam2.state[p2]['qmap2'].clone()

            path = get_temp_dir()
            torch.save(adam2.state_dict(),join(path, 'opt.pt'))
            del adam2
            adam2 = None
            adam2 = bnb.optim.Adam8bit([p2])
            adam2.load_state_dict(torch.load(join(path, 'opt.pt')))
            rm_path(path)
            torch.testing.assert_allclose(raws1cpy, adam2.state[p2]['state1'])
            torch.testing.assert_allclose(raws2cpy, adam2.state[p2]['state2'])
            torch.testing.assert_allclose(qmap1, adam2.state[p2]['qmap1'])
            torch.testing.assert_allclose(qmap2, adam2.state[p2]['qmap2'])


            s1 = F.dequantize_with_absmax(adam2.state[p2]['qmap1'], adam2.state[p2]['max1'], adam2.state[p2]['state1'])
            s2 = F.dequantize_with_absmax(adam2.state[p2]['qmap2'], adam2.state[p2]['max2'], adam2.state[p2]['state2'])

            torch.testing.assert_allclose(s1cpy, s1)
            torch.testing.assert_allclose(s2cpy, s2)

            torch.testing.assert_allclose(adam1.state[p1]['exp_avg'], s1, atol=atol, rtol=rtol)
            torch.testing.assert_allclose(adam1.state[p1]['exp_avg_sq'], s2, atol=atol, rtol=rtol)
            torch.testing.assert_allclose(p1, p2.float(), atol=patol, rtol=prtol)

        # the parameters diverge quickly. Here we keep them close
        # together so we can test against the Adam error
        p1.data = p1.data.to(gtype).float()
        p2.copy_(p1.data)
        adam1.state[p1]['exp_avg'].copy_(s1.data)
        adam1.state[p1]['exp_avg_sq'].copy_(s2.data)
        torch.testing.assert_allclose(p1.to(gtype), p2)



dim1 = [1024]
dim2 = [32, 1024, 4097]
gtype = [torch.float32]
optim_bits = [32, 8]
values = list(product(dim1,dim2, gtype, optim_bits))
names = ['dim1_{0}_dim2_{1}_gtype_{2}_optim_bits_{3}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, gtype, optim_bits", values, ids=names)
def test_adam_percentile_clipping(dim1, dim2, gtype, optim_bits):
    if dim1 == 1 and dim2 == 1: return
    p1 = torch.randn(dim1,dim2, device='cpu', dtype=gtype)*0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8
    p1 = p1.cuda()
    p2 = p1.clone()
    adam1 = bnb.optim.Adam([p1], lr, (beta1, beta2), eps, optim_bits=optim_bits)
    adam2 = bnb.optim.Adam([p2], lr, (beta1, beta2), eps, optim_bits=optim_bits, percentile_clipping=5)

    gnorm_vec = torch.zeros(100).cuda()
    step = 0

    for i in range(50):
        step += 1
        g1 = torch.randn(dim1,dim2, device='cuda', dtype=gtype)*0.1 + (0.01*i)
        g2 = g1.clone()
        p2.grad = g2

        current_gnorm, clip_val, gnorm_scale = F.percentile_clipping(g1, gnorm_vec, step, 5)
        g1 = (g1.float()*gnorm_scale).to(gtype)
        p1.grad = g1

        adam1.step()
        adam2.step()

        # gnorm_scale is not deterministic (warp reductions), as such there can be slight differences in state
        if optim_bits == 32:
            torch.testing.assert_allclose(p1, p2)
            torch.testing.assert_allclose(adam1.state[p1]['state1'], adam2.state[p2]['state1'], atol=5e-5, rtol=1e-4)
            torch.testing.assert_allclose(adam1.state[p1]['state2'], adam2.state[p2]['state2'], atol=5e-5, rtol=1e-4)
        elif optim_bits == 8:
            torch.testing.assert_allclose(p1, p2, atol=1e-4, rtol=1e-3)
            torch.testing.assert_allclose(adam1.state[p1]['state1'], adam2.state[p2]['state1'], atol=2, rtol=1e-3)
            torch.testing.assert_allclose(adam1.state[p1]['state2'], adam2.state[p2]['state2'], atol=2, rtol=1e-3)
            adam1.state[p1]['state1'].copy_(adam2.state[p2]['state1'])
            adam1.state[p1]['state2'].copy_(adam2.state[p2]['state2'])
        if i % 10 == 0 and i > 0:
            path = get_temp_dir()
            torch.save(adam2.state_dict(),join(path, 'opt.pt'))
            del adam2
            adam2 = None
            adam2 = bnb.optim.Adam([p2], lr, (beta1, beta2), eps, optim_bits=optim_bits, percentile_clipping=5)
            adam2.load_state_dict(torch.load(join(path, 'opt.pt')))


