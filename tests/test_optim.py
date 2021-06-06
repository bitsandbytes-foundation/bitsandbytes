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




