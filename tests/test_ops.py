import torch

from bitsandbytes import ops

def setup():
    pass

def teardown():
    pass

def test_scalar():
    a = torch.rand(10,10).cuda()
    out = torch.rand(10,10).cuda()
    ops.testmul(a, 3.0, out)
    torch.testing.assert_allclose(a*3.0, out)
