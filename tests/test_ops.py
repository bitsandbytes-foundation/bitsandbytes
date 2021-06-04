import torch

from bitsandbytes import cluster_net as gpu


def setup():
    pass

def teardown():
    pass

def test_scalar():
    a = torch.rand(10,10).cuda()
    out = torch.rand(10,10).cuda()
    gpu.testmul(a, 3.0, out)
    torch.testing.assert_allclose(a*3.0, out)
