import pytest
import time
import torch

from itertools import product

from bitsandbytes import functional as F

class Timer(object):
    def __init__(self):
        self.starts = {}
        self.ends = {}
        self.agg = {}

    def tick(self, name='default'):
        if name not in self.starts:
            self.starts[name] = torch.cuda.Event(enable_timing=True)
            self.ends[name] = torch.cuda.Event(enable_timing=True)
            self.starts[name].record()
        else:
            ms = self.tock(name, evict=True, print_ms=False)

    def tock(self, name='default', evict=True, print_ms=True):
        if name in self.ends:
            self.ends[name].record()
            torch.cuda.synchronize()
            ms = self.starts[name].elapsed_time(self.ends[name])
            if name not in self.agg: self.agg[name] = 0.0
            self.agg[name] += ms
            if evict:
                self.starts.pop(name)
                self.ends.pop(name)

        if print_ms and name in self.agg:
            print('{0} took: {1:.5f}s'.format(name, self.agg[name]/1000.0))

        return self.agg[name]

    def reset(self):
        self.starts  = {}
        self.ends = {}
        self.agg = {}
        print('Resetting benchmark data')

def setup():
    pass

def teardown():
    pass

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=['float', 'half'])
def test_estimate_quantiles(dtype):
    A = torch.rand(1024, 1024, device='cuda')
    A = A.to(dtype)
    code = F.estimate_quantiles(A)

    percs = torch.linspace(1/512, 511/512, 256, device=A.device)
    torch.testing.assert_allclose(percs, code, atol=1e-3, rtol=1e-2)

    A = torch.randn(1024, 1024, device='cuda')
    A = A.to(dtype)
    code = F.estimate_quantiles(A)

    quantiles = torch.quantile(A.float(), percs)
    diff = torch.abs(code-quantiles)
    assert (diff > 5e-02).sum().item() == 0


def test_quantile_quantization():
    for i in range(100):
        A1 = torch.randn(1024, 1024, device='cuda')
        code = F.estimate_quantiles(A1)
        C = F.quantize_no_absmax(code, A1)
        A2 = F.dequantize_no_absmax(code, C)
        diff = torch.abs(A1-A2).mean().item()
        assert diff < 0.0075

        A1 = torch.rand(1024, 1024, device='cuda')
        code = F.estimate_quantiles(A1)
        C = F.quantize_no_absmax(code, A1)
        A2 = F.dequantize_no_absmax(code, C)
        diff = torch.abs(A1-A2).mean().item()
        torch.testing.assert_allclose(A1, A2, atol=5e-3, rtol=0)
        assert diff < 0.001


def test_dynamic_quantization():
    diffs = []
    reldiffs = []
    for i in range(100):
        A1 = torch.randn(1024, 1024, device='cuda')
        absmax, C = F.quantize(A1)
        A2 = F.dequantize(absmax, C)
        diff = torch.abs(A1-A2)
        reldiff = diff/torch.abs(A1+1e-8)
        diffs.append(diff.mean().item())
        reldiffs.append(reldiff.mean().item())
        assert diff.mean().item() < 0.0135
    #print(sum(diffs)/len(diffs))
    #print(sum(reldiffs)/len(reldiffs))

    for i in range(100):
        A1 = torch.rand(1024, 1024, device='cuda')
        absmax, C = F.quantize(A1)
        A2 = F.dequantize(absmax, C)
        diff = torch.abs(A1-A2).mean().item()
        torch.testing.assert_allclose(A1, A2, atol=1e-2, rtol=0)
        assert diff < 0.004


def test_dynamic_blockwise_quantization():
    diffs = []
    reldiffs = []
    for i in range(100):
        A1 = torch.randn(1024, 1024, device='cuda')
        absmax, C = F.quantize_blockwise(A1)
        A2 = F.dequantize_blockwise(absmax, C)
        diff = torch.abs(A1-A2)
        reldiff = diff/torch.abs(A1+1e-8)
        diffs.append(diff.mean().item())
        reldiffs.append(reldiff.mean().item())
        assert diffs[-1] < 0.011
    #print(sum(diffs)/len(diffs))
    #print(sum(reldiffs)/len(reldiffs))

    diffs = []
    for i in range(100):
        A1 = torch.rand(1024, 1024, device='cuda')
        absmax, C = F.quantize_blockwise(A1)
        A2 = F.dequantize_blockwise(absmax, C)
        diff = torch.abs(A1-A2).mean().item()
        assert diff < 0.0033
        diffs.append(diff)
        torch.testing.assert_allclose(A1, A2, atol=1e-2, rtol=0)
    #print(sum(diffs)/len(diffs))

def test_dynamic_blockwise_stochastic_quantization():
    diffs = []
    reldiffs = []
    rand = torch.rand(1024).cuda()
    for i in range(100):
        A1 = torch.randn(1024, 1024, device='cuda')
        absmax, C1 = F.quantize_blockwise(A1, rand=rand)
        absmax, C2 = F.quantize_blockwise(A1)
        # a maximunm distance of quantized values of 1
        torch.testing.assert_allclose(C1, C2, atol=1, rtol=0)
        fraction_smaller = (C1<C2).float().sum()/C1.numel()
        fraction_larger = (C1>C2).float().sum()/C1.numel()
        torch.testing.assert_allclose(fraction_larger, fraction_smaller, atol=0.01, rtol=0)



@pytest.mark.parametrize("gtype", [torch.float32, torch.float16], ids=['float', 'half'])
def test_percentile_clipping(gtype):
    gnorm_vec1 = torch.zeros(100, device='cuda')
    gnorm_vec2 = torch.zeros(100, device='cuda')
    n = 4
    step = 0
    percentile=5
    for i in range(1000):
        step += 1
        g = torch.randn(n, n, dtype=gtype, device='cuda')
        gnorm1, clip2, gnorm_scale = F.percentile_clipping(g, gnorm_vec2, step, percentile=percentile)
        assert gnorm_scale == 1.0 if gnorm1 < clip2 else clip2/gnorm1

        gnorm2 = torch.norm(g.float())
        if step == 1:
            gnorm_vec1[:] = gnorm2
        else:
            gnorm_vec1[step % 100] = gnorm2

        vals, idx = torch.sort(gnorm_vec1)
        clip1 = vals[percentile]

        torch.testing.assert_allclose(gnorm_vec1, torch.sqrt(gnorm_vec2))
        torch.testing.assert_allclose(clip1, clip2)
        torch.testing.assert_allclose(gnorm1, gnorm2)


dim1 = torch.randint(1,1024*4, size=(4,)).tolist()
dim2 = torch.randint(1,1024*4, size=(4,)).tolist()
values = list(product(dim1,dim2))
names = ['dim1_{0}_dim2_{1}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2", values, ids=names)
def test_igemm(dim1, dim2):
    dim1 = dim1 - (dim1 % 64)
    dim2 = dim2 - (dim2 % 64)
    for i in range(100):
        A = torch.randint(-128, 127, size=(dim1, dim2), device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=(dim2, dim1), device='cuda').to(torch.int8)
        #A = torch.arange(16*16, device='cuda').view(32, 8).to(torch.int8).contiguous()
        #B = torch.arange(16*16, device='cuda').view(8, 32).to(torch.int8).contiguous()
        out = F.mmi(A, B)
        out2 = torch.mm(A.float(), B.float())
        torch.testing.assert_allclose(out.float(), out2)


def test_igemm_bench():
    dim1 = 1024*20
    dim2 = 1024*20
    dim3 = 1024*20
    A = torch.randint(-128, 127, size=(dim1, dim2), device='cuda').to(torch.int8)
    B = torch.randint(-128, 127, size=(dim2, dim3), device='cuda').to(torch.int8)
    C = torch.zeros(dim1, dim3, device=A.device, dtype=torch.int32)

    t = Timer()
    A = torch.randn(dim1, dim2, device='cuda')
    B = torch.randn(dim2, dim3, device='cuda')
    A = A.half()
    B = B.half()
    C = torch.zeros(dim1, dim3, device=A.device, dtype=B.dtype)



    for i in range(25):
        if i == 5:
            t.reset()
            t.tick('total time')

        #F.mmi(A, B, out=C)
        torch.mm(A, B, out=C)

    t.tock('total time')

