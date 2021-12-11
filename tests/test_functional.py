import pytest
import random
import time
import torch
import bitsandbytes as bnb

from itertools import product

from bitsandbytes import functional as F

torch.set_printoptions(precision=4, sci_mode=False)

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
        C = F.quantize_no_absmax(A1, code)
        A2 = F.dequantize_no_absmax(C, code)
        diff = torch.abs(A1-A2).mean().item()
        assert diff < 0.0075

        A1 = torch.rand(1024, 1024, device='cuda')
        code = F.estimate_quantiles(A1)
        C = F.quantize_no_absmax(A1, code)
        A2 = F.dequantize_no_absmax(C, code)
        diff = torch.abs(A1-A2).mean().item()
        torch.testing.assert_allclose(A1, A2, atol=5e-3, rtol=0)
        assert diff < 0.001


def test_dynamic_quantization():
    diffs = []
    reldiffs = []
    for i in range(100):
        A1 = torch.randn(1024, 1024, device='cuda')
        C, S = F.quantize(A1)
        A2 = F.dequantize(C, S)
        diff = torch.abs(A1-A2)
        reldiff = diff/torch.abs(A1+1e-8)
        diffs.append(diff.mean().item())
        reldiffs.append(reldiff.mean().item())
        assert diff.mean().item() < 0.0135
    #print(sum(diffs)/len(diffs))
    #print(sum(reldiffs)/len(reldiffs))

    for i in range(100):
        A1 = torch.rand(1024, 1024, device='cuda')
        C, S = F.quantize(A1)
        A2 = F.dequantize(C, S)
        diff = torch.abs(A1-A2).mean().item()
        torch.testing.assert_allclose(A1, A2, atol=1e-2, rtol=0)
        assert diff < 0.004


def test_dynamic_blockwise_quantization():
    diffs = []
    reldiffs = []
    for i in range(100):
        A1 = torch.randn(1024, 1024, device='cuda')
        C, S = F.quantize_blockwise(A1)
        A2 = F.dequantize_blockwise(C, S)
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
        C, S = F.quantize_blockwise(A1)
        A2 = F.dequantize_blockwise(C, S)
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
        C1, S1 = F.quantize_blockwise(A1, rand=rand)
        C2, S2 = F.quantize_blockwise(A1)
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


dim1 = torch.randint(32,1024*4, size=(4,)).tolist()
dim2 = torch.randint(32,1024*4, size=(4,)).tolist()
values = list(product(dim1,dim2))
names = ['dim1_{0}_dim2_{1}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2", values, ids=names)
def test_igemm(dim1, dim2):
    dim1 = dim1 - (dim1 % 32)
    dim2 = dim2 - (dim2 % 32)
    for i in range(100):
        A = torch.randint(-128, 127, size=(dim1, dim2), device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=(dim2, dim1), device='cuda').to(torch.int8)
        #A = torch.arange(16*16, device='cuda').view(32, 8).to(torch.int8).contiguous()
        #B = torch.arange(16*16, device='cuda').view(8, 32).to(torch.int8).contiguous()
        out = F.igemm(A, B)
        out2 = torch.mm(A.float(), B.float())
        torch.testing.assert_allclose(out.float(), out2)


def quant(x):
    max1 = torch.abs(x).max()
    x = torch.round(x/max1*127)
    return max1, x.to(torch.int8)

def dequant(c, maxC):
    return c.float()*(maxC/127)

def mm_dequant(maxA, maxB, C):
    return C.float()*(maxA/127)*(maxB/127)

def quant_multi(x, dim):
    max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
    max1[max1==0] = 1.0
    x = torch.round(x/max1*127)
    return max1, x.to(torch.int8)

def quant_multi_chunk(x, dim, chunk_size=32):
    if dim==1:
        x_chunked = einops.rearrange(x, '(c a) b -> c a b', c=chunk_size)
        max1 = torch.amax(torch.abs(x_chunked), dim=dim+1, keepdim=True)
        max1 = torch.tile(max1, (1, 1, x.shape[1]))
        max1 = max1.view(x.shape)
    elif dim==0:
        x_chunked = einops.rearrange(x, 'a (b c) -> a b c', c=chunk_size)
        max1 = torch.amax(torch.abs(x_chunked), dim=dim, keepdim=True)
        max1 = torch.tile(max1, (x.shape[0], 1, 1))
        max1 = max1.view(x.shape)
    max1[max1==0] = 1.0
    x = torch.round(x/max1*127)
    return max1, x.to(torch.int8)

def quant_minmax(A):
    minA = A.min()
    maxA = A.max()

def mean(xx):
    return sum(xx)/float(len(xx))

#dim1 = torch.randint(1,1024*4, size=(4,)).tolist()
#dim2 = torch.randint(1,1024*4, size=(4,)).tolist()
dim1 = [1024*2]
dim2 = [1024*16]
methods = [(lambda x, dim: quant(x), lambda x, dim: quant(x), dequant, dequant, mm_dequant)]
methods.append((quant_multi, quant_multi, dequant, dequant, mm_dequant))
#methods.append((lambda x: quant_multi_chunk(x, dim=-1), lambda x: quant_multi_chunk(x, dim=0), dequant, dequant, mm_dequant))
method_names = ['linear', 'vectorwise']
batched = [False, True]
values = list(product(dim1,dim2, methods, batched))
values_names = list(product(dim1,dim2, method_names, batched))
names = ['dim1_{0}_dim2_{1}_quant_{2}_batched_{3}'.format(*vals) for vals in values_names]
@pytest.mark.parametrize("dim1, dim2, quant_methods, batched", values, ids=names)
def test_approx_igemm(dim1, dim2, quant_methods, batched):
    dim1 = dim1 - (dim1 % 32)
    dim2 = dim2 - (dim2 % 32)
    errors = []
    relerrors = []
    print('')
    for i in range(5):
        if batched:
            A = torch.normal(0, 0.5, size=(32, dim1, dim2//32), device='cuda')
            B = torch.normal(0, 0.5, size=(32, dim2//32, dim1), device='cuda')
            maxA, Ac = quant_methods[0](A, 2)
            maxB, Bc = quant_methods[1](B, 1)
        else:
            A = torch.normal(0, 0.5, size=(dim1, dim2), device='cuda')
            B = torch.normal(0, 0.5, size=(dim2, dim1), device='cuda')
            maxA, Ac = quant_methods[0](A, 1)
            maxB, Bc = quant_methods[1](B, 0)
        torch.testing.assert_allclose(quant_methods[2](maxA, Ac), A, atol=0.025, rtol=0.05)
        if batched:
            out2 = torch.bmm(A, B)
            C = torch.bmm(Ac.float(), Bc.float())
        else:
            out2 = torch.mm(A, B)
            C = F.igemm(Ac, Bc)
        out = quant_methods[4](maxA, maxB, C)
        std = out2.std()
        out/= std
        out2/= std
        err = torch.abs(out-out2)
        relerr = err/torch.abs(out2)
        errors.append(err.mean().item())
        relerrors.append(relerr.mean().item())
    print(mean(errors))
    print(mean(relerrors))





def test_igemm_bench():
    dim1 = 4096*1
    dim2 = 4096*1
    dim3 = 4096*1
    A = torch.randint(-128, 127, size=(dim1, dim2), device='cuda').to(torch.int8)
    B = torch.randint(-128, 127, size=(dim2, dim3), device='cuda').to(torch.int8)
    C = torch.zeros(dim1, dim3, device=A.device, dtype=torch.int32)

    t = Timer()
    A = torch.randn(dim1, dim2, device='cuda')
    B = torch.randn(dim2, dim3, device='cuda')
    A = A.half()
    B = B.half()
    C = torch.zeros(dim1, dim3, device=A.device, dtype=B.dtype)


    for i in range(128):
        #F.mmi(A, B, out=C)
        #torch.mm(A, B, out=C)
        F.cutlass_igemm(A, B, out=C)

def test_stable_embedding():
    layer = bnb.nn.StableEmbedding(1024, 1024)
    layer.reset_parameters()



n = 3
k = 100
hidden_dim = torch.randint(32,256, size=(n,)).tolist()
batch_dim = torch.randint(16,256, size=(n,)).tolist()
seq_dim = torch.randint(16,256, size=(n,)).tolist()
transpose = [(False, False), (False, True), (True, False), (True, True)]
values = list(product(hidden_dim,batch_dim, transpose, seq_dim))
names = ['hidden_dim_{0}_batch_dim_{1},transpose_{2}_seq_dim_{3}'.format(*vals) for vals in values]
@pytest.mark.parametrize("hidden_dim, batch_dim, transpose, seq_dim", values, ids=names)
def test_igemm(hidden_dim, batch_dim, transpose, seq_dim):
    hidden_dim = hidden_dim - (hidden_dim % 32)
    batch_dim = batch_dim - (batch_dim % 16)
    seq_dim = seq_dim - (seq_dim % 16)
    for i in range(k):
        shapeA = (batch_dim, hidden_dim) if not transpose[0] else (hidden_dim, batch_dim)
        shapeB = ((32*random.randint(1, 4), hidden_dim) if transpose[1] else (hidden_dim, 32*random.randint(1, 4)))
        A = torch.randint(-128, 127, size=shapeA, device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device='cuda').to(torch.int8)
        if not transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.matmul(A.float(), B.t().float())
            out = F.igemm(A, B.t())
        elif transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.t().float(), B.float())
            out = F.igemm(A.t(), B)
        elif transpose[0] and transpose[1]:
            out2 = torch.matmul(A.t().float(), B.t().float())
            out = F.igemm(A.t(), B.t())

        torch.testing.assert_allclose(out.float(), out2)

    for i in range(k):
        shapeA = (batch_dim, seq_dim, hidden_dim)
        shapeB = ((32*random.randint(1, 4), hidden_dim) if transpose[1] else (hidden_dim, 32*random.randint(1, 4)))
        A = torch.randint(-128, 127, size=shapeA, device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device='cuda').to(torch.int8)
        if not transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.matmul(A.float(), B.t().float())
            out = F.igemm(A, B.t())

        torch.testing.assert_allclose(out.float(), out2)


n = 3
k = 25
seq_dim = torch.randint(32,512, size=(n,)).tolist()
hidden_dim = torch.randint(32,1024*4, size=(n,)).tolist()
batch_dim = torch.randint(2,16, size=(n,)).tolist()
values = list(product(seq_dim,hidden_dim,batch_dim))
names = ['seq_dim{0}_hidden_dim{1}_batch_dim{2}'.format(*vals) for vals in values]
@pytest.mark.parametrize("seq_dim, hidden_dim, batch_dim", values, ids=names)
def test_dim3_igemm(seq_dim, hidden_dim, batch_dim):
    seq_dim = seq_dim - (seq_dim % 32)
    hidden_dim = hidden_dim - (hidden_dim % 32)
    batch_dim = batch_dim - (batch_dim % 2)
    for i in range(25):
        A = torch.randint(-128, 127, size=(batch_dim, seq_dim, hidden_dim), device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=(batch_dim, seq_dim, 1024), device='cuda').to(torch.int8)
        out2 = torch.einsum('bsi, bso->io', A.float(), B.float())
        iout = torch.empty(A.shape[2], B.shape[2], dtype=torch.int32, device=A.device)
        out = F.igemm(A, B, out=iout)

        torch.testing.assert_allclose(out.float(), out2)

n = 3
k = 50
seq_dim = torch.randint(32,512, size=(n,)).tolist()
hidden_dim = torch.randint(32,1024*4, size=(n,)).tolist()
batch_dim = torch.randint(2,16, size=(n,)).tolist()
transpose = [False, True]
values = list(product(seq_dim,hidden_dim,batch_dim, transpose))
names = ['seq_dim={0}_hidden_dim={1}_batch_dim={2}_transpose{3}'.format(*vals) for vals in values]
@pytest.mark.parametrize("seq_dim, hidden_dim, batch_dim, transpose", values, ids=names)
def test_minmax_igemm(seq_dim, hidden_dim, batch_dim, transpose):

    def min_max(x):
        maxA = torch.amax(x, dim=2, keepdim=True)
        minA = torch.amin(x, dim=2, keepdim=True)
        scale = (maxA-minA)/2.0
        return (127*(x-minA-scale)/scale).to(torch.int8), minA, scale

    seq_dim = seq_dim - (seq_dim % 16)
    hidden_dim = hidden_dim - (hidden_dim % 16)
    batch_dim = batch_dim - (batch_dim % 2)
    errs = []
    relerrs = []
    errs2 = []
    relerrs2 = []
    for i in range(k):
        A = torch.normal(0.0, 0.5, size=(batch_dim, seq_dim, hidden_dim), device='cuda')
        if transpose:
            B = torch.normal(0, 0.5, size=(1024, hidden_dim), device='cuda')
        else:
            B = torch.normal(0, 0.5, size=(hidden_dim, 1024), device='cuda')
        Ac, minA, scale = min_max(A)
        if transpose:
            maxB, Bc = quant_multi(B, dim=(1 if transpose else 0))
            out = F.igemm(Ac, Bc.t())
            out2 = torch.matmul(A,B.t())
            offset = B.t().sum(0)*(minA+scale)
            out = out.float()
            out = (out*maxB.t()*scale/(127*127))+offset

            maxA, Ac = quant_multi(A, dim=2)
            out3 = F.igemm(Ac, Bc.t())
            out3 = mm_dequant(maxA, maxB.t(), out3)
        else:
            maxB, Bc = quant_multi(B, dim=0)
            offset = B.sum(0)*(minA+scale)
            out = F.igemm(Ac, Bc)
            out2 = torch.matmul(A,B)
            out = out.float()
            out = (out*maxB*scale/(127*127))+offset

            maxA, Ac = quant_multi(A, dim=2)
            out3 = F.igemm(Ac, Bc)
            out3 = mm_dequant(maxA, maxB, out3)

        std = out2.std()
        out2 /= std
        out /= std
        out3 /= std

        err = torch.abs(out-out2)
        relerr = err/(torch.abs(out2)+1e-7)

        err2 = torch.abs(out3-out2)
        relerr2 = err2/(torch.abs(out2)+1e-7)

        errs.append(err.mean().item())
        relerrs.append(relerr.mean().item())
        errs2.append(err2.mean().item())
        relerrs2.append(relerr2.mean().item())
    #print(mean(errs))
    #print(mean(relerrs))
    #print(mean(errs2))
    #print(mean(relerrs2))
    assert mean(errs) < 0.015
    assert mean(relerrs) < 0.3

n = 2
k = 25
dim1 = torch.randint(1,64, size=(n,)).tolist()
dim2 = torch.randint(32,128, size=(n,)).tolist()
dim3 = torch.randint(32,256, size=(n,)).tolist()
dim4 = torch.randint(32,256, size=(n,)).tolist()
transpose = [(False, False), (True, False), (False, True), (True, True)]
values = list(product(dim1,dim2,dim3,dim4,transpose))
names = ['dim1_{0}_dim2_{1}_dim3_{2}_dim4_{3}_transpose_{4}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, dim3, dim4, transpose", values, ids=names)
def test_ibmm(dim1, dim2, dim3, dim4, transpose):
    dim2 = dim2 - (dim2 % 16)
    dim3 = dim3 - (dim3 % 16)
    dim4 = dim4 - (dim4 % 16)
    for i in range(k):
        shapeA = (dim1, dim3, dim2) if transpose[0] else (dim1, dim2, dim3)
        shapeB = (dim1, dim4, dim3) if transpose[1] else (dim1, dim3, dim4)
        A = torch.randint(-128, 127, size=shapeA, device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device='cuda').to(torch.int8)

        if not transpose[0] and not transpose[1]:
            out2 = torch.bmm(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.bmm(A.float(), B.permute([0, 2, 1]).float())
            out = F.igemm(A, B.permute([0, 2, 1]))
        elif transpose[0] and not transpose[1]:
            out2 = torch.bmm(A.permute([0, 2, 1]).float(), B.float())
            out = F.igemm(A.permute([0, 2, 1]), B)
        elif transpose[0] and transpose[1]:
            out2 = torch.bmm(A.permute([0, 2, 1]).float(), B.permute([0, 2, 1]).float())
            out = F.igemm(A.permute([0, 2, 1]), B.permute([0, 2, 1]))
        torch.testing.assert_allclose(out.float(), out2.float())

n = 1
k = 1
dim1 = torch.randint(1,64, size=(n,)).tolist()
dim2 = torch.randint(32,128, size=(n,)).tolist()
dim3 = torch.randint(32,256, size=(n,)).tolist()
values = list(product(dim1,dim2,dim3))
names = ['dim1_{0}_dim2_{1}_dim3_{2}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, dim3", values, ids=names)
def test_vector_quant(dim1, dim2, dim3):
    dim2 = dim2 - (dim2 % 16)
    dim3 = dim3 - (dim3 % 16)
    for i in range(k):
        A = torch.randn(size=(dim2, dim3), device='cuda')
        qA, SA = F.vectorwise_quant(A, dim=0)
        A1 = F.vectorwise_dequant(qA, SA)
        torch.testing.assert_allclose(A1, A, atol=0.01, rtol=0.1)



dim1 = torch.randint(32,1024*4, size=(4,)).tolist()
dim2 = torch.randint(32,1024*4, size=(4,)).tolist()
values = list(product(dim1,dim2))
names = ['dim1_{0}_dim2_{1}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2", values, ids=names)
def test_cutlass_igemm(dim1, dim2):
    dim1 = dim1 - (dim1 % 32)
    dim2 = dim2 - (dim2 % 32)
    for i in range(100):
        A = torch.randint(-128, 127, size=(dim1, dim2), device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=(dim2, dim1), device='cuda').to(torch.int8)
        #A = torch.randn(dim1, dim2, device='cuda').to(torch.float16)
        #B = torch.randn(dim2, dim1, device='cuda').to(torch.float16)
        #A = torch.arange(16*16, device='cuda').view(32, 8).to(torch.int8).contiguous()
        #B = torch.arange(16*16, device='cuda').view(8, 32).to(torch.int8).contiguous()
        #out = F.cutlass_igemm(A.half(), B.half())
        out = F.cutlass_igemm(A, B)
        out2 = torch.mm(A.float(), B.float())
        torch.testing.assert_allclose(out.float(), out2)



def test_cutlass_bench():
    batch = 4
    seq = 512
    model = 1024
    hidden = 8*model
    t = Timer()
    A = torch.randn(batch*seq, model, device='cuda')
    B = torch.randn(model, hidden, device='cuda')
    A = A.half()
    B = B.half()
    C = torch.zeros(batch*seq, hidden, device=A.device, dtype=B.dtype)
    A2 = torch.randint(-128, 127, size=(model, batch*seq), device='cuda').to(torch.int8)
    B2 = torch.randint(-128, 127, size=(model, hidden), device='cuda').to(torch.int8)
    C2 = torch.zeros(batch*seq, hidden, device=A.device, dtype=torch.int32)

    for i in range(1000):
        F.cutlass_igemm(A2.t(), B2, out=C2)
    torch.cuda.synchronize()

    for i in range(1000):
        torch.mm(A, B, out=C)
    torch.cuda.synchronize()


