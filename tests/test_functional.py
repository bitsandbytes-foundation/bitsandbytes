import pytest
import random
import time
import torch
import bitsandbytes as bnb
import einops

from itertools import product

from bitsandbytes import functional as F

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)

class FFN(torch.nn.Module):
    def __init__(self, input_features, hidden_size, bias=True):
        super(FFN, self).__init__()
        self.fc1 = torch.nn.Linear(input_features, hidden_size, bias=bias)
        self.fc2 = torch.nn.Linear(hidden_size, input_features, bias=bias)

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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



n = 2
dim1 = torch.randint(2,256, size=(n,)).tolist()
dim2 = torch.randint(2,256, size=(n,)).tolist()
dim3 = torch.randint(2,256, size=(n,)).tolist()
#dim1, dim2 = (256,), (256,)
dtype = [torch.int8, torch.int32]
a_order = ['row']
out_order = ['col', 'row', 'col32']
transpose = [False]
dims = [2, 3]
values = list(product(dim1,dim2,dim3, dims,dtype, a_order, out_order, transpose))
names = ['dim1_{0}_dim2_{1}_dim3_{2}_dims_{3}_dtype_{4}_orderA_{5}_orderOut_{6}_{7}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose", values, ids=names)
def test_transform(dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose):
    if dims == 3 and out_order != 'col32': return
    if dtype == torch.int32 and out_order != 'col32': return
    func = F.get_transform_func(dtype, orderA, orderOut, transpose)

    if dims == 2:
        A = torch.randint(-128, 127, size=(dim1, dim2), device='cuda').to(dtype)
    elif dims == 3:
        A = torch.randint(-128, 127, size=(dim1, dim2, dim3), device='cuda').to(dtype)

    out, S = F.transform(A, to_order=orderOut)

    if orderOut == 'row':
        torch.testing.assert_allclose(A.flatten(), out.flatten())
    elif orderOut == 'col':
        torch.testing.assert_allclose(A.t().flatten(), out.flatten())
    elif orderOut == 'col32':
        if dims == 2:
            n = A.shape[0]*(A.shape[1] + (32 - (A.shape[1]%32)))
        elif dims == 3:
            n = A.shape[0]*A.shape[1]*(A.shape[2] + (32 - (A.shape[2]%32)))
        assert out.numel() == n
    elif orderOut == 'col_turing':
        # 32 col 8 row tiles
        n = (A.shape[0]+(8- A.shape[0]%8))*(A.shape[1] + (32 - (A.shape[1]%32)))
        assert out.numel() == n
        total_coltile = (A.shape[1] // 32) + (1 if A.shape[1] % 32 != 0 else 0)
        for row in range(A.shape[0]):
            for col in range(A.shape[1]):
                i = row*A.shape[1]
                j = col

                coltile = (col // 32) + (1 if col % 32 != 0 else 0)
                rowtile = ((row // 8) + (1 if row % 8 != 0 else 0))*total_coltile
                offset = 32*8*(rowtile+coltile)
                col2 = col % 32
                row2 = (row%8)*32


                assert A.flatten()[i+j] == A[row, col]
                #assert A.flatten()[i+j] == out.flatten()[row2+col2]
                #torch.testing.assert_allclose(A.flatten()[i+j], A[row, col])
                #torch.testing.assert_allclose(A.flatten()[i+j], out.flatten()[row2+ col2+block_offset])

    if orderOut == 'col32':
        out2, S = F.transform(out, from_order=orderOut, to_order='row', state=S)
        torch.testing.assert_allclose(A, out2)



n = 1
dim1 = torch.randint(1,256, size=(n,)).tolist()
dim2 = torch.randint(32,512, size=(n,)).tolist()
dim3 = torch.randint(32,1024, size=(n,)).tolist()
dim4 = torch.randint(32,1024, size=(n,)).tolist()

#dim1 = [2]
#dim2 = [2]
#dim3 = [2]
#dim4 = [2]

dims = (2,3)
ldb = [0]
#ldb = list(range(256, 1*1024, 256))
values = list(product(dim1,dim2,dim3,dim4,dims, ldb))
names = ['dim1_{0}_dim2_{1}_dim3_{2}_dim4_{3}_dims_{4}_ldb_{5}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, dim3, dim4, dims, ldb", values, ids=names)
def test_igemmlt_int(dim1, dim2, dim3, dim4, dims, ldb):
    print(k)
    for i in range(k):
        if dims == 2:
            A = torch.randint(-128, 127, size=(dim1, dim3), device='cuda').to(torch.int8)
        elif dims == 3:
            A = torch.randint(-128, 127, size=(dim1, dim2, dim3), device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=(dim4, dim3), device='cuda').to(torch.int8)
        C1 = torch.matmul(A.float(), B.t().float())

        A2, SA = F.transform2(A, 'col32')
        B2, SB = F.transform2(B, 'col_turing')
        C2, SC = F.igemmlt(A2, B2, SA, SB)
        C3, S = F.transform(C2, 'row', state=SC)
        torch.testing.assert_allclose(C1, C3.float())

        # transpose
        B = torch.randint(-128, 127, size=(dim3, dim4), device='cuda').to(torch.int8)
        C1 = torch.matmul(A.float(), B.float())

        B2t, SBt = F.transform2(B, 'col_turing', transpose=True)
        C2, SC = F.igemmlt(A2, B2t, SA, SBt)
        C3, S = F.transform(C2, 'row', state=SC)
        torch.testing.assert_allclose(C1, C3.float())

dim1 = [32]
dim2 = [32]
dim3 = [32]
dim4 = [32]

dims = (2,)
#ldb = list(range(256, 1*1024, 256))
values = list(product(dim1,dim2,dim3,dim4,dims))
names = ['dim1_{0}_dim2_{1}_dim3_{2}_dim4_{3}_dims_{4}'.format(*vals) for vals in values]
k = 1
@pytest.mark.parametrize("dim1, dim2, dim3, dim4, dims", values, ids=names)
def test_igemmlt_half(dim1, dim2, dim3, dim4, dims):
    formatB = F.get_special_format_str()
    k = 1
    for i in range(k):
        if dims == 2:
            A = torch.normal(0, 0.5, size=(dim1, dim3), device='cuda').half()
        elif dims == 3:
            A = torch.normal(0, 0.5, size=(dim1, dim2, dim3), device='cuda').half()
        B = torch.randn((dim4, dim3), device='cuda').half()
        torch.nn.init.xavier_uniform_(B)
        C1 = torch.matmul(A, B.t())
        C2 = bnb.matmul(A, B.t())

        A = A.view(-1, A.shape[-1])

        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        CB, CBt, statsB, statsBt, coo_tensor = F.double_quant(B)
        C32A, SA = F.transform2(CA, 'col32')
        CxB, SB = F.transform2(CB, to_order=formatB)
        out1_32, Sout1_32 = F.igemmlt(C32A, CxB, SA, SB)
        output = F.mm_dequant(out1_32, Sout1_32, statsAt, statsBt)

        #print('')
        #print(output.flatten()[:10])
        #print(C1.flatten()[:10])
        #print(C2.flatten()[:10])


        #torch.testing.assert_allclose(C1.view(-1, C1.shape[-1]), output, atol=0.025, rtol=0.05)

        # transpose
        #B = torch.randint(-128, 127, size=(dim3, dim4), device='cuda').to(torch.int8)
        #C1 = torch.matmul(A.float(), B.float())

        #B2t, SBt = F.transform2(B, 'col_turing', transpose=True)
        #C2, SC = F.igemmlt(A2, B2t, SA, SBt)
        #C3, S = F.transform(C2, 'row', state=SC)
        #torch.testing.assert_allclose(C1, C3.float())

seq = [2048]
model = [4*1024]
hidden = [16*1024]
batch_size = 2
seqdim = 2048
values = [(batch_size, seqdim, 4*1024, 16*1024),(batch_size, seqdim, 5120, 4*5120),(batch_size, seqdim, 12*1024, 4*12*1024)]


#values = list(product(batch, seq, model, hidden))
names = ['batch_{0}_seq_{1}_model_{2}_hidden_{3}'.format(*vals) for vals in values]
@pytest.mark.parametrize("batch, seq, model, hidden", values, ids=names)
def test_bench_8bit_training(batch, seq, model, hidden):
    formatB = 'col_ampere'
    A = torch.randn(batch, seq, model, device='cuda').half()
    grad = torch.randn(batch, seq, model, device='cuda').half()
    w1 = torch.randint(-128, 127, size=(hidden, model), device='cuda').half()
    w2 = torch.randint(-128, 127, size=(model, hidden), device='cuda').half()
    print('')

    torch.cuda.synchronize()
    # warmup
    for i in range(100):
        torch.matmul(A, w1.t())
    torch.cuda.synchronize()

    k = 50
    dtype = torch.int8
    A = A.view(-1, A.shape[-1]).contiguous()
    grad = grad.view(-1, grad.shape[-1]).contiguous()
    t0 = time.time()
    for i in range(k):

        out1  = torch.matmul(A, w1.t()) # fc1
        out2 = torch.matmul(out1, w2.t())# fc2

        d1 = torch.matmul(grad, w2) # delta1
        d2 = torch.matmul(d1, w1) # delta2

        grad1 = torch.einsum('bo,bh->oh', out1, grad) # grad w2
        grad2 = torch.einsum('bh,bo->ho', A, d2) # grad w1

    torch.cuda.synchronize()
    t16 = time.time() - t0
    print(t16)

    torch.cuda.empty_cache()

    Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)

    CTw1, Sw1 = F.transform2(Cw1, formatB)
    CTw2, Sw2 = F.transform2(Cw2, formatB)
    CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)
    CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)

    CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
    C32A, SA = F.transform2(CA, 'col32')
    # fc1
    out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1, out_dtype=dtype)
    #out1 = F.mm_dequant(out1_32, Sout1_32, statsAt, statsw1t)

    # fc2
    Cout1, Cout1t, statsout1, statsout1t, coo_tensor = F.double_quant(out1)
    C32out1, Sout1 = F.transform2(Cout1, 'col32')
    out2_32, Sout2_32 = F.igemmlt(C32out1, CTw2, Sout1, Sw2, out_dtype=dtype)
    #out2 = F.mm_dequant(out2_32, Sout2_32, statsout1t, statsw2t)

    # delta1
    Cgrad, Cgradt, statsgrad, statsgradt, coo_tensor = F.double_quant(grad)
    C32grad, Sgrad = F.transform2(Cgrad, 'col32')
    d1_32, Sd1_32 = F.igemmlt(C32grad, CTw2t, Sgrad, Sw2t, out_dtype=dtype)
    #d1 = F.mm_dequant(d1_32, Sd1_32, statsgradt, statsw2)

    # delta2
    Cd1, Cd1t, statsd1, statsd1t, coo_tensor = F.double_quant(d1)
    C32d1, Sd1 = F.transform2(Cd1, 'col32')
    d2_32, Sd2_32 = F.igemmlt(C32d1, CTw1t, Sd1, Sw1t, out_dtype=dtype)
    #d2 = F.mm_dequant(d2_32, Sd2_32, statsd1t, statsw1)

    # grad1
    C32out1t, Sout1t = F.transform2(Cout1t, 'col32', transpose=True)
    CTgradt, Sgradt = F.transform2(Cgradt, formatB, transpose=True)
    grad1_32, Sgrad1_32 = F.igemmlt(C32out1t, CTgradt, Sout1t, Sgradt, out_dtype=dtype)
    #grad1 = F.mm_dequant(grad1_32, Sgrad1_32, statsout1, statsgrad)

    # grad2
    C32At, SAt = F.transform2(CAt, 'col32', transpose=True)
    CTd1t, Sd1t = F.transform2(Cd1t, formatB, transpose=True)
    grad2_32, Sgrad2_32 = F.igemmlt(C32At, CTd1t, SAt, Sd1t, out_dtype=dtype)
    #grad2 = F.mm_dequant(grad2_32, Sgrad2_32, statsA, statsd1)

    torch.cuda.synchronize()

    t0 = time.time()
    for i in range(k):
        Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
        Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)

        CTw1, Sw1 = F.transform2(Cw1, formatB)
        CTw2, Sw2 = F.transform2(Cw2, formatB)
        CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)
        CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)

        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        C32A, SA = F.transform2(CA, 'col32')

        # fc1
        #out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1, out_dtype=dtype)
        out1 = F.mm_dequant(out1_32, Sout1_32, statsAt, statsw1t)

        # fc2
        Cout1, Cout1t, statsout1, statsout1t, coo_tensor = F.double_quant(out1)
        C32out1, Sout1 = F.transform2(Cout1, 'col32')
        #out2_32, Sout2_32 = F.igemmlt(C32out1, CTw2, Sout1, Sw2, out_dtype=dtype)
        out2 = F.mm_dequant(out2_32, Sout2_32, statsout1t, statsw2t)

        # delta1
        Cgrad, Cgradt, statsgrad, statsgradt, coo_tensor = F.double_quant(grad)
        C32grad, Sgrad = F.transform2(Cgrad, 'col32')
        #d1_32, Sd1_32 = F.igemmlt(C32grad, CTw2t, Sgrad, Sw2t, out_dtype=dtype)
        d1 = F.mm_dequant(d1_32, Sd1_32, statsgradt, statsw2)

        # delta2
        Cd1, Cd1t, statsd1, statsd1t, coo_tensor = F.double_quant(d1)
        C32d1, Sd1 = F.transform2(Cd1, 'col32')
        #d2_32, Sd2_32 = F.igemmlt(C32d1, CTw1t, Sd1, Sw1t, out_dtype=dtype)
        d2 = F.mm_dequant(d2_32, Sd2_32, statsd1t, statsw1)

        # grad1
        C32out1t, Sout1t = F.transform2(Cout1t, 'col32', transpose=True)
        CTgradt, Sgradt = F.transform2(Cgradt, formatB, transpose=True)
        #grad1_32, Sgrad1_32 = F.igemmlt(C32out1t, CTgradt, Sout1t, Sgradt, out_dtype=dtype)
        grad1 = F.mm_dequant(grad1_32, Sgrad1_32, statsout1, statsgrad)

        # grad2
        C32At, SAt = F.transform2(CAt, 'col32', transpose=True)
        CTd1t, Sd1t = F.transform2(Cd1t, formatB, transpose=True)
        #grad2_32, Sgrad2_32 = F.igemmlt(C32At, CTd1t, SAt, Sd1t, out_dtype=dtype)
        grad2 = F.mm_dequant(grad2_32, Sgrad2_32, statsA, statsd1)

    torch.cuda.synchronize()
    t8 = time.time() - t0
    print(t8)




dim1 = torch.randint(32,1024*4, size=(4,)).tolist()
dim2 = torch.randint(32,1024*4, size=(4,)).tolist()
values = list(product(dim1,dim2))
names = ['dim1_{0}_dim2_{1}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2", values, ids=names)
def test_cutlass_igemm(dim1, dim2):
    dim1 = dim1 - (dim1 % 32)
    dim2 = dim2 - (dim2 % 32)
    for i in range(100):
        A = torch.randint(-128, 127, size=(4, dim1, dim2), device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=(dim2, dim1), device='cuda').to(torch.int8)
        #A = torch.randn(dim1, dim2, device='cuda').to(torch.float16)
        #B = torch.randn(dim2, dim1, device='cuda').to(torch.float16)
        #A = torch.arange(16*16, device='cuda').view(32, 8).to(torch.int8).contiguous()
        #B = torch.arange(16*16, device='cuda').view(8, 32).to(torch.int8).contiguous()
        out = F.cutlass_igemm(A, B)
        out2 = torch.matmul(A.float(), B.float())
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


tols = {}
tols['forward'] = {'atol': 5e-2, 'rtol': 0.1}
tols['backward'] = {'atol': 5e-2, 'rtol': 0.1}

tols_strict = {}
tols_strict['forward'] = {'atol': 1e-5, 'rtol': 0.1}
tols_strict['backward'] = {'atol': 1e-5, 'rtol': 0.2}
values = ['forward', 'backward']
@pytest.mark.parametrize("action", values, ids=values)
def test_MLP(action):
    batch = 1
    seq = 2
    model = 4
    hidden = 8

    ffn1 = FFN(model, hidden, False)
    ffn2 = bnb.nn.FFN(model, hidden, False)
    ffn1 = ffn1.cuda().half()
    ffn2 = ffn2.cuda().half()

    with torch.no_grad():
        ffn1.fc1.weight.copy_(ffn2.w1)
        ffn1.fc2.weight.copy_(ffn2.w2)
        # same data but different tensors
        assert ffn1.fc1.weight.sum() == ffn2.w1.sum()
        assert ffn1.fc2.weight.sum() == ffn2.w2.sum()
        assert ffn1.fc1.weight.data.storage().data_ptr != ffn2.w1.data.storage().data_ptr



    num_batches = 50
    batches = torch.randn(num_batches, seq, batch, model).cuda()
    total_not_close = 0
    for i in range(num_batches):
        batch = batches[i].half()
        out1 = ffn1(batch)
        out2 = ffn2(batch)
        if 'forward' in action:
            torch.testing.assert_allclose(out1, out2, **tols['forward'])
            total_not_close += (torch.isclose(out1, out2, **tols_strict['forward'])==0).sum().item()
        if 'backward' in action:
            out1.mean().backward()
            out2.mean().backward()
            assert hasattr(ffn1.fc1.weight, 'grad')
            assert hasattr(ffn1.fc2.weight, 'grad')
            assert hasattr(ffn2.w1, 'grad')
            assert hasattr(ffn2.w2, 'grad')
            assert ffn1.fc1.weight.grad is not None
            assert ffn1.fc2.weight.grad is not None
            assert ffn2.w2.grad is not None
            assert ffn2.w1.grad is not None
            #torch.testing.assert_allclose(ffn2.w2.grad, ffn1.fc2.weight.grad, **tols['backward'])
            #torch.testing.assert_allclose(ffn2.w1.grad, ffn1.fc1.weight.grad, **tols['backward'])

            total_not_close += (torch.isclose(ffn2.w2.grad, ffn1.fc2.weight.grad, **tols_strict['backward'])==0).sum().item()
            total_not_close += (torch.isclose(ffn2.w1.grad, ffn1.fc1.weight.grad, **tols_strict['backward'])==0).sum().item()


    print('error exceeded on', total_not_close, 'out of', out2.numel()*num_batches, 'elements')
    assert total_not_close <= (num_batches*out2.numel()*0.1)



backends = ['torch', 'cublaslt']
dims = [(4, 512, 1*1024, 4*1024)]#,(4, 512, 1*1024, 8*1024)]#,(4, 1024, 2*1024, 8*1024),(4, 1024, 2*1024, 16*1024),(4, 2048, 4*1024, 16*1024),(4, 2048, 4*1024, 32*1024)]
#dims = [(4, 2048, 4*1024, 32*1024)]
values = list(product(dims, backends))
names = ['dims_{0}_backend_{1}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dims, backend", values, ids=names)
def test_benchmlp(dims, backend):
    batch, seq, model, hidden = dims
    num_batches = 100

    if backend == 'torch':
        ffn1 = FFN(model, hidden, False)
    else:
        ffn1 = bnb.nn.FFN(model, hidden, False)
    ffn1 = ffn1.cuda().half()
    batches = torch.randn(num_batches, seq, batch, model, device='cuda')

    for i in range(num_batches):
        batch = batches[i].half()
        out1 = ffn1(batch)
        out1.mean().backward()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(num_batches):
        batch = batches[i].half()
        out1 = ffn1(batch)
        out1.mean().backward()
    torch.cuda.synchronize()
    print(time.time()-t0)



n = 2
dim1 = torch.randint(64,256, size=(n,)).tolist()
dim4 = torch.randint(64,1024, size=(n,)).tolist()

#dim1 = [2*1024]
#dim4 = [2*1024]

#dim1 = [4]
#dim4 = [4]

dims = (2,)
#ldb = list(range(256, 1*1024, 256))
formatB = ['col_turing', 'col_ampere']
values = list(product(dim1,dim4,dims, formatB))
names = ['dim1_{0}_dim4_{1}_dims_{2}_formatB_{3}'.format(*vals) for vals in values]
k = 10
@pytest.mark.parametrize("dim1, dim4, dims, formatB", values, ids=names)
def test_dequant_mm(dim1, dim4, dims, formatB):
    inner = torch.randint(1, 128, size=(1,)).item()
    formatB = F.get_special_format_str()
    for i in range(k):
        A = torch.randn(dim1, inner, device='cuda')
        B = torch.randn(dim4, inner, device='cuda')
        C1 = torch.matmul(A.half(), B.t().half())

        A1, maxA = F.vectorwise_quant(A, dim=1)
        B1, maxB = F.vectorwise_quant(B, dim=1)

        A2, SA = F.transform(A1, 'col32')
        B2, SB = F.transform(B1, formatB)
        C2, SC = F.igemmlt(A2, B2, SA, SB)

        C3, S = F.transform(C2, 'row', state=SC)
        C4 = F.vectorwise_mm_dequant(C3.float(), maxA, maxB.t())

        count = (torch.isclose(C1, C4, atol=0.01, rtol=0.1) == 0).sum().item()
        n = C1.numel()
        p = 0.06
        assert count/n < p, f'error in more than {p} of elements: {count}/{n}={count/n}'

        C5 = F.mm_dequant(C2, SC, maxA.flatten(), maxB.flatten())
        torch.testing.assert_allclose(C5, C4)
        #print(C2)



n = 2
dim1 = [1*1024]
dim2 = [1*1024]
#dim1 = torch.randint(1,4*1024, size=(n,)).tolist()
#dim2 = torch.randint(1,4*1024, size=(n,)).tolist()

dims = (2,)
#ldb = list(range(256, 1*1024, 256))
values = list(product(dim1,dim2,dims))
names = ['dim1_{0}_dim2_{1}_dims_{2}'.format(*vals) for vals in values]
k = 1
@pytest.mark.parametrize("dim1, dim2, dims", values, ids=names)
def test_colrow_absmax(dim1, dim2, dims):
    for i in range(k):
        threshold = 3.0
        A = torch.randn(dim1, dim2, device='cuda').half()
        A_truncated = A.clone()
        A_truncated[torch.abs(A_truncated) >= 3.0] = 0.0
        if dims == 2:
            row_stats1, _ = torch.abs(A.float()).max(1)
            col_stats1, _ = torch.abs(A.float()).max(0)
            row_stats1_trunc, _ = torch.abs(A_truncated.float()).max(1)
            col_stats1_trunc, _ = torch.abs(A_truncated.float()).max(0)
        else:
            assert False

        row_stats2, col_stats2, nnz_block_ptr2 = F.get_colrow_absmax(A, threshold=threshold)

        A_blocked = einops.rearrange(torch.abs(A), '(rows row_tiles) (cols block_size)-> rows cols row_tiles block_size', row_tiles=16, block_size=64*4)
        nnz_rows1_counts = (torch.abs(A_blocked)>=threshold).sum(3).flatten()
        nnz_block_ptr1 = torch.zeros(nnz_rows1_counts.shape[0]+1, dtype=nnz_rows1_counts.dtype, device=nnz_rows1_counts.device)
        nnz_block_ptr1[1:] = nnz_rows1_counts.cumsum(0)

        torch.testing.assert_allclose(col_stats1_trunc, col_stats2)
        torch.testing.assert_allclose(row_stats1_trunc, row_stats2)
        torch.testing.assert_allclose(nnz_block_ptr1, nnz_block_ptr2)

        row_stats2, col_stats2, nnz_block_ptr2 = F.get_colrow_absmax(A, threshold=0.0)

        torch.testing.assert_allclose(col_stats1, col_stats2)
        torch.testing.assert_allclose(row_stats1, row_stats2)
        assert nnz_block_ptr2 is None



n = 2
dim1 = [1*1024]
dim2 = [1*1024]
#dim1 = torch.randint(1,4*1024, size=(n,)).tolist()
#dim2 = torch.randint(1,4*1024, size=(n,)).tolist()

dims = (2,)
values = list(product(dim1,dim2,dims))
names = ['dim1_{0}_dim2_{1}_dims_{2}'.format(*vals) for vals in values]
k = 1
@pytest.mark.parametrize("dim1, dim2, dims", values, ids=names)
def test_double_quant(dim1, dim2, dims):
    for i in range(k):
        A = torch.randn(dim1, dim2, device='cuda').half()
        if dims == 2:
            out_col1, Scol = F.vectorwise_quant(A, dim=0)
            out_row1, Srow = F.vectorwise_quant(A, dim=1)
        else:
            assert False

        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)

        # max difference is 1 due to rounding differences
        torch.testing.assert_allclose(CA, out_col1, atol=1, rtol=0)
        torch.testing.assert_allclose(CAt, out_row1, atol=1, rtol=0)


        n = CAt.numel()
        num_not_close_rows = (torch.isclose(CAt, out_row1)==0).sum().item()
        num_not_close_cols = (torch.isclose(CA, out_col1)==0).sum().item()

        # allow for 1:500 error due to rounding differences
        min_error = 1/500
        if num_not_close_cols > (min_error*n):
            print(f'Min error exceeded {num_not_close_cols} elements are different')
            assert False
        if num_not_close_rows > (min_error*n):
            print(f'Min error exceeded {num_not_close_rows} elements are different')
            assert False

        torch.testing.assert_allclose(Scol.flatten(), statsA)
        torch.testing.assert_allclose(Srow.flatten(), statsAt)


# fw
batch = 2
seq = 512
model = 1024
hidden = 4*model

# bw
#batch = 4
#seq = 1024
#hidden = 1024
#model = 4*hidden
batch_seq = batch*seq

# fw
dim1 = [batch_seq, batch_seq, batch_seq]
inner = [4*model, 5*model, 12*model]
dim4 = [4*hidden, 5*hidden, 12*hidden]

# fw2
#dim1 = [batch_seq, batch_seq, batch_seq]
#dim4 = [4*model, 5*model, 12*model]
#inner = [4*hidden, 5*hidden, 12*hidden]

# grad
#inner = [batch_seq, batch_seq, batch_seq]
#dim1 = [4*model, 5*model, 12*model]
#dim4 = [4*hidden, 5*hidden, 12*hidden]

n = 10
#dim1 = torch.randint(1,4*1024, size=(n,)).tolist()
#dim2 = torch.randint(1,4*1024, size=(n,)).tolist()

dims = (2,2, 2)
#ldb = list(range(256, 1*1024, 256))
#values = list(product(dim1,dim4,dims, inner))
values = list(zip(dim1, dim4, dims, inner))
names = ['dim1_{0}_dim4_{1}_dims_{2}_inner_{3}'.format(*vals) for vals in values]
k = 1
@pytest.mark.parametrize("dim1, dim4, dims, inner", values, ids=names)
def test_integrated_igemmlt(dim1, dim4, dims, inner):
    A = torch.randn(dim1, inner, device='cuda')*0.1
    B = torch.randn(dim4, inner, device='cuda')*0.1
    A = A.half()
    B = B.half()
    for i in range(k):
        C1 = torch.matmul(A.half(), B.t().half())

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        C1 = torch.matmul(A, B.t())
    torch.cuda.synchronize()
    t_fp16 = time.time() - t0


    C2, SC = F.transform(torch.zeros(A.shape[0], B.shape[0], dtype=torch.int32, device='cuda'), 'col32')
    torch.cuda.synchronize()
    C1a, C1b, stats1a, stats1b, coo_tensor = F.double_quant(A)
    C2a, C2b, stats2a, stats2b, coo_tensor = F.double_quant(B)
    A2, SA = F.transform(C1a, 'col32')
    B2, SB = F.transform(C2b, 'col_turing')
    F.igemmlt(A2, B2, C2, SA, SB, SC)
    C5 = F.mm_dequant(C2, SC, stats1b, stats2b)
    torch.cuda.synchronize()
    t0 = time.time()
    #print(C1a.numel()/1e6, C1b.numel()/1e6)
    for i in range(k):
        S = F.double_quant(A)
        C1a *= 0
        C1b *= 0
        #A2, SA = F.transform(C1a, 'col32')
        #B2, SB = F.transform(C2b, 'col_turing')

        F.igemmlt(A2, B2, C2, SA, SB, SC)
        C5 = F.mm_dequant(C2, SC, stats1b, stats2b, C5)
    torch.cuda.synchronize()
    t_i8 = time.time() - t0

    print(t_i8, t_fp16, t_fp16/t_i8, dim1, inner, dim4)
    #if t_i8 < t_fp16:
        #print(t_i8, t_fp16, t_fp16/t_i8, dim1, inner, dim4)



n = 2
#dim1 = torch.randint(2,1024, size=(n,)).tolist()
#dim2 = torch.randint(2,1024, size=(n,)).tolist()
dim1 = [8*1024]
dim2 = [4*1024]
#dim1 = [257]
#dim2 = [257]

dim3 = [0]
dtype = [torch.int8]
a_order = ['row']
out_order = ['col32', 'col_turing', 'col_ampere']
transpose = [False, True]
dims = [2]
values = list(product(dim1,dim2,dim3, dims,dtype, a_order, out_order, transpose))
names = ['dim1_{0}_dim2_{1}_dim3_{2}_dims_{3}_dtype_{4}_orderA_{5}_orderOut_{6}_{7}'.format(*vals) for vals in values]
k = 1000
@pytest.mark.parametrize("dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose", values, ids=names)
def test_transform2(dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose):
    for i in range(k):
        if dims == 2:
            A = torch.randint(10, 99, size=(dim1, dim2), device='cuda').to(dtype)
        elif dims == 3:
            A = torch.randint(10, 99, size=(dim1, dim2, dim3), device='cuda').to(dtype)

        A.view(-1)[-1] = -1
        if transpose:
            At = A.t().contiguous()
            out1, S1 = F.transform(At, to_order=orderOut)
        else:
            out1, S1 = F.transform(A, to_order=orderOut)
        out2, S2 = F.transform2(A, to_order=orderOut, transpose=transpose)

        assert S1[0][0] == S2[0][0]
        assert S1[0][1] == S2[0][1]
        #print(out1)
        #print(out2)

        torch.testing.assert_allclose(out1, out2)




def test_overflow():
    for i in range(2):
        a = torch.arange(5, 15).cuda().to(torch.int8).view(-1,1 )
        b = torch.arange(5, 15).cuda().to(torch.int8).view(-1,1 )

        Ca, Sa = F.transform(a, 'col32')
        Cb, Sb = F.transform(b, 'col_ampere')

        c = F.igemmlt(Ca, Cb, Sa, Sb, out_dtype=torch.int8)
        c2 = torch.matmul(a.float(), b.float().t())
    print(c)
    print(c2)



dim1 = [1]
dim2 = [4*1024]
#dim1 = [32]
#dim2 = [32]
#dim1 = torch.randint(1,4*1024, size=(n,)).tolist()
#dim2 = torch.randint(1,4*1024, size=(n,)).tolist()

dims = (2,)
values = list(product(dim1,dim2,dims))
names = ['dim1_{0}_dim2_{1}_dims_{2}'.format(*vals) for vals in values]
k = 1000
@pytest.mark.parametrize("dim1, dim2, dims", values, ids=names)
def test_coo_double_quant(dim1, dim2, dims):
    threshold = 3.0
    for i in range(k):
        A = torch.randn(dim1, dim2, device='cuda').half()
        vals1 = A[(torch.abs(A) >= 3.0)]
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A, threshold=threshold)


        #print(coo_tensor.rowidx.numel())
        #print(coo_tensor.rowidx)
        #print(coo_tensor.colidx)

        if coo_tensor is not None:
            val, counts = torch.unique(coo_tensor.rowidx, return_counts=True)
            start = 0
            for c in counts:
                col = coo_tensor.colidx[start:start+c]
                val2 = coo_tensor.values[start:start+c]
                if c > 1:
                    colval, idx = torch.sort(col)
                    val2 = val2[idx]
                #print(vals1[start:start+c])
                #print(val2)
                print(vals1)
                print(coo_tensor.values)
                print((coo_tensor.values == 0).sum().item())
                #print(coo_tensor.rowidx)
                #print(coo_tensor.colidx)
                torch.testing.assert_allclose(vals1[start:start+c], val2)

                start += c

        # max difference is 1 due to rounding differences
        #torch.testing.assert_allclose(CA, out_col1, atol=1, rtol=0)
        #torch.testing.assert_allclose(CAt, out_row1, atol=1, rtol=0)


        #n = CAt.numel()
        #num_not_close_rows = (torch.isclose(CAt, out_row1)==0).sum().item()
        #num_not_close_cols = (torch.isclose(CA, out_col1)==0).sum().item()

        ## allow for 1:500 error due to rounding differences
        #min_error = 1/500
        #if num_not_close_cols > (min_error*n):
        #    print(f'Min error exceeded {num_not_close_cols} elements are different')
        #    assert False
        #if num_not_close_rows > (min_error*n):
        #    print(f'Min error exceeded {num_not_close_rows} elements are different')
        #    assert False

        #torch.testing.assert_allclose(Scol.flatten(), statsA)
        #torch.testing.assert_allclose(Srow.flatten(), statsAt)

n = 2
dim1 = torch.randint(1,1*1024, size=(n,)).tolist()
dim2 = torch.randint(1,1*1024, size=(n,)).tolist()
values = list(product(dim1,dim2))
names = ['dim1_{0}_dim2_{1}'.format(*vals) for vals in values]
k = 10
@pytest.mark.parametrize("dim1, dim2", values, ids=names)
def test_spmm_coo(dim1, dim2):
    threshold = 3.01
    for i in range(k):
        A = torch.randn(dim1, dim2).cuda().half()
        B = torch.randn(dim2, dim1).cuda().half()

        idx = torch.abs(A) >= threshold
        nnz = (idx == 1).sum().item()
        rows, cols = torch.where(idx)
        values = A[idx]
        cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
        out2 = F.spmm_coo(cooA, B)

        A2 = A*idx
        out1 = torch.matmul(A2, B)
        torch.testing.assert_allclose(out1, out2, rtol=0.01, atol=1.2e-2)



def test_spmm_bench():
    dim1 = 1024*8
    dim2 = 1024*8
    threshold = 6
    A = torch.randn(dim1, dim2).cuda().half()
    B = torch.randn(dim2, dim1).cuda().half()
    for i in range(100):
        C1 = bnb.matmullt(A, B)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        C1 = bnb.matmullt(A, B)
    torch.cuda.synchronize()
    t8 = time.time()-t0

    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)

    for i in range(100):
        out2 = F.spmm_coo(cooA, B)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        out2 = F.spmm_coo(cooA, B)
    torch.cuda.synchronize()
    tsp = time.time()-t0
    print(tsp, t8)
    print(tsp/t8)

