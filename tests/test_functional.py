import math
import random
import time

import einops
import numpy as np
import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F
from bitsandbytes.cextension import HIP_ENVIRONMENT, ROCM_GPU_ARCH
from tests.helpers import (
    BOOLEAN_TUPLES,
    TRUE_FALSE,
    describe_dtype,
    get_available_devices,
    get_test_dims,
    id_formatter,
    is_supported_on_hpu,
)

torch.set_printoptions(precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)
k = 20


def assert_all_approx_close(a, b, rtol=1e-3, atol=1e-3, count=0, throw=True):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    sumval = (idx == 0).sum().item()
    if sumval > count:
        if throw:
            print(f"Too many values not close: assert {sumval} < {count}")
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

    return sumval


class FFN(torch.nn.Module):
    def __init__(self, input_features, hidden_size, bias=True):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_features, hidden_size, bias=bias)
        self.fc2 = torch.nn.Linear(hidden_size, input_features, bias=bias)

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Timer:
    def __init__(self):
        self.starts = {}
        self.ends = {}
        self.agg = {}

    def tick(self, name="default"):
        if name not in self.starts:
            self.starts[name] = torch.cuda.Event(enable_timing=True)
            self.ends[name] = torch.cuda.Event(enable_timing=True)
            self.starts[name].record()
        else:
            ms = self.tock(name, evict=True, print_ms=False)

    def tock(self, name="default", evict=True, print_ms=True):
        if name in self.ends:
            self.ends[name].record()
            torch.cuda.synchronize()
            ms = self.starts[name].elapsed_time(self.ends[name])
            if name not in self.agg:
                self.agg[name] = 0.0
            self.agg[name] += ms
            if evict:
                self.starts.pop(name)
                self.ends.pop(name)

        if print_ms and name in self.agg:
            print(f"{name} took: {self.agg[name] / 1000.0:.5f}s")

        return self.agg[name]

    def reset(self):
        self.starts = {}
        self.ends = {}
        self.agg = {}
        print("Resetting benchmark data")


class Test8BitBlockwiseQuantizeFunctional:
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
    @pytest.mark.parametrize("nested", TRUE_FALSE, ids=id_formatter("nested"))
    @pytest.mark.parametrize(
        "blocksize",
        [4096, 2048, 1024, 512, 256, 128, 64] if not HIP_ENVIRONMENT else [4096, 2048, 1024, 512, 256, 128],
    )
    @pytest.mark.parametrize("signed", TRUE_FALSE, ids=id_formatter("signed"))
    def test_dynamic_blockwise_quantization(self, device, dtype, nested, blocksize, signed):
        iters = 100

        if device == "cpu":
            iters = 10

            # This test is slow on CPU, so avoid atypical use cases.
            if nested:
                pytest.skip("Not a typical use case.")
            if blocksize != 256:
                pytest.skip("Only blocksize 256 is used in CPU/XPU")
            if dtype != torch.float32:
                pytest.skip("Only float32 is used in CPU/XPU")

        diffs = []
        reldiffs = []
        for i in range(iters):
            A1 = torch.randn(1024, 1024, device=device, dtype=dtype)
            C, S = F.quantize_blockwise(A1, blocksize=blocksize, nested=nested)
            if i == 0:
                d = S.as_dict()
                S = F.QuantState.from_dict(d, device=torch.device(device))
            A2 = F.dequantize_blockwise(C, S)
            diff = torch.abs(A1 - A2).float()
            reldiff = diff / torch.abs(A1.float() + 1e-8)
            diffs.append(diff.mean().item())
            reldiffs.append(reldiff.mean().item())
        abserr = sum(diffs) / len(diffs)
        relerr = sum(reldiffs) / len(reldiffs)
        assert abserr < 0.011
        assert relerr < 0.018
        assert A2.dtype == dtype

        diffs = []
        code = F.create_dynamic_map(signed=signed)
        for i in range(iters):
            A1 = torch.rand(1024, 1024, device=device, dtype=dtype)
            C, S = F.quantize_blockwise(A1, blocksize=blocksize, nested=nested, code=code)
            if i == 0:
                d = S.as_dict()
                S = F.QuantState.from_dict(d, device=torch.device(device))
            A2 = F.dequantize_blockwise(C, S)
            diff = torch.abs(A1 - A2).float()
            reldiff = diff / torch.abs(A1.float() + 1e-8)
            diffs.append(diff.mean().item())
            reldiffs.append(reldiff.mean().item())
            # torch.testing.assert_close(A1, A2, atol=1e-2, rtol=0)
        abserr = sum(diffs) / len(diffs)
        relerr = sum(reldiffs) / len(reldiffs)
        if signed:
            threshold_abserr = 0.0036 if device in ("cpu", "xpu") and (F.ipex_cpu or F.ipex_xpu) else 0.0035
            assert abserr < 0.0036
            assert relerr < 0.015
        else:
            assert abserr < 0.00175 if device in ("cpu", "xpu") and (F.ipex_cpu or F.ipex_xpu) else 0.0023
            assert relerr < 0.012
        assert A2.dtype == dtype

    @pytest.mark.skipif("cpu" not in get_available_devices(), reason="CPU is required")
    @pytest.mark.parametrize("hidden", [128])
    @pytest.mark.parametrize("blocksize", [4096, 16384])
    def test_blockwise_cpu_large(self, hidden, blocksize):
        diffs = []
        reldiffs = []
        batch = 128
        seq = 128

        for i in range(2):
            A1 = torch.randn(batch, seq, hidden, device="cpu")
            t0 = time.time()
            C, S = F.quantize_blockwise(A1, blocksize=blocksize)
            A2 = F.dequantize_blockwise(C, S, blocksize=blocksize)
            print(time.time() - t0)
            diff = torch.abs(A1 - A2)
            reldiff = diff / torch.abs(A1 + 1e-8)
            diffs.append(diff.mean().item())
            reldiffs.append(reldiff.mean().item())
            assert diffs[-1] < 0.011
        # print(sum(diffs)/len(diffs))
        # print(sum(reldiffs)/len(reldiffs))

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("bits", range(2, 9), ids=id_formatter("bits"))
    @pytest.mark.parametrize("method", ["linear", "fp8", "dynamic"])
    def test_few_bit_quant(self, device, bits, method):
        if bits != 8 and (device == "cpu" or (device == "xpu" and F.ipex_xpu)):
            pytest.skip("CPU/XPU implementation only supports 8 bits")

        abserrs = []
        relerrs = []
        code = None
        if method == "linear":
            code = F.create_linear_map(True, total_bits=bits).to(device)
        elif method == "fp8":
            ebits = math.ceil(bits / 2)
            pbits = bits - ebits - 1
            code = F.create_fp8_map(True, ebits, pbits, bits).to(device)
        elif method == "dynamic":
            code = F.create_dynamic_map(True, bits - 0, bits).to(device)

        # for some data types we have no zero
        # for some data types we have one zero
        # for some data types we have two zeros
        assert torch.unique(code).numel() in [2**bits, 2**bits - 1], f"bits: {bits}, method: {method}"
        # print(method, (code==0).sum())
        assert code.numel() == 256
        for i in range(10):
            values = torch.randn(1, 32, device=device)
            values /= values.abs().max()
            # values[values.abs() < 1e-6] += 1e-5

            q1 = []
            v1 = []
            for v in values[0]:
                idx = torch.abs(v - code).argmin()
                q1.append(idx.item())
                v1.append(code[idx].item())

            q1 = torch.tensor(q1, device=device)
            v1 = torch.tensor(v1, device=device)

            q2, S2 = F.quantize_blockwise(values, code=code)
            v2 = F.dequantize_blockwise(q2, S2)

            idx = torch.isclose(q1.int(), q2.int())
            err2 = torch.abs(v2 - values)
            abserrs.append(err2.mean().item())
            relerrs.append((err2 / (1e-10 + values).abs()).mean().item())
            if idx.sum():
                # some weird cases
                err1 = torch.abs(v1 - values).mean()
                # assert err2.mean() <= err1
            else:
                torch.testing.assert_close(q1, q2)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_fp8_quant(self, device):
        # TODO
        if device == "cpu":
            pytest.skip("CPU implementation segfaults")

        for e_bits in range(1, 7):
            p_bits = 7 - e_bits
            code = F.create_fp8_map(True, e_bits, p_bits).to(device)

            abserr = []
            relerr = []
            for i in range(100):
                A1 = torch.randn(1024, 1024, device=device)
                C, SC = F.quantize_blockwise(A1, code=code)
                if i == 0:
                    d = SC.as_dict()
                    SC = F.QuantState.from_dict(d, device=torch.device(device))
                A2 = F.dequantize_blockwise(C, SC)
                diff = torch.abs(A1 - A2)
                reldiff = diff / torch.abs(A1 + 1e-8)
                abserr.append(diff.mean().item())
                relerr.append(reldiff.mean().item())
                # assert diff < 0.0075
            # print(sum(abserr)/len(abserr))
            # print(sum(relerr)/len(relerr))

            abserr = []
            relerr = []
            for i in range(100):
                A1 = torch.rand(1024, 1024, device=device)
                C, SC = F.quantize_blockwise(A1, code=code)
                A2 = F.dequantize_blockwise(C, SC)
                diff = torch.abs(A1 - A2)
                reldiff = diff / torch.abs(A1 + 1e-8)
                abserr.append(diff.mean().item())
                relerr.append(reldiff.mean().item())
                # assert diff < 0.0075
            # print(sum(abserr)/len(abserr))
            # print(sum(relerr)/len(relerr))

            abserr = []
            relerr = []
            for i in range(100):
                A1 = torch.randn(1024, 1024, device=device)
                C, SC = F.quantize_blockwise(A1)
                A2 = F.dequantize_blockwise(C, SC)
                diff = torch.abs(A1 - A2)
                reldiff = diff / torch.abs(A1 + 1e-8)
                abserr.append(diff.mean().item())
                relerr.append(reldiff.mean().item())
                # assert diff < 0.0075
            # print(3, sum(abserr)/len(abserr))
            # print(3, sum(relerr)/len(relerr))

    @pytest.mark.benchmark
    def test_bench_dequantization(self):
        a = torch.rand(1024, 1024, device="cuda").half()
        code = F.create_fp8_map(True, 3, 0, 4).cuda()
        qa, SA = F.quantize_blockwise(a, code=code)
        print(qa.max())

        max_theoretical_mu = 1024 * 1024 * 2 / 1024**3 / 672 * 1000 * 1000
        # print(max_theoretical_mu)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(100):
            qa, SA = F.quantize_blockwise(a)
        torch.cuda.synchronize()
        # print((time.time()-t0)/1e6)


def test_stable_embedding():
    layer = bnb.nn.StableEmbedding(1024, 1024)
    layer.reset_parameters()


def quant(x):
    max1 = torch.abs(x).max()
    x = torch.round(x / max1 * 127)
    return max1, x.to(torch.int8)


def dequant(c, maxC):
    return c.float() * (maxC / 127)


def mm_dequant(maxA, maxB, C):
    return C.float() * (maxA / 127) * (maxB / 127)


def quant_multi(x, dim):
    max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
    max1[max1 == 0] = 1.0
    x = torch.round(x / max1 * 127)
    return max1, x.to(torch.int8)


def quant_multi_chunk(x, dim, chunk_size=32):
    if dim == 1:
        x_chunked = einops.rearrange(x, "(c a) b -> c a b", c=chunk_size)
        max1 = torch.amax(torch.abs(x_chunked), dim=dim + 1, keepdim=True)
        max1 = torch.tile(max1, (1, 1, x.shape[1]))
        max1 = max1.view(x.shape)
    elif dim == 0:
        x_chunked = einops.rearrange(x, "a (b c) -> a b c", c=chunk_size)
        max1 = torch.amax(torch.abs(x_chunked), dim=dim, keepdim=True)
        max1 = torch.tile(max1, (x.shape[0], 1, 1))
        max1 = max1.view(x.shape)
    max1[max1 == 0] = 1.0
    x = torch.round(x / max1 * 127)
    return max1, x.to(torch.int8)


def mean(xx):
    return sum(xx) / float(len(xx))


methods = {
    "linear": (
        lambda x, dim: quant(x),
        lambda x, dim: quant(x),
        dequant,
        dequant,
        mm_dequant,
    ),
    "vectorwise": (quant_multi, quant_multi, dequant, dequant, mm_dequant),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestIGEMMFunctional:
    @pytest.mark.parametrize("dim1", [1024 * 2], ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim2", [1024 * 16], ids=id_formatter("dim2"))
    @pytest.mark.parametrize("quant_methods", methods.values(), ids=methods.keys())
    @pytest.mark.parametrize("batched", TRUE_FALSE, ids=id_formatter("batched"))
    def test_approx_igemm(self, dim1, dim2, quant_methods, batched):
        dim1 = dim1 - (dim1 % 32)
        dim2 = dim2 - (dim2 % 32)
        errors = []
        relerrors = []
        # print("")
        for i in range(5):
            if batched:
                A = torch.normal(0, 0.5, size=(32, dim1, dim2 // 32), device="cuda")
                B = torch.normal(0, 0.5, size=(32, dim2 // 32, dim1), device="cuda")
                maxA, Ac = quant_methods[0](A, 2)
                maxB, Bc = quant_methods[1](B, 1)
            else:
                A = torch.normal(0, 0.5, size=(dim1, dim2), device="cuda")
                B = torch.normal(0, 0.5, size=(dim2, dim1), device="cuda")
                maxA, Ac = quant_methods[0](A, 1)
                maxB, Bc = quant_methods[1](B, 0)
            torch.testing.assert_close(quant_methods[2](maxA, Ac), A, atol=0.025, rtol=0.05)
            if batched:
                out2 = torch.bmm(A, B)
                C = torch.bmm(Ac.float(), Bc.float())
            else:
                out2 = torch.mm(A, B)
                C = F.igemm(Ac, Bc)
            out = quant_methods[4](maxA, maxB, C)
            std = out2.std()
            out /= std
            out2 /= std
            err = torch.abs(out - out2)
            relerr = err / torch.abs(out2)
            errors.append(err.mean().item())
            relerrors.append(relerr.mean().item())
        # print(mean(errors))
        # print(mean(relerrors))

    @pytest.mark.parametrize("hidden_dim", [32, 256], ids=id_formatter("hidden_dim"))
    @pytest.mark.parametrize("batch_dim", [16, 256], ids=id_formatter("batch_dim"))
    @pytest.mark.parametrize("seq_dim", [16, 256], ids=id_formatter("seq_dim"))
    @pytest.mark.parametrize("transpose", BOOLEAN_TUPLES, ids=id_formatter("transpose"))
    def test_igemm(self, hidden_dim, batch_dim, transpose, seq_dim):
        hidden_dim = hidden_dim - (hidden_dim % 32)
        batch_dim = batch_dim - (batch_dim % 16)
        seq_dim = seq_dim - (seq_dim % 16)
        for i in range(k):
            shapeA = (batch_dim, hidden_dim) if not transpose[0] else (hidden_dim, batch_dim)
            shapeB = (
                (32 * random.randint(1, 4), hidden_dim) if transpose[1] else (hidden_dim, 32 * random.randint(1, 4))
            )
            A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
            B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)
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

            torch.testing.assert_close(out.float(), out2)

        for i in range(k):
            shapeA = (batch_dim, seq_dim, hidden_dim)
            shapeB = (
                (32 * random.randint(1, 4), hidden_dim) if transpose[1] else (hidden_dim, 32 * random.randint(1, 4))
            )
            A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
            B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)
            if not transpose[0] and not transpose[1]:
                out2 = torch.matmul(A.float(), B.float())
                out = F.igemm(A, B)
            elif not transpose[0] and transpose[1]:
                out2 = torch.matmul(A.float(), B.t().float())
                out = F.igemm(A, B.t())

            torch.testing.assert_close(out.float(), out2)

    @pytest.mark.parametrize("seq_dim", [32, 256, 512], ids=id_formatter("seq_dim"))
    @pytest.mark.parametrize("hidden_dim", [64, 1024, 4096], ids=id_formatter("hidden_dim"))
    @pytest.mark.parametrize("batch_dim", [2, 8, 16], ids=id_formatter("batch_dim"))
    def test_dim3_igemm(self, seq_dim, hidden_dim, batch_dim):
        seq_dim = seq_dim - (seq_dim % 32)
        hidden_dim = hidden_dim - (hidden_dim % 32)
        batch_dim = batch_dim - (batch_dim % 2)
        for i in range(25):
            A = torch.randint(-128, 127, size=(batch_dim, seq_dim, hidden_dim), device="cuda").to(torch.int8)
            B = torch.randint(-128, 127, size=(batch_dim, seq_dim, 1024), device="cuda").to(torch.int8)
            out2 = torch.einsum("bsi, bso->io", A.float(), B.float())
            iout = torch.empty(A.shape[2], B.shape[2], dtype=torch.int32, device=A.device)
            out = F.igemm(A, B, out=iout)

            torch.testing.assert_close(out.float(), out2)

    @pytest.mark.parametrize("seq_dim", [32, 512], ids=id_formatter("seq_dim"))
    @pytest.mark.parametrize("hidden_dim", [32, 1024 * 4], ids=id_formatter("hidden_dim"))
    @pytest.mark.parametrize("batch_dim", [2, 16], ids=id_formatter("batch_dim"))
    @pytest.mark.parametrize("transpose", TRUE_FALSE, ids=id_formatter("transpose"))
    def test_minmax_igemm(self, seq_dim, hidden_dim, batch_dim, transpose):
        def min_max(x):
            maxA = torch.amax(x, dim=2, keepdim=True)
            minA = torch.amin(x, dim=2, keepdim=True)
            scale = (maxA - minA) / 2.0
            return (127 * (x - minA - scale) / scale).to(torch.int8), minA, scale

        seq_dim = seq_dim - (seq_dim % 16)
        hidden_dim = hidden_dim - (hidden_dim % 16)
        batch_dim = batch_dim - (batch_dim % 2)
        errs = []
        relerrs = []
        errs2 = []
        relerrs2 = []
        for i in range(k):
            A = torch.normal(0.0, 0.5, size=(batch_dim, seq_dim, hidden_dim), device="cuda")
            if transpose:
                B = torch.normal(0, 0.5, size=(256, hidden_dim), device="cuda")
            else:
                B = torch.normal(0, 0.5, size=(hidden_dim, 256), device="cuda")
            Ac, minA, scale = min_max(A)
            if transpose:
                maxB, Bc = quant_multi(B, dim=(1 if transpose else 0))
                out = F.igemm(Ac, Bc.t())
                out2 = torch.matmul(A, B.t())
                offset = B.t().sum(0) * (minA + scale)
                out = out.float()
                out = (out * maxB.t() * scale / (127 * 127)) + offset

                maxA, Ac = quant_multi(A, dim=2)
                out3 = F.igemm(Ac, Bc.t())
                out3 = mm_dequant(maxA, maxB.t(), out3)
            else:
                maxB, Bc = quant_multi(B, dim=0)
                offset = B.sum(0) * (minA + scale)
                out = F.igemm(Ac, Bc)
                out2 = torch.matmul(A, B)
                out = out.float()
                out = (out * maxB * scale / (127 * 127)) + offset

                maxA, Ac = quant_multi(A, dim=2)
                out3 = F.igemm(Ac, Bc)
                out3 = mm_dequant(maxA, maxB, out3)

            std = out2.std()
            out2 /= std
            out /= std
            out3 /= std

            err = torch.abs(out - out2)
            relerr = err / (torch.abs(out2) + 1e-7)

            err2 = torch.abs(out3 - out2)
            relerr2 = err2 / (torch.abs(out2) + 1e-7)

            errs.append(err.mean().item())
            relerrs.append(relerr.mean().item())
            errs2.append(err2.mean().item())
            relerrs2.append(relerr2.mean().item())
        # print(mean(errs))
        # print(mean(relerrs))
        # print(mean(errs2))
        # print(mean(relerrs2))
        assert mean(errs) < 0.015

        # There's a higher relerr on L40S with torch 2.4+cu118.
        is_sm89 = torch.cuda.get_device_capability() == (8, 9)
        if torch.version.cuda == "11.8" and is_sm89 and torch.__version__ < (2, 5):
            assert mean(relerrs) < 0.41
        else:
            assert mean(relerrs) < 0.3

    @pytest.mark.parametrize("dim1", [1, 64], ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim2", [32, 128], ids=id_formatter("dim2"))
    @pytest.mark.parametrize("dim3", [32, 256], ids=id_formatter("dim3"))
    @pytest.mark.parametrize("dim4", [32, 256], ids=id_formatter("dim4"))
    @pytest.mark.parametrize("transpose", BOOLEAN_TUPLES, ids=id_formatter("transpose"))
    def test_ibmm(self, dim1, dim2, dim3, dim4, transpose):
        dim2 = dim2 - (dim2 % 16)
        dim3 = dim3 - (dim3 % 16)
        dim4 = dim4 - (dim4 % 16)
        for i in range(k):
            shapeA = (dim1, dim3, dim2) if transpose[0] else (dim1, dim2, dim3)
            shapeB = (dim1, dim4, dim3) if transpose[1] else (dim1, dim3, dim4)
            A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
            B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)

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
            torch.testing.assert_close(out.float(), out2.float())


class TestLLMInt8Functional:
    @staticmethod
    def vectorwise_mm_dequant(xq, S1, S2, dtype=torch.half):
        """Reference implementation for the F.int8_mm_dequant function."""
        C = 127.0

        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= S1 / C
        else:
            x *= S1 / C
        x *= S2 / C
        return x.to(dtype)

    @staticmethod
    def vectorwise_quant(x, dim=1):
        """Reference implementation"""
        max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        xq = torch.round(x * (127.0 / max1)).to(torch.int8)
        return xq, max1

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dim1", [128], ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim2", [256], ids=id_formatter("dim2"))
    @pytest.mark.parametrize("dim3", [499, 512], ids=id_formatter("dim3"))
    @pytest.mark.parametrize("dim4", [512], ids=id_formatter("dim4"))
    @pytest.mark.parametrize("dims", (2, 3), ids=id_formatter("dims"))
    @pytest.mark.parametrize("ldb", (0,), ids=id_formatter("ldb"))
    def test_int8_linear_matmul(self, device, dim1, dim2, dim3, dim4, dims, ldb):
        for i in range(k):
            if dims == 2:
                A = torch.randint(-128, 127, size=(dim1, dim3), dtype=torch.int8, device=device)
            elif dims == 3:
                A = torch.randint(-128, 127, size=(dim1, dim2, dim3), dtype=torch.int8, device=device)
            B = torch.randint(-128, 127, size=(dim4, dim3), dtype=torch.int8, device=device)
            C1 = torch.matmul(A.float(), B.t().float())

            C2 = F.int8_linear_matmul(A, B)
            torch.testing.assert_close(C1, C2.float())

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dim1", [32], ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim2", [32], ids=id_formatter("dim2"))
    @pytest.mark.parametrize("dim3", [32], ids=id_formatter("dim3"))
    @pytest.mark.parametrize("dim4", [32], ids=id_formatter("dim4"))
    @pytest.mark.parametrize("dims", (2,), ids=id_formatter("dims"))
    def test_int8_linear_matmul_half(self, device, dim1, dim2, dim3, dim4, dims):
        for i in range(k):
            if dims == 2:
                A = torch.normal(0, 0.5, size=(dim1, dim3), device=device).half()
            elif dims == 3:
                A = torch.normal(0, 0.5, size=(dim1, dim2, dim3), device=device).half()
            B = torch.randn((dim4, dim3), device=device).half()
            torch.nn.init.xavier_uniform_(B)
            C1 = torch.matmul(A, B.t())

            A = A.view(-1, A.shape[-1])

            CA, statsA, _ = F.int8_vectorwise_quant(A)
            CB, statsB, _ = F.int8_vectorwise_quant(B)
            output = F.int8_mm_dequant(F.int8_linear_matmul(CA, CB), statsA, statsB)

            torch.testing.assert_close(C1.view(-1, C1.shape[-1]), output, atol=0.025, rtol=0.05)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dim1", (64, 256), ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim4", (64, 1024), ids=id_formatter("dim4"))
    @pytest.mark.parametrize("dims", (2,), ids=id_formatter("dims"))
    @pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
    def test_dequant_mm(self, device, dim1, dim4, dims, has_bias):
        inner = 128
        bias = None
        if has_bias:
            bias = torch.randn(dim4, device=device, dtype=torch.float16)

        for i in range(1):
            A = torch.randn(dim1, inner, device=device)
            B = torch.randn(dim4, inner, device=device)
            C1 = torch.matmul(A.half(), B.t().half())
            if has_bias:
                C1 += bias

            A1, maxA = self.vectorwise_quant(A, dim=1)
            B1, maxB = self.vectorwise_quant(B, dim=1)

            C2 = F.int8_linear_matmul(A1, B1)

            C4 = self.vectorwise_mm_dequant(C2.float(), maxA, maxB.t())
            if has_bias:
                C4 += bias

            # TODO: is something wrong here? If so, the problem goes deeper
            # n = C1.numel()
            # p = 0.06
            std = C1.std(0).view(1, -1)
            C1 /= std
            C4 /= std
            # assert_all_approx_close(C1, C4, atol=0.02, rtol=0.1, count=int(n*0.06))
            # assert (count / n < p), f"error in more than {p} of elements: {count}/{n}={count/n}"

            C5 = F.int8_mm_dequant(C2, maxA, maxB, bias=bias)
            C5 /= std
            torch.testing.assert_close(C5, C4, atol=0.015, rtol=0.1)
            n = C5.numel()
            assert_all_approx_close(C1, C4, atol=0.015, rtol=0.1, count=int(0.01 * n))

    @pytest.mark.parametrize("dim1", [1 * 1024], ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim2", [1 * 1024], ids=id_formatter("dim2"))
    @pytest.mark.parametrize("dims", (2,), ids=id_formatter("dims"))
    @pytest.mark.parametrize("threshold", [0.0, 3.0], ids=id_formatter("decomp"))
    @pytest.mark.deprecated
    def test_colrow_absmax(self, dim1, dim2, dims, threshold):
        for i in range(k):
            A = torch.randn(dim1, dim2, device="cuda").half()

            assert dims == 2

            row_stats1, _ = torch.abs(A.float()).max(1)
            col_stats1, _ = torch.abs(A.float()).max(0)

            if threshold > 0.0:
                A_truncated = A.clone()
                A_truncated[torch.abs(A_truncated) >= threshold] = 0.0
                row_stats1_trunc, _ = torch.abs(A_truncated.float()).max(1)
                col_stats1_trunc, _ = torch.abs(A_truncated.float()).max(0)

                row_stats2, col_stats2, nnz_block_ptr2 = F.get_colrow_absmax(A, threshold=threshold)

                nnz_rows1_counts = (torch.abs(A) >= threshold).sum(1).flatten()
                nnz_block_ptr1 = torch.zeros(
                    nnz_rows1_counts.shape[0] + 1,
                    dtype=nnz_rows1_counts.dtype,
                    device=nnz_rows1_counts.device,
                )
                nnz_block_ptr1[1:] = nnz_rows1_counts.cumsum(0)

                torch.testing.assert_close(col_stats1_trunc, col_stats2)
                torch.testing.assert_close(row_stats1_trunc, row_stats2)
                # torch.testing.assert_close(nnz_block_ptr1, nnz_block_ptr2)
            else:
                row_stats2, col_stats2, nnz_block_ptr2 = F.get_colrow_absmax(A, threshold=0.0)
                assert nnz_block_ptr2 is None
                torch.testing.assert_close(col_stats1, col_stats2)
                torch.testing.assert_close(row_stats1, row_stats2)

    @pytest.mark.parametrize("dim1", [2048, 4096], ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim2", [512, 1024], ids=id_formatter("dim2"))
    @pytest.mark.deprecated
    def test_int8_double_quant(self, dim1, dim2):
        for i in range(k):
            A = torch.randn(dim1, dim2, device="cuda").half()
            out_col1, Scol = self.vectorwise_quant(A, dim=0)
            out_row1, Srow = self.vectorwise_quant(A, dim=1)

            CA, CAt, statsA, statsAt, _ = F.int8_double_quant(A)

            # max difference is 1 due to rounding differences
            torch.testing.assert_close(CA, out_row1, atol=1, rtol=0)
            torch.testing.assert_close(CAt, out_col1, atol=1, rtol=0)

            n = CAt.numel()
            num_not_close_rows = (torch.isclose(CA, out_row1, atol=1) == 0).sum().item()
            num_not_close_cols = (torch.isclose(CAt, out_col1, atol=1) == 0).sum().item()

            # allow for 1:500 error due to rounding differences
            min_error = 1 / 500
            if num_not_close_cols > (min_error * n):
                print(
                    f"Min error exceeded {num_not_close_cols} elements are different. Error: {num_not_close_cols / n:.4f}"
                )
                assert False
            if num_not_close_rows > (min_error * n):
                print(
                    f"Min error exceeded {num_not_close_rows} elements are different. Error: {num_not_close_rows / n:.4f}"
                )
                assert False

            torch.testing.assert_close(Srow.flatten().float(), statsA)
            torch.testing.assert_close(Scol.flatten().float(), statsAt)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        ("dim1", "dim4", "inner"),
        (
            pytest.param(dim1, dim4, inner, id=f"{dim1=},{dim4=},{inner=}")
            for (dim1, dim4, inner) in zip(
                (1, 8, 2048, 4096),
                (2, 128, 2048, 4096),
                (4, 256, 512, 4096),
            )
        ),
    )
    def test_integrated_int8_linear_matmul(self, device, dim1, dim4, inner):
        if device == "cpu" and inner > 2048:
            pytest.skip("Slow on CPU")

        for i in range(k):
            A = torch.randn(dim1, inner, device=device).half()
            B = torch.randn(dim4, inner, device=device).half()

            out1 = torch.matmul(A.half(), B.t().half())

            C1a, stats1a, _ = F.int8_vectorwise_quant(A)
            C2a, stats2a, _ = F.int8_vectorwise_quant(B)
            A1, maxA = self.vectorwise_quant(A, dim=1)
            B1, maxB = self.vectorwise_quant(B, dim=1)

            torch.testing.assert_close(maxA.flatten().float(), stats1a)
            torch.testing.assert_close(maxB.flatten().float(), stats2a)
            torch.testing.assert_close(C1a, A1, rtol=0, atol=1)
            torch.testing.assert_close(C2a, B1, rtol=0, atol=1)

            out2 = F.int8_linear_matmul(A1, B1)

            C2 = F.int8_linear_matmul(A1, B1)

            out3 = self.vectorwise_mm_dequant(C2.float(), maxA, maxB.t())

            err1 = torch.abs(out1 - out2).mean().item()
            err2 = torch.abs(out1 - out3).mean().item()
            assert err2 <= err1 * 1.025

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dim1", [512, 2048], ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim2", [1024, 4096], ids=id_formatter("dim2"))
    def test_coo_double_quant(self, device, dim1, dim2):
        threshold = 2.00
        for i in range(k):
            A = torch.randn(dim1, dim2, device=device).half()

            idx = torch.abs(A) >= threshold
            CA, statsA, outlier_cols = F.int8_vectorwise_quant(A, threshold=threshold)

            if outlier_cols is not None:
                A1 = A * idx
                A2 = torch.zeros_like(A) + A1
                torch.testing.assert_close(A1, A2)

                A[:, outlier_cols] = 0
                A2 = (CA.float() * statsA.unsqueeze(1) / 127).half()
                torch.testing.assert_close(A, A2, rtol=0.05, atol=1.5e-2)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dim1", [512, 2048], ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim2", [1024, 4096], ids=id_formatter("dim2"))
    def test_coo_int8_vectorwise_quant(self, device, dim1, dim2):
        threshold = 3.00
        for i in range(k):
            A = torch.randn(dim1, dim2, device=device).half()

            idx = torch.abs(A) >= threshold
            CA, statsA, outlier_cols = F.int8_vectorwise_quant(A, threshold=threshold)

            if outlier_cols is not None:
                A2 = (CA.float() * statsA.unsqueeze(1) / 127).half()
                A[:, outlier_cols] = 0
                torch.testing.assert_close(A * (idx == 0), A2, rtol=0.05, atol=1.5e-2)


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestSpMMFunctional:
    @pytest.mark.parametrize("dim1", [256, 1024], ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim2", [128, 512], ids=id_formatter("dim2"))
    @pytest.mark.parametrize("transposed_B", TRUE_FALSE, ids=id_formatter("transposed_B"))
    def test_spmm_coo(self, dim1, dim2, transposed_B):
        threshold = 1.5
        dim3 = torch.randint(32, 128, size=(1,)).item()
        # dim3 = 17
        for i in range(k):
            A = torch.randn(dim1, dim2).cuda().half()
            if transposed_B:
                B = torch.randn(dim3, dim2).cuda().half()
            else:
                B = torch.randn(dim2, dim3).cuda().half()

            idx = torch.abs(A) >= threshold
            nnz = (idx == 1).sum().item()
            rows, cols = torch.where(idx)
            values = A[idx]
            cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
            A2 = A * idx

            if transposed_B:
                out2 = F.spmm_coo(cooA, B.t())
                out1 = torch.matmul(A2, B.t())
            else:
                out2 = F.spmm_coo(cooA, B)
                out1 = torch.matmul(A2, B)

            assert_all_approx_close(out1, out2, rtol=0.01, atol=3.0e-2, count=30)

    @pytest.mark.benchmark
    def test_spmm_bench(self):
        batch = 2
        model = 1024 * 1
        hidden = model * 4
        seq = 1024
        dim1 = batch * seq
        dim2 = model
        dim3 = hidden
        threshold = 4
        A = torch.randn(dim1, dim2, device="cuda").half()
        B = torch.randn(dim2, dim3, device="cuda").half()
        for i in range(10):
            C1 = bnb.matmul(A, B.t())

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(k):
            C1 = bnb.matmul(A, B.t())
        torch.cuda.synchronize()
        t8 = time.time() - t0

        idx = torch.abs(A) >= threshold
        nnz = (idx == 1).sum().item()
        print(nnz / idx.numel())
        rows, cols = torch.where(idx)
        values = A[idx]
        cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)

        for i in range(10):
            out2 = F.spmm_coo(cooA, B)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(k):
            out2 = F.spmm_coo(cooA, B)
        torch.cuda.synchronize()
        tsp = time.time() - t0
        print(tsp, t8)
        print(tsp / t8)

    @pytest.mark.parametrize("dim1", [1 * 2048], ids=id_formatter("dim1"))
    @pytest.mark.parametrize("dim2", [12288], ids=id_formatter("dim2"))
    @pytest.mark.parametrize("dtype", [torch.float16], ids=describe_dtype)
    @pytest.mark.parametrize("out_func", ["zeros", "ones"], ids=id_formatter("out_func"))
    def test_spmm_coo_very_sparse(self, dim1, dim2, dtype, out_func):
        out_func = getattr(torch, out_func)

        threshold = 3.3
        # threshold = 2.8
        # threshold = 0.0
        A = torch.randn(dim1, dim2, device="cuda").half()
        if dtype == torch.float16:
            B = torch.randn(dim2, dim2 * 4, device="cuda").half()
            torch.nn.init.xavier_uniform_(B)
        else:
            B = torch.randn(dim2, dim2 * 4, device="cuda").half()
            torch.nn.init.xavier_uniform_(B)

            SB = torch.abs(B).max().float()
            B = torch.round(B / SB * 127).to(torch.int8)

        print("")
        idx = torch.abs(A) >= threshold
        nnz = (idx == 1).sum().item()
        rows, cols = torch.where(idx)
        values = A[idx]
        cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
        A2 = A * idx
        out1 = torch.matmul(A2.half(), B.half())
        out = out_func(out1.shape, dtype=torch.float16, device=out1.device)
        out1 += out.clone()
        out2 = F.spmm_coo_very_sparse(cooA, B, out=out)
        # print(B)
        # print(out1)
        # print(out2)
        p = 200 / (2048 * 12288 * 4)
        n = out1.numel()
        count = math.ceil(p * n)
        std = out1.std()
        out1 /= std
        out2 /= std
        assert_all_approx_close(out1, out2.half(), rtol=0.01, atol=3.0e-2, count=count)
        # assert_all_approx_close(out1, out2.half(), rtol=0.05, atol=0.01, count=count)

        idx_col = torch.randint(0, A2.shape[-1], size=(15,))

        # torch.testing.assert_close(out1, out2.half(), rtol=0.05, atol=0.001)

        # Bt = torch.randn(dim2*4, dim2, device='cuda').half()
        # torch.cuda.synchronize()
        # t0 = time.time()
        # print(A2.shape, B.shape)
        # for i in range(100):
        #   #out3 = F.spmm_coo(cooA, Bt.t())
        #   #out2 = F.spmm_coo(cooA, B)
        #   #out2 = F.spmm_coo_very_sparse(cooA, B)
        #   #out1 = torch.matmul(A, Bt.t())

        # torch.cuda.synchronize()
        # print(time.time() - t0)

    @pytest.mark.parametrize("dim1", [1 * 2048])
    @pytest.mark.parametrize("dim2", [2048])
    @pytest.mark.parametrize("dtype", [torch.int8])
    def test_spmm_coo_dequant(self, dim1, dim2, dtype):
        threshold = 6.0
        # threshold = 2.8
        # threshold = 0.0
        A = torch.randn(dim1, dim2, device="cuda").half()
        B = torch.empty(dim2, dim2 * 4, device="cuda", dtype=torch.float16)
        torch.nn.init.xavier_uniform_(B)
        Bt = B.t().contiguous()

        CB, CBt, statsB, statsBt, coo_tensor = F.int8_double_quant(B)

        rowidx = torch.randint(0, A.shape[-1], size=(15,))

        A[:, rowidx] = 8.0

        idx = torch.abs(A) >= threshold
        nnz = (idx == 1).sum().item()
        rows, cols = torch.where(idx)
        values = A[idx]
        cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
        A2 = A * idx
        out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
        out1 = torch.matmul(A2, B.half())
        out3 = F.spmm_coo_very_sparse(cooA, CBt.half())
        out3 = out3 * statsBt.half() / 127

        values, counts = torch.unique(cooA.rowidx, return_counts=True)
        offset = counts.cumsum(0).int()
        max_count, max_idx = torch.sort(counts, descending=True)
        print(torch.median(max_count.float()))

        torch.testing.assert_close(out2, out3, rtol=0.05, atol=0.001)

        p = 200 / (2048 * 12288 * 4)
        n = out1.numel()
        count = math.ceil(p * n)
        assert_all_approx_close(out1, out2, rtol=0.01, atol=3.0e-2, count=count)

        # torch.cuda.synchronize()
        # t0 = time.time()
        # for i in range(100):
        #   out2 = F.spmm_coo_very_sparse(cooA, B)
        # torch.cuda.synchronize()
        # print('fp16', time.time() - t0)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(100):
            out2 = F.spmm_coo(cooA, B)
        torch.cuda.synchronize()
        print("cusparse fp16", time.time() - t0)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(100):
            out2 = F.spmm_coo_very_sparse(cooA, CBt)
        torch.cuda.synchronize()
        print("int8", time.time() - t0)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(100):
            out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
        torch.cuda.synchronize()
        print("int8+dequant", time.time() - t0)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(100):
            out2 = torch.matmul(A, B)
        torch.cuda.synchronize()
        print("matmul", time.time() - t0)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(100):
            out1 = bnb.matmul(A, Bt)
            out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
            out = out1 + out2
        torch.cuda.synchronize()
        print("sparse+ matmul", time.time() - t0)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(100):
            out1 = bnb.matmul(A, Bt)
            torch.matmul(A[:, rowidx], Bt.t()[rowidx], out=out1)
        torch.cuda.synchronize()
        print("partial matmul", time.time() - t0)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(100):
            out1 = bnb.matmul(A, Bt)
        torch.cuda.synchronize()
        print("partial matmul", time.time() - t0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestSparseTensorFunctional:
    def test_coo2csr(self):
        threshold = 1
        A = torch.randn(128, 128).half().cuda()
        idx = torch.abs(A) >= threshold
        nnz = (idx == 1).sum().item()
        rows, cols = torch.where(idx)
        values = A[idx]
        cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
        A2 = A * idx
        csrA = F.coo2csr(cooA)
        counts = csrA.rowptr[1:] - csrA.rowptr[:-1]
        assert counts.numel() == A.shape[0]

        torch.testing.assert_close(counts.long(), (A2 != 0).sum(1))
        idx = A2 != 0
        torch.testing.assert_close(A2[idx], csrA.values)

    def test_coo2csc(self):
        threshold = 1
        A = torch.randn(128, 128).half().cuda()
        idx = torch.abs(A) >= threshold
        nnz = (idx == 1).sum().item()
        rows, cols = torch.where(idx)
        values = A[idx]
        cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
        A2 = A * idx
        cscA = F.coo2csc(cooA)
        counts = cscA.colptr[1:] - cscA.colptr[:-1]
        assert counts.numel() == A.shape[1]

        torch.testing.assert_close(counts.long(), (A2 != 0).sum(0))
        # torch uses row-major -> use transpose to transfer to col-major
        idx = A2.t() != 0
        torch.testing.assert_close(A2.t()[idx], cscA.values)


class TestQuantize4BitFunctional:
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize(
        "blocksize",
        [64, 128, 256, 512, 1024, 2048, 4096] if not HIP_ENVIRONMENT else [128, 256, 512, 1024, 2048, 4096],
    )
    def test_4bit_quant(self, device, dtype, quant_type, blocksize):
        if device == "hpu" and not is_supported_on_hpu(quant_type, dtype):
            pytest.skip("This configuration is not supported on HPU.")

        A1 = torch.randn(1024, 1024, device=device, dtype=dtype)
        qa, SA = F.quantize_4bit(A1, blocksize=blocksize, quant_type=quant_type)
        d = SA.as_dict()
        SA = F.QuantState.from_dict(d, device=torch.device(device))
        A2 = F.dequantize_4bit(qa, SA, blocksize=blocksize, quant_type=quant_type)

        err = (A1 - A2).abs().float()
        relerr = (err / (A1.abs().float() + 1e-8)).mean()
        err = err.mean()

        assert A2.dtype == dtype

        # With larger block sizes, we can expect this to blow up.
        # At blocksize>=1024, don't even bother looking at relerr.
        #
        # Actually, the above is not true anymore after fixing the integer packing bug.
        # The following values were taken from averaging 1k samples per test configuration after fixing the bug.
        error_dict = dict()
        error_dict["fp4"] = dict()
        error_dict["nf4"] = dict()
        error_dict["fp4"]["err"] = {
            64: 0.096545,
            128: 0.102947,
            256: 0.108685,
            512: 0.114087,
            1024: 0.119312,
            2048: 0.124460,
            4096: 0.129573,
        }
        error_dict["fp4"]["rel_err"] = {
            64: 0.260130,
            128: 0.275734,
            256: 0.289842,
            512: 0.302852,
            1024: 0.314982,
            2048: 0.326402,
            4096: 0.337228,
        }

        error_dict["nf4"]["err"] = {
            64: 0.072792,
            128: 0.076835,
            256: 0.080326,
            512: 0.083535,
            1024: 0.086603,
            2048: 0.089592,
            4096: 0.092537,
        }
        error_dict["nf4"]["rel_err"] = {
            64: 0.203299,
            128: 0.215252,
            256: 0.226044,
            512: 0.236021,
            1024: 0.245365,
            2048: 0.254146,
            4096: 0.262457,
        }

        assert err < error_dict[quant_type]["err"][blocksize] + 1e-3
        assert relerr < error_dict[quant_type]["rel_err"][blocksize] + 1e-3

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    @pytest.mark.parametrize("blocksize", [64, 128] if not HIP_ENVIRONMENT else [128], ids=id_formatter("blocksize"))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=describe_dtype)
    def test_4bit_compressed_stats(self, device, quant_type, blocksize, dtype):
        if device == "hpu" and not is_supported_on_hpu(quant_type, dtype):
            pytest.skip("FP4 quantization is not supported on HPU.")

        errs1 = []
        errs2 = []
        for i in range(10):
            A1 = torch.randn(1024, 1024, device=device, dtype=dtype)
            q2, SA2 = F.quantize_4bit(A1, blocksize=blocksize, quant_type=quant_type)
            q3, SA3 = F.quantize_4bit(A1, blocksize=blocksize, compress_statistics=True, quant_type=quant_type)
            A2 = F.dequantize_4bit(q2, SA2, quant_type=quant_type)
            A3 = F.dequantize_4bit(q3, SA3, quant_type=quant_type)

            err = (A1 - A2).abs().float()
            relerr = (err / (A1.abs().float() + 1e-15)).mean()
            err = err.mean()

            errs1.append(err.item())

            assert err.item() < 0.11
            assert relerr.item() < 0.28

            err = (A1 - A3).abs().float()
            relerr = (err / (A1.abs().float() + 1e-15)).mean()
            err = err.mean()

            errs2.append(err.item())

            assert err.item() < 0.11
            assert relerr.item() < 0.28

    # @pytest.mark.parametrize("quant_type", ['fp4', 'nf4'])
    @pytest.mark.parametrize("quant_type", ["nf4"])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.benchmark
    def test_bench_4bit_dequant(self, quant_type):
        blocksize = 256
        a = torch.rand(1024 * 12 * 4, 1024 * 12, device="cuda").half()
        qa, SA = F.quantize_4bit(a, blocksize=blocksize, quant_type=quant_type)

        input_size = a.numel() / 2
        output_size = a.numel() * 2
        num_bytes = input_size + output_size
        GB = num_bytes / 1e9
        max_theoretical_s = GB / 768
        # print(max_theoretical_s*1e6)
        b = torch.randn(128, 1024 * 12, device="cuda").half()

        iters = 100
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(iters):
            F.dequantize_4bit(qa, SA, blocksize=blocksize, quant_type=quant_type)
            # b.copy_(a)
        torch.cuda.synchronize()
        # print((time.time()-t0)/iters*1e6)

        # torch.cuda.synchronize()
        # t0 = time.time()
        # for i in range(iters):
        #    torch.matmul(b, a.t())
        # torch.cuda.synchronize()
        # print((time.time()-t0)/iters*1e6)

    @pytest.mark.skipif(
        HIP_ENVIRONMENT, reason="gemv 4bit tests are partially enabled on MI300, others being fixed for warpsize 64"
    )
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("double_quant", TRUE_FALSE, ids=lambda double_quant: f"DQ_{double_quant}")
    @pytest.mark.parametrize("storage_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("kind", ["fc1", "fc2", "attn", "attn_packed"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
    @pytest.mark.parametrize(
        "quant_storage",
        [torch.uint8, torch.float16, torch.bfloat16, torch.float32],
        ids=describe_dtype,
    )
    @pytest.mark.parametrize("dim", [128, 256, 512, 1024], ids=id_formatter("dim"))
    def test_gemv_4bit(self, device, dim, dtype, storage_type, quant_storage, double_quant, kind):
        if device == "hpu" and not is_supported_on_hpu(storage_type, dtype, quant_storage):
            pytest.skip("This configuration is not supported on HPU.")

        errs1 = []
        errs2 = []
        errs3 = []
        relerrs1 = []
        relerrs2 = []
        relerrs3 = []
        max_errs1 = []
        max_errs2 = []
        max_errs3 = []

        # Large number of iterations is excessive and slow on CPU.
        # Keep for CUDA for now.
        iters = 100 if device == "cuda" else 10

        for i in range(iters):
            if kind == "fc1":
                A = torch.randn(1, dim, dtype=dtype, device=device)
                B = torch.randn(dim * 4, dim, dtype=dtype, device=device) / math.sqrt(dim)
            elif kind == "fc2":
                A = torch.randn(1, 4 * dim, dtype=dtype, device=device)
                B = torch.randn(dim, 4 * dim, dtype=dtype, device=device) / math.sqrt(dim)
            elif kind == "attn":
                A = torch.randn(1, dim, dtype=dtype, device=device)
                B = torch.randn(dim, dim, dtype=dtype, device=device) / math.sqrt(dim)
            elif kind == "attn_packed":
                A = torch.randn(1, dim, dtype=dtype, device=device)
                B = torch.randn(dim * 3, dim, dtype=dtype, device=device) / math.sqrt(dim)

            qB, state = F.quantize_4bit(
                B,
                quant_type=storage_type,
                compress_statistics=double_quant,
                quant_storage=quant_storage,
            )
            C3 = torch.matmul(A, B.t())
            C2 = F.gemv_4bit(A, qB.t(), state=state)
            A.requires_grad = True
            C1 = bnb.matmul_4bit(A, qB.t(), state)

            err1 = (C1 - C2).abs().float()
            err2 = (C3 - C2).abs().float()
            err3 = (C3 - C1).abs().float()

            mag1 = torch.abs(C1).float() + 1e-5
            mag2 = torch.abs(C3).float() + 1e-5
            mag3 = torch.abs(C3).float() + 1e-5

            relerr1 = err1 / mag1
            relerr2 = err2 / mag2
            relerr3 = err3 / mag3

            max_err1 = err1.max()
            max_err2 = err2.max()
            max_err3 = err3.max()

            errs1.append(err1.mean().item())
            errs2.append(err2.mean().item())
            errs3.append(err3.mean().item())

            relerrs1.append(relerr1.mean().item())
            relerrs2.append(relerr2.mean().item())
            relerrs3.append(relerr3.mean().item())

            max_errs1.append(max_err1.item())
            max_errs2.append(max_err2.item())
            max_errs3.append(max_err3.item())

            c = int(C1.numel() * 0.0014 * (dim / 256)) + 1

            c = assert_all_approx_close(C1, C2, 1e-5, 0.01, count=0, throw=False)
        err1 = sum(errs1) / len(errs1) / math.sqrt(dim)
        err2 = sum(errs2) / len(errs2) / math.sqrt(dim)
        err3 = sum(errs3) / len(errs3) / math.sqrt(dim)
        relerr1 = sum(relerrs1) / len(relerrs1) / math.sqrt(dim)
        relerr2 = sum(relerrs2) / len(relerrs2) / math.sqrt(dim)
        relerr3 = sum(relerrs3) / len(relerrs3) / math.sqrt(dim)
        maxerr1 = sum(max_errs1) / len(max_errs1) / math.sqrt(dim)
        maxerr2 = sum(max_errs2) / len(max_errs2) / math.sqrt(dim)
        maxerr3 = sum(max_errs3) / len(max_errs3) / math.sqrt(dim)
        absratio = err2 / err3
        relratio = relerr2 / relerr3
        maxratio = relerr2 / relerr3

        # for debugging if the tests fails
        #
        # print('='*80)
        # print(f'For matmul: {A.shape}, {B.shape}, {kind}, {dtype}, {storage_type}, double_quant={double_quant}:')
        # print(C1.flatten()[-20:])
        # print(C2.flatten()[-20:])
        # print(f'inference vs training abs: {err1}')
        # print(f'inference vs training rel: {relerr1}')
        # print(f'inference vs training max: {maxerr1}')
        # print(f'inference vs training vs torch err ratio abs: {absratio}')
        # print(f'inference vs training vs torch err ratio rel: {relratio}')
        # print(f'inference vs training vs torch err ratio max: {maxratio}')
        if dtype == torch.float16:
            if dim <= 512:
                assert err1 < 7e-5

                # TODO(matthewdouglas): On T4, dim=128-fp16-fc2-fp4-DQ will have relerror ~ 0.00092727
                if (
                    device == "cuda"
                    and double_quant
                    and storage_type == "fp4"
                    and kind == "fc2"
                    and torch.cuda.get_device_capability() == (7, 5)
                ):
                    assert relerr1 < 0.00093
                else:
                    assert relerr1 < 0.0008
            else:
                assert err1 < 6e-5
                assert relerr1 < 2e-4
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.005 and relratio > 0.995
            assert maxratio < 1.005 and maxratio > 0.995
        elif dtype == torch.float32:
            if dim <= 512:
                assert err1 < 5e-8
                assert relerr1 < 1e-6
                assert maxerr1 < 1e-7
            else:
                assert err1 < 5e-8
                assert relerr1 < 8e-6
                assert maxerr1 < 1e-7
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.005 and relratio > 0.995
            assert maxratio < 1.005 and maxratio > 0.995
        elif dtype == torch.bfloat16:
            if dim <= 512:
                assert err1 < 6e-4
                assert relerr1 < 0.007
                assert maxerr1 < 0.015
            else:
                assert err1 < 2e-4
                assert relerr1 < 0.002
                assert maxerr1 < 0.0012
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.04 and relratio > 0.96
            assert maxratio < 1.02 and maxratio > 0.98

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("storage_type", ["nf4", "fp4"], ids=["nf4", "fp4"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
    @pytest.mark.parametrize("double_quant", [False], ids=["DQ_True"])
    @pytest.mark.skipif(
        HIP_ENVIRONMENT and ROCM_GPU_ARCH == "gfx90a",
        reason="this test is not supported on ROCm with gfx90a architecture yet",
    )
    def test_gemv_eye_4bit(self, device, storage_type, dtype, double_quant):
        if device == "cpu" and dtype == torch.bfloat16 and torch.__version__ < (2, 3):
            pytest.skip("eye doe not support bfloat16 on CPU in torch < 2.3")

        if device == "hpu" and not is_supported_on_hpu(storage_type, dtype):
            pytest.skip("This configuration is not supported on HPU.")

        dims = 10
        torch.random.manual_seed(np.random.randint(0, 412424242))
        dims = get_test_dims(0, 8192, n=dims)
        dims = [dim + (64 - (dim % 64)) for dim in dims]
        # for dim in [576, 5120, 3520, 5184, 1280, 4992, 5312, 2048]:
        for dim in dims:
            A = torch.normal(0, 0.1, size=(1, 1, dim), dtype=dtype, device=device)
            B = torch.eye(dim, dtype=dtype, device=device)

            qB, state = F.quantize_4bit(B, quant_type=storage_type, compress_statistics=double_quant)
            C3 = torch.matmul(A, B.t())
            C2 = bnb.matmul_4bit(A, qB.t(), state)
            A.requires_grad = True
            C1 = bnb.matmul_4bit(A, qB.t(), state)

            torch.testing.assert_close(A, C3)
            torch.testing.assert_close(A, C1)
            torch.testing.assert_close(A, C2)
        # torch.testing.assert_close(A, C1, rtol=1e-5, atol=0.00001)
        # torch.testing.assert_close(A, C2, rtol=1e-5, atol=0.080)


def test_normal_map_tree():
    code = F.create_normal_map()
    values = code[:8].tolist() + code[-8:].tolist()
    num_pivots = 1
    # print(values)
    while num_pivots < 16:
        idx = list(range(16 // num_pivots // 2, 16, 16 // num_pivots))
        # print(idx)
        num_pivots *= 2
        pivots = []
        for i in idx:
            pivots.append((values[i - 1] + values[i]) / 2)
        # print(pivots)
