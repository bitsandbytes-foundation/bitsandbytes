import ctypes
import os
import shutil
import time
import uuid
from itertools import product
from os.path import join

import pytest
from lion_pytorch import Lion

import torch

import bitsandbytes as bnb
import bitsandbytes.functional as F


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import enable_wrap, wrap

# import apex

k = 20

def assert_most_approx_close(a, b, rtol=1e-3, atol=1e-3, max_error_count=0):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()
    if error_count > max_error_count:
        print(f"Too many values not close: assert {error_count} < {max_error_count}")
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def get_temp_dir():
    path = f"/tmp/autoswap/{str(uuid.uuid4())}"
    os.makedirs(path, exist_ok=True)
    return path


def rm_path(path):
    shutil.rmtree(path)

str2optimizers = {}
str2optimizers["adam_pytorch"] = (None, torch.optim.Adam, bnb.optim.Adam)
str2optimizers["lion_pytorch"] = (None, Lion, bnb.optim.Lion)
str2optimizers["momentum_pytorch"] = (
    None,
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    bnb.optim.Adam,
)
str2optimizers["adam"] = (torch.optim.Adam, bnb.optim.Adam)
str2optimizers["paged_adamw"] = (torch.optim.AdamW, bnb.optim.PagedAdamW)
str2optimizers["paged_adam"] = (torch.optim.Adam, bnb.optim.PagedAdam)
str2optimizers["lion"] = (Lion, bnb.optim.Lion)
str2optimizers["paged_lion"] = (Lion, bnb.optim.PagedLion)
str2optimizers["momentum"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.SGD(pxx, 0.01, 0.9, block_wise=True),
)
str2optimizers["rmsprop"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.RMSprop(pxx, 0.01, 0.9, block_wise=True),
)

str2optimizers["adam8bit_blockwise"] = (torch.optim.Adam, lambda pxx: bnb.optim.Adam8bit(pxx, block_wise=True))
str2optimizers["paged_adamw8bit_blockwise"] = (torch.optim.AdamW, lambda pxx: bnb.optim.PagedAdamW8bit(pxx, block_wise=True))
str2optimizers["paged_adam8bit_blockwise"] = (torch.optim.Adam, lambda pxx: bnb.optim.PagedAdam8bit(pxx, block_wise=True))
str2optimizers["lion8bit_blockwise"] = (Lion, lambda pxx: bnb.optim.Lion8bit(pxx, block_wise=True))
str2optimizers["paged_lion8bit_blockwise"] = (Lion, lambda pxx: bnb.optim.PagedLion8bit(pxx, block_wise=True))
str2optimizers["momentum8bit_blockwise"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.SGD8bit(pxx, 0.01, 0.9, block_wise=True),
)
str2optimizers["rmsprop8bit_blockwise"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.RMSprop8bit(pxx, 0.01, 0.9, block_wise=True),
)

str2statenames = {}
str2statenames["adam"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["paged_adamw"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["paged_adam"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["lion"] = [("exp_avg", "state1")]
str2statenames["paged_lion"] = [("exp_avg", "state1")]
str2statenames["momentum"] = [("momentum_buffer", "state1")]
str2statenames["lamb"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["rmsprop"] = [("square_avg", "state1")]
str2statenames["adam8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1"), ("exp_avg_sq", "state2", "qmap2", "absmax2")]
str2statenames["paged_adam8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1"), ("exp_avg_sq", "state2", "qmap2", "absmax2")]
str2statenames["paged_adamw8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1"), ("exp_avg_sq", "state2", "qmap2", "absmax2")]
str2statenames["momentum8bit_blockwise"] = [("momentum_buffer", "state1", "qmap1", "absmax1")]
str2statenames["rmsprop8bit_blockwise"] = [("square_avg", "state1", "qmap1", "absmax1")]
str2statenames["lion8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1")]
str2statenames["paged_lion8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1")]

dim1 = [1024]
dim2 = [32, 1024, 4097, 1]
gtype = [torch.float32, torch.float16, torch.bfloat16]
optimizer_names = ["adam", "momentum", "rmsprop", 'paged_adamw', 'paged_adam', 'lion', 'paged_lion']
values = list(product(dim1, dim2, gtype, optimizer_names))
names = ["dim1_{}_dim2_{}_gtype_{}_optim_{}".format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name", values, ids=names)
def test_optimizer32bit(dim1, dim2, gtype, optim_name):
    if gtype == torch.bfloat16 and optim_name in ['momentum', 'rmsprop']: pytest.skip()
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1
    p2 = p1.clone()
    p1 = p1.float()

    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    elif gtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
    else:
        atol, rtol = 1e-4, 1e-3

    for i in range(k):
        g = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.01
        p1.grad = g.clone().float()
        p2.grad = g.clone()

        bnb_optimizer.step()
        torch_optimizer.step()


        for name1, name2 in str2statenames[optim_name]:
            torch.testing.assert_close(
                torch_optimizer.state[p1][name1],
                bnb_optimizer.state[p2][name2].cuda(),
                atol=atol,
                rtol=rtol,
            )

        # since Lion can have pretty noisy updates where things lie at the boundary
        # allow up to 10 errors for Lion
        assert_most_approx_close(p1, p2.float(), atol, rtol, max_error_count=10)

        if i % (k // 5) == 0 and i > 0:
            path = get_temp_dir()
            torch.save(bnb_optimizer.state_dict(), join(path, "opt.pt"))
            del bnb_optimizer
            bnb_optimizer = None
            bnb_optimizer = str2optimizers[optim_name][1]([p2])
            bnb_optimizer.load_state_dict(torch.load(join(path, "opt.pt")))
            rm_path(path)
            # since Lion can have pretty noisy updates where things lie at the boundary
            # allow up to 10 errors for Lion
            assert_most_approx_close(p1, p2.float(), atol, rtol, max_error_count=10)
            for name1, name2 in str2statenames[optim_name]:
                # since Lion can have pretty noisy updates where things lie at the boundary
                # allow up to 10 errors for Lion
                assert_most_approx_close(torch_optimizer.state[p1][name1], bnb_optimizer.state[p2][name2],
                                         atol=atol, rtol=rtol,
                                         max_error_count=10)

        if gtype != torch.float32:
            # the adam buffers should also be close because they are 32-bit
            # but the paramters can diverge because they are 16-bit
            # the difference grow larger and larger with each update
            # --> copy the state to keep weights close
            p1.data = p1.data.to(p2.dtype).float()
            p2.copy_(p1.data)
            torch.testing.assert_close(p1.to(p2.dtype), p2)
        if optim_name in ["lars", "lamb"]:
            assert bnb_optimizer.state[p2]["unorm_vec"] > 0.0


dim1 = [1024]
dim2 = [32, 1024, 4097]
gtype = [torch.float32, torch.float16]
values = list(product(dim1, dim2, gtype))
names = ["dim1_{}_dim2_{}_gtype_{}".format(*vals) for vals in values]


@pytest.mark.parametrize("dim1, dim2, gtype", values, ids=names)
def test_global_config(dim1, dim2, gtype):
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    p2 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    p3 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    mask = torch.rand_like(p2) < 0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8

    bnb.optim.GlobalOptimManager.get_instance().initialize()
    bnb.optim.GlobalOptimManager.get_instance().override_config(
        p3, "optim_bits", 8
    )

    bnb.optim.GlobalOptimManager.get_instance().register_parameters(
        [p1, p2, p3]
    )
    p1 = p1.cuda()
    p2 = p2.cuda()
    p3 = p3.cuda()

    adam2 = bnb.optim.Adam([p1, p2, p3], lr, (beta1, beta2), eps)

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    else:
        atol, rtol = 1e-4, 1e-3

    for i in range(50):
        g1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + 0.001
        g2 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + 0.001
        g3 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + 0.001
        p1.grad = g1
        p2.grad = g2
        p3.grad = g3

        adam2.step()

        assert adam2.state[p3]["state1"].dtype == torch.uint8
        assert adam2.state[p3]["state2"].dtype == torch.uint8


dim1 = [1024]
dim2 = [32, 1024, 4097]
gtype = [torch.float32, torch.float16, torch.bfloat16]
optimizer_names = [
    "adam8bit_blockwise",
    "lion8bit_blockwise",
    "momentum8bit_blockwise",
    "rmsprop8bit_blockwise",
]
values = list(product(dim1, dim2, gtype, optimizer_names))
names = [
    "dim1_{}_dim2_{}_gtype_{}_optim_{}".format(*vals) for vals in values
]


@pytest.mark.parametrize("dim1, dim2, gtype, optim_name", values, ids=names)
def test_optimizer8bit(dim1, dim2, gtype, optim_name):
    if gtype == torch.bfloat16 and optim_name not in ['adam8bit_blockwise', 'lion8bit_blockwise']: pytest.skip()
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1
    p2 = p1.clone()
    p1 = p1.float()
    blocksize = 256

    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])

    if gtype == torch.float32:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3
    elif gtype == torch.bfloat16:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-4, 1e-2
    else:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3

    errors = []
    relerrors = []

    for i in range(100):
        g = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.01
        p1.grad = g.clone().float()
        p2.grad = g.clone()

        bnb_optimizer.step()
        torch_optimizer.step()

        # since Lion can have pretty noisy updates where things lie at the boundary
        # allow up to 5 errors for Lion
        assert_most_approx_close(p1, p2.float(), patol, prtol, max_error_count=5)

        dequant_states = []
        for name1, name2, qmap, max_val in str2statenames[optim_name]:
            # print(bnb_optimizer.state[p2][max_val], name1)
            if "blockwise" in optim_name:
                s1 = F.dequantize_blockwise(
                    code=bnb_optimizer.state[p2][qmap],
                    absmax=bnb_optimizer.state[p2][max_val],
                    A=bnb_optimizer.state[p2][name2],
                    blocksize=blocksize,
                )
            else:
                s1 = F.dequantize(
                    code=bnb_optimizer.state[p2][qmap],
                    absmax=bnb_optimizer.state[p2][max_val],
                    A=bnb_optimizer.state[p2][name2],
                )
            num_not_close = (
                torch.isclose(
                    torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol
                )
                == 0
            )
            assert num_not_close.sum().item() < 20
            dequant_states.append(s1.clone())

        err = torch.abs(p1 - p2)
        relerr = err / (torch.abs(p1)+1e-9)
        if g.dtype == torch.bfloat16:
            assert err.mean() < 0.00015
            assert relerr.mean() < 0.0016
        else:
            assert err.mean() < 0.00012
            assert relerr.mean() < 0.0012

        errors.append(err.mean().item())
        relerrors.append(relerr.mean().item())

        if i % 10 == 0 and i > 0:
            for (name1, name2, qmap, max_val), s in zip(
                str2statenames[optim_name], dequant_states
            ):
                s1cpy = s.clone()
                raws1cpy = bnb_optimizer.state[p2][name2].clone()
                qmap1 = bnb_optimizer.state[p2][qmap].clone()

                path = get_temp_dir()
                torch.save(bnb_optimizer.state_dict(), join(path, "opt.pt"))
                del bnb_optimizer
                bnb_optimizer = None
                bnb_optimizer = str2optimizers[optim_name][1]([p2])
                bnb_optimizer.load_state_dict(torch.load(join(path, "opt.pt")))
                rm_path(path)
                torch.testing.assert_close(raws1cpy, bnb_optimizer.state[p2][name2])
                torch.testing.assert_close(qmap1, bnb_optimizer.state[p2][qmap])

                if "blockwise" in optim_name:
                    s1 = F.dequantize_blockwise(
                        code=bnb_optimizer.state[p2][qmap],
                        absmax=bnb_optimizer.state[p2][max_val],
                        A=bnb_optimizer.state[p2][name2],
                        blocksize=blocksize,
                    )
                else:
                    s1 = F.dequantize(
                        code=bnb_optimizer.state[p2][qmap],
                        absmax=bnb_optimizer.state[p2][max_val],
                        A=bnb_optimizer.state[p2][name2],
                    )
                torch.testing.assert_close(s1cpy, s1)

                num_not_close = (torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol) == 0)
                assert num_not_close.sum().item() < 20
            # since Lion can have pretty noisy updates where things lie at the boundary
            # allow up to 5 errors for Lion
            assert_most_approx_close(p1, p2.float(), patol, prtol, max_error_count=5)

        # the parameters diverge quickly. Here we keep them close
        # together so we can test against the Adam error
        p1.data = p1.data.to(gtype).float()
        p2.copy_(p1.data)
        torch.testing.assert_close(p1.to(gtype), p2)
        for (name1, name2, qmap, max_val), s in zip(str2statenames[optim_name], dequant_states):
            torch_optimizer.state[p1][name1].copy_(s.data)

    # print(sum(errors)/len(errors))
    # print(sum(relerrors)/len(relerrors))


dim1 = [1024]
dim2 = [32, 1024, 4097]
gtype = [torch.float32]
optim_bits = [32, 8]
values = list(product(dim1, dim2, gtype, optim_bits))
names = [
    "dim1_{}_dim2_{}_gtype_{}_optim_bits_{}".format(*vals)
    for vals in values
]


@pytest.mark.parametrize("dim1, dim2, gtype, optim_bits", values, ids=names)
def test_adam_percentile_clipping(dim1, dim2, gtype, optim_bits):
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8
    p1 = p1.cuda()
    p2 = p1.clone()
    adam1 = bnb.optim.Adam([p1], lr, (beta1, beta2), eps, optim_bits=optim_bits)
    adam2 = bnb.optim.Adam(
        [p2],
        lr,
        (beta1, beta2),
        eps,
        optim_bits=optim_bits,
        percentile_clipping=5,
    )

    gnorm_vec = torch.zeros(100).cuda()
    step = 0

    for i in range(50):
        step += 1
        g1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + (
            0.01 * i
        )
        g2 = g1.clone()
        p2.grad = g2

        current_gnorm, clip_val, gnorm_scale = F.percentile_clipping(
            g1, gnorm_vec, step, 5
        )
        g1 = (g1.float() * gnorm_scale).to(gtype)
        p1.grad = g1

        adam1.step()
        adam2.step()

        # gnorm_scale is not deterministic (warp reductions), as such there can be slight differences in state
        if optim_bits == 32:
            torch.testing.assert_close(p1, p2)
            torch.testing.assert_close(
                adam1.state[p1]["state1"],
                adam2.state[p2]["state1"],
                atol=5e-5,
                rtol=1e-4,
            )
            torch.testing.assert_close(
                adam1.state[p1]["state2"],
                adam2.state[p2]["state2"],
                atol=5e-5,
                rtol=1e-4,
            )
        elif optim_bits == 8:
            torch.testing.assert_close(p1, p2, atol=1e-4, rtol=1e-3)
            torch.testing.assert_close(
                adam1.state[p1]["state1"],
                adam2.state[p2]["state1"],
                atol=2,
                rtol=1e-3,
            )
            torch.testing.assert_close(
                adam1.state[p1]["state2"],
                adam2.state[p2]["state2"],
                atol=2,
                rtol=1e-3,
            )
            adam1.state[p1]["state1"].copy_(adam2.state[p2]["state1"])
            adam1.state[p1]["state2"].copy_(adam2.state[p2]["state2"])
        if i % 10 == 0 and i > 0:
            path = get_temp_dir()
            torch.save(adam2.state_dict(), join(path, "opt.pt"))
            del adam2
            adam2 = None
            adam2 = bnb.optim.Adam(
                [p2],
                lr,
                (beta1, beta2),
                eps,
                optim_bits=optim_bits,
                percentile_clipping=5,
            )
            adam2.load_state_dict(torch.load(join(path, "opt.pt")))


dim1 = [4096]
dim2 = [4096]
gtype = [torch.float32, torch.float16]
# optimizer_names = ['adam8bit_blockwise', 'adam8bit', 'lamb8bit']
# optimizer_names = ['adam8bit_blockwise', 'adam_apex', 'adam8bit', 'adam', 'adam_pytorch']
# optimizer_names = ['momentum_apex', 'momentum8bit', 'momentum_pytorch']
optimizer_names = ['adam_pytorch', "adam8bit_blockwise", 'paged_adam8bit_blockwise', 'paged_adamw8bit_blockwise', 'paged_lion8bit_blockwise']
values = list(product(dim1, dim2, gtype, optimizer_names))
names = [
    "dim1_{}_dim2_{}_gtype_{}_optim_{}".format(*vals) for vals in values
]


@pytest.mark.parametrize("dim1, dim2, gtype, optim_name", values, ids=names)
def test_benchmark_blockwise(dim1, dim2, gtype, optim_name):
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1

    bnb_optimizer = str2optimizers[optim_name][1]([p1])

    g = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.01
    p1.grad = g
    for i in range(10):
        if i == k // 5:
            # 100 iterations for burn-in
            torch.cuda.synchronize()
            t0 = time.time()

        bnb_optimizer.step()

    torch.cuda.synchronize()
    s = time.time() - t0
    print("")
    params = (k - k // 5) * dim1 * dim2
    print(optim_name, gtype, s / params)
    # assert s < 3.9

dim1 = [2*1024]
gtype = [torch.float16]
#mode = ['torch', 'bnb']
mode = ['bnb']
optimizer_names = ['paged_adamw']
#optimizer_names = ['paged_adamw8bit_blockwise']
values = list(product(dim1,gtype, optimizer_names, mode))
names = ['dim1_{0}_gtype_{1}_optim_{2}_mode_{3}'.format(*vals) for vals in values]
@pytest.mark.parametrize("dim1, gtype, optim_name, mode", values, ids=names)
def test_stream_optimizer_bench(dim1, gtype, optim_name, mode):
    layers1 = torch.nn.Sequential(*torch.nn.ModuleList([torch.nn.Linear(dim1, dim1) for i in range(10)]))
    layers1 = layers1.to(gtype)
    layers1 = layers1.cuda()

    large_tensor = None
    if mode == 'torch':
        optim = str2optimizers[optim_name][0](layers1.parameters())
    else:
        optim = str2optimizers[optim_name][1](layers1.parameters())
        # 12 GB
        large_tensor = torch.empty((int(4.5e9),), device='cuda')

    torch.cuda.synchronize()
    time.sleep(5)

    num_batches = 5
    batches = torch.randn(num_batches, 128, dim1, device='cuda').to(gtype)
    lbls = torch.randint(0, 10, size=(num_batches,128)).cuda()

    for i in range(num_batches):
        print(i)
        b = batches[i]
        if i ==2:
            torch.cuda.synchronize()
            t0 = time.time()

        out1 = layers1(b)

        loss1 = torch.nn.functional.cross_entropy(out1, lbls[i]).mean()
        loss1.backward()
        optim.step()
    torch.cuda.synchronize()
    print(mode, time.time() - t0)


#def setup(rank, world_size):

#def cleanup():

def fsdp_main(rank, world_size, linear_type, optim_bits, requires_grads):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(88181178817)

    dim = 128
    layers = 10
    num_batches = 10
    num_iter = 100
    batch_size = 32


    if linear_type == '16bit':
        net2 = torch.nn.Sequential(*[torch.nn.Linear(dim, dim) for i in range(layers)])
        net = torch.nn.Sequential(*[torch.nn.Linear(dim, dim) for i in range(layers)])
    elif linear_type == '4bit':
        modules2 = [torch.nn.Linear(dim, dim) if i % 2 == 1 else bnb.nn.Linear4bit(dim, dim) for i in range(layers)]
        net2 = torch.nn.Sequential(*modules2)
        net = torch.nn.Sequential(*[torch.nn.Linear(dim, dim) for i in range(layers)])

    if not requires_grads:
        for i in range(layers):
            if i % 2 == 0:
                net[i].weight.requires_grad=False
                net2[i].weight.requires_grad=False
                net[i].bias.requires_grad=False
                net2[i].bias.requires_grad=False

    with torch.no_grad():
        for i in range(layers):
            net[i].weight.copy_(net2[i].weight.data)
            net[i].bias.copy_(net2[i].bias.data)


    torch.cuda.set_device(rank)
    model = net

    if not requires_grads:
        with enable_wrap(wrapper_cls=FSDP, **{'device_id' : rank, 'use_orig_params' : True, 'sync_module_states' : True}):
            if not requires_grads:
                for i in range(len(net2)):
                    if not net2[i].weight.requires_grad:
                        net2[i] = wrap(net2[i])
            model2 = wrap(net2)
    else:
        model2 = FSDP(net2, device_id=rank)

    model = model.to(rank)
    model2 = model2.to(rank)
    betas = (0.99, 0.95) # we have random labels, so the default is unstable
    eps = 1e-7
    optim = torch.optim.Adam(model.parameters(), lr=0.0003, betas=betas, eps=eps)
    if optim_bits == 8:
        optim8bit = bnb.optim.Adam8bit(model2.parameters(), lr=0.0003, betas=betas, eps=eps)
    elif optim_bits == 32:
        optim8bit = torch.optim.Adam(model2.parameters(), lr=0.0003, betas=betas, eps=eps)


    batches = torch.randn(num_batches, dim, requires_grad=True)
    lbls = torch.randint(0, dim, size=(num_batches,))
    for i in range(num_batches):
        with torch.no_grad():
            batches[i][lbls[i]] += 0.5

    ddp_loss = torch.zeros(2).to(rank)

    failures = 0
    for i in range(num_iter):
        idx = torch.randint(0, num_batches, size=(batch_size,))
        data, lbl = batches[idx].to(rank), lbls[idx].to(rank)
        data2 = data.detach().clone()
        lbl2 = lbl.detach().clone()
        optim.zero_grad()
        optim8bit.zero_grad()

        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, lbl).mean()
        loss.backward()
        optim.step()


        output2 = model2(data2)
        loss2 = torch.nn.functional.cross_entropy(output2, lbl2).mean()
        loss2.backward()

        outputs = torch.zeros(2).to(rank)
        outputs[0] = output.sum().item()
        outputs[1] = output2.sum().item()
        dist.all_reduce(outputs, op=dist.ReduceOp.SUM)

        optim8bit.step()
        ddp_loss[0] = loss.item()
        ddp_loss[1] = loss2.item()

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            if optim_bits == 32:
                torch.testing.assert_close(ddp_loss[0], ddp_loss[1], atol=1e-4, rtol=0.1)
            elif optim_bits == 8:
                failures += torch.allclose(ddp_loss[0], ddp_loss[1], atol=1e-4, rtol=0.1) == 0
    assert failures < 15
    
    dist.barrier()
    time.sleep(1)
    dist.destroy_process_group()
    time.sleep(1)

@pytest.mark.parametrize("linear_type", ['16bit', '4bit'], ids=['Linear16bit', 'Linear4bit'])
@pytest.mark.parametrize("optim_bits", [8, 32], ids=['optim_8bit', 'optim_32bit'])
@pytest.mark.parametrize("requires_grads", [True, False], ids=['mixedgrads_False', 'mixedgrads_True'])
#@pytest.mark.parametrize("optim_bits", ['32bit'], ids=['optim=32bit'])
def test_fsdp_bnb(linear_type, optim_bits, requires_grads):
    if linear_type == '4bit' and requires_grads == True: pytest.skip('invalid configuration')
    torch.manual_seed(43434484747)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE,linear_type, optim_bits, requires_grads),
        nprocs=WORLD_SIZE,
        join=True)

