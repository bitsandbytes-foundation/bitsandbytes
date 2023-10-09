import os
from os.path import join

import pytest

import torch

import bitsandbytes as bnb

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import enable_wrap, wrap

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
    dist.destroy_process_group()

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

def fsdp():
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=40,
    )