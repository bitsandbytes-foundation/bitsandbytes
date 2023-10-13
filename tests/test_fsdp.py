import os
from os.path import join
import time

from peft.peft_model import PeftModel
import pytest
from pytest_cases import case, parametrize_with_cases
import torch
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import bitsandbytes as bnb
from bitsandbytes.distributed.fsdp import (
    bnb_fsdp_auto_wrap_policy,
    parameters_all_consistent,
)
from bitsandbytes.nn import Linear4bit
from bitsandbytes.utils import replace_linear

from .models import HierarchicalModel, SimpleModel


def fsdp_main(rank, world_size, optim_bits):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(1337)

    torch.cuda.set_device(rank)

    # model_name = "huggyllama/llama-7b"
    # adapters_name = 'timdettmers/qlora-flan-7b'

    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # model = replace_linear(PeftModel.from_pretrained(model, adapters_name), Linear4bit)

    # model = PeftModel.from_pretrained(model, adapters_name)
    # AutoTokenizer.from_pretrained(model_name)

    # model_fsdp = FSDP(SimpleModel(), # HierarchicalModel(), 
    #     auto_wrap_policy=bnb_fsdp_auto_wrap_policy)
    
    model_fsdp = SimpleModel()
    from torch.distributed.fsdp.wrap import enable_wrap, wrap
    with enable_wrap(
        wrapper_cls=FSDP,
        **{
            'device_id' : rank,
            # 'use_orig_params' : True,
            'sync_module_states' : True
        }):
        model_fsdp.block.layer1 = wrap(model_fsdp.block.layer1)
        model_fsdp.block.layer2 = wrap(model_fsdp.block.layer2)
        model_fsdp = wrap(model_fsdp)

    model_fsdp = model_fsdp.to(rank)
    print(model_fsdp)

    # model_plain = HierarchicalModel()
    # model_fsdp = FSDP(HierarchicalModel(), auto_wrap_policy=bnb_fsdp_auto_wrap_policy)
    # print(f'{model_plain = }')
    # print(f'{model_fsdp = }')

    # model_plain = model_plain.to(rank)
    # model_fsdp = model_fsdp.to(rank)

    # betas = (0.99, 0.95)  # we have random labels, so the default is unstable
    # eps = 1e-7
    # optim = torch.optim.Adam(model_plain.parameters(), lr=0.0003, betas=betas, eps=eps)
    # if optim_bits == 8:
    #     optim8bit = bnb.optim.Adam8bit(
    #         model_fsdp.parameters(), lr=0.0003, betas=betas, eps=eps)
    # elif optim_bits == 32:
    #     optim8bit = torch.optim.Adam(
    #         model_fsdp.parameters(), lr=0.0003, betas=betas, eps=eps)

    # batches = torch.randn(num_batches, dim, requires_grad=True)
    # lbls = torch.randint(0, dim, size=(num_batches, ))
    # for i in range(num_batches):
    #     with torch.no_grad():
    #         batches[i][lbls[i]] += 0.5

    # ddp_loss = torch.zeros(2).to(rank)

    # failures = 0
    # for _ in range(num_iter):
    #     idx = torch.randint(0, num_batches, size=(batch_size, ))
    #     data, lbl = batches[idx].to(rank), lbls[idx].to(rank)
    #     data2 = data.detach().clone()
    #     lbl2 = lbl.detach().clone()
    #     optim.zero_grad()
    #     optim8bit.zero_grad()

    #     output_plain = model_plain(data)
    #     loss_plain = torch.nn.functional.cross_entropy(output_plain, lbl).mean()
    #     loss_plain.backward()
    #     optim.step()

    #     output_fsdp = model_fsdp(data2)
    #     loss_fsdp = torch.nn.functional.cross_entropy(output_fsdp, lbl2).mean()
    #     loss_fsdp.backward()

    #     outputs = torch.zeros(2).to(rank)
    #     outputs[0] = output_plain.sum().item()
    #     outputs[1] = output_fsdp.sum().item()
    #     dist.all_reduce(outputs, op=dist.ReduceOp.SUM)

    #     optim8bit.step()
    #     ddp_loss[0] = loss_plain.item()
    #     ddp_loss[1] = loss_fsdp.item()

    #     dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    #     if rank == 0:
    #         if optim_bits == 32:
    #             torch.testing.assert_close(
    #                 ddp_loss[0], ddp_loss[1], atol=1e-4, rtol=0.1)
    #         elif optim_bits == 8:
    #             failures += torch.allclose(
    #                 ddp_loss[0], ddp_loss[1], atol=1e-4, rtol=0.1) == 0
    # assert failures < 15
    dist.barrier()
    time.sleep(1)
    dist.destroy_process_group()
    time.sleep(1)


@pytest.mark.parametrize("optim_bits", [32], ids=['optim_32bit'])
# @pytest.mark.parametrize("optim_bits", [8, 32], ids=['optim_8bit', 'optim_32bit'])
def test_fsdp_bnb(optim_bits):
    torch.manual_seed(1337)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, optim_bits), nprocs=WORLD_SIZE, join=True)


class ParametersConsistencyCases:

    def case_no_parameters(self):
        return [], True

    def case_single_parameter(self):
        return [torch.nn.Parameter(torch.tensor([1., 1.]))], True

    def case_same_dtype_same_grad_true(self):
        return [
            torch.nn.Parameter(
                torch.tensor([1, 2], dtype=torch.float32), requires_grad=True),
            torch.nn.Parameter(
                torch.tensor([3, 4], dtype=torch.float32), requires_grad=True)
        ], True

    def case_same_dtype_same_grad_false(self):
        return [
            torch.nn.Parameter(
                torch.tensor([1, 2], dtype=torch.float32), requires_grad=False),
            torch.nn.Parameter(
                torch.tensor([3, 4], dtype=torch.float32), requires_grad=False)
        ], True

    def case_mixed_dtype_mixed_grad(self):
        return [
            torch.nn.Parameter(
                torch.tensor([1, 2], dtype=torch.float32), requires_grad=True),
            torch.nn.Parameter(
                torch.tensor([3, 4], dtype=torch.float64), requires_grad=False)
        ], False

    def case_mixed_dtype_same_grad(self):
        return [
            torch.nn.Parameter(
                torch.tensor([1, 2], dtype=torch.float32), requires_grad=True),
            torch.nn.Parameter(
                torch.tensor([3, 4], dtype=torch.float64), requires_grad=True)
        ], False

    def case_same_dtype_mixed_grad(self):
        return [
            torch.nn.Parameter(
                torch.tensor([1, 2], dtype=torch.float32), requires_grad=True),
            torch.nn.Parameter(
                torch.tensor([3, 4], dtype=torch.float32), requires_grad=False)
        ], False


@parametrize_with_cases("params, expected", cases=ParametersConsistencyCases)
def test_parameters_all_consistent(params, expected):
    assert parameters_all_consistent(params) == expected
