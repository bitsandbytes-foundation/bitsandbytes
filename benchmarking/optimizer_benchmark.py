"""
Extracted from tests/test_optim.py

Usage: pytest benchmarking/optimizer_benchmark.py
"""

import time

import pytest
from tests.helpers import describe_dtype, id_formatter
import torch

import bitsandbytes as bnb

str2optimizers = {"paged_adamw": (torch.optim.AdamW, bnb.optim.PagedAdamW)}


@pytest.mark.parametrize("dim1", [2 * 1024], ids=id_formatter("dim1"))
@pytest.mark.parametrize("gtype", [torch.float16], ids=describe_dtype)
@pytest.mark.parametrize("optim_name", ["paged_adamw"], ids=id_formatter("optim_name"))
@pytest.mark.parametrize("mode", ["bnb"], ids=id_formatter("mode"))
@pytest.mark.benchmark
def test_stream_optimizer_bench(dim1, gtype, optim_name, mode):
    layers1 = torch.nn.Sequential(*torch.nn.ModuleList([torch.nn.Linear(dim1, dim1) for i in range(10)]))
    layers1 = layers1.to(gtype)
    layers1 = layers1.cuda()

    large_tensor = None
    if mode == "torch":
        optim = str2optimizers[optim_name][0](layers1.parameters())
    else:
        optim = str2optimizers[optim_name][1](layers1.parameters())
        # 12 GB
        large_tensor = torch.empty((int(4.5e9),), device="cuda")

    torch.cuda.synchronize()
    time.sleep(5)

    num_batches = 5
    batches = torch.randn(num_batches, 128, dim1, device="cuda").to(gtype)
    lbls = torch.randint(0, 10, size=(num_batches, 128)).cuda()

    for i in range(num_batches):
        print(i)
        b = batches[i]
        if i == 2:
            torch.cuda.synchronize()
            t0 = time.time()

        out1 = layers1(b)

        loss1 = torch.nn.functional.cross_entropy(out1, lbls[i]).mean()
        loss1.backward()
        optim.step()
    torch.cuda.synchronize()
    print(mode, time.time() - t0)
