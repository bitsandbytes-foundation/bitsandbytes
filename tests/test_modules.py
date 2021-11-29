# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import pytest
import torch
import bitsandbytes as bnb

from itertools import product

from bitsandbytes import functional as F


@pytest.mark.parametrize("embcls", [bnb.nn.Embedding, bnb.nn.StableEmbedding], ids=['Embedding', 'StableEmbedding'])
def test_embeddings(embcls):
    bnb.optim.GlobalOptimManager.get_instance().initialize()
    emb1 = torch.nn.Embedding(100, 512).cuda()
    emb2 = embcls(100, 512).cuda()

    adam1 = bnb.optim.Adam8bit(emb1.parameters())
    adam2 = bnb.optim.Adam8bit(emb2.parameters())

    batches = torch.randint(1, 100, size=(100, 4, 32)).cuda()

    for i in range(100):
        batch = batches[i]

        embedded1 = emb1(batch)
        embedded2 = emb2(batch)

        l1 = embedded1.mean()
        l2 = embedded2.mean()

        l1.backward()
        l2.backward()

        adam1.step()
        adam2.step()

        adam1.zero_grad()
        adam2.zero_grad()

        assert adam1.state[emb1.weight]['state1'].dtype == torch.uint8
        assert adam2.state[emb2.weight]['state1'].dtype == torch.float32


