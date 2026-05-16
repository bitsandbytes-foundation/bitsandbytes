import torch

import bitsandbytes as bnb

p = torch.nn.Parameter(torch.rand(10, 10).cuda())
a = torch.rand(10, 10).cuda()

p1 = p.data.sum().item()

adam = bnb.optim.Adam([p])

out = a * p
loss = out.sum()
loss.backward()
adam.step()

p2 = p.data.sum().item()

if p1 == p2:
    raise AssertionError
print("SUCCESS!")
print("Installation was successful!")
