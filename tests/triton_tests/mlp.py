
import time
import torch
import torch.nn as nn
import bitsandbytes.nn as bnn
from bitsandbytes.nn.triton_based_modules import SwitchBackLinear, SwitchBackGlobalLinear, MyLinear

import triton.language as tl

def construct_model(dim, layers, module):
    modules = []
    for _ in range(layers):
        modules.append(module(dim, 4*dim))
        modules.append(module(4*dim, dim))
    return nn.Sequential(*modules).cuda().train()

def get_time(model, x, name):
    for _ in range(repeat // 2):
        #with torch.cuda.amp.autocast():
        out = model(x)
        #(2**16 * out.pow(2).mean()).backward()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        # with torch.cuda.amp.autocast():
        out = model(x)
        #(2**16 * out.pow(2).mean()).backward()

    torch.cuda.synchronize()
    end = time.time()
    print(f"time {name}: {(end - start) / repeat * 1000:.3f} ms")

if __name__ == '__main__':
    torch.manual_seed(0)

    # hparams
    repeat = 16
    dim=2048
    layers =4 
    batch_size = 2
    sequence_length = 2**15

    # construct models
    standard = construct_model(dim, layers, nn.Linear).half()
    my_standard = construct_model(dim, layers, MyLinear).half()
    switchback = construct_model(dim, layers, SwitchBackLinear).half()
    switchback_global = construct_model(dim, layers, SwitchBackGlobalLinear).half()
    #bnb_8bitmixed = construct_model(dim, layers, bnn.Linear8bitLt)

    # simulate forward pass
    x = torch.randn(batch_size * sequence_length, dim, dtype=torch.float16).cuda()

    # get time for forward and backward
    get_time(standard, x, "standard")
    get_time(my_standard, x, "my_standard")
    get_time(switchback, x, "switchback")
    get_time(switchback_global, x, "switchback_global")
    #get_time(bnb_8bitmixed, x, "bnb_8bitmixed")






    