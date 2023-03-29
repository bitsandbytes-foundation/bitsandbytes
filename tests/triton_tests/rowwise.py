
import time
import torch
import torch
import torch.nn as nn
import bitsandbytes.nn as bnn
from bitsandbytes.nn.triton_based_modules import SwitchBackLinear, SwitchBackGlobalLinear

from bitsandbytes.nn.triton_utils.v0.quantize_rowwise_nogroup import quantize_rowwise_nogroup


# 256 * 256 * 4096 _> 0.7
# 256 * 128 * 8192 -> 10
if __name__ == '__main__':
    torch.manual_seed(0)

    # hparams
    repeat = 16
    dim=8192
    layers = 4

    batch_size = 256 * 128

    # simulate forward pass
    x = torch.randn(batch_size, dim, dtype=torch.float16).cuda()

    for _ in range(repeat // 2):
        quantize_rowwise_nogroup(x)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        quantize_rowwise_nogroup(x)
    torch.cuda.synchronize()
    end = time.time()

    print(f"time: {(end - start) / repeat * 1000:.3f} ms")





    