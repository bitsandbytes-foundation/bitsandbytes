import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F




def debug_blocksize(block):
    x = torch.randn(4096, 4096).cuda()
    qx, qstate = F.quantize_fp4(x, blocksize=block)
    dq = F.dequantize_fp4(qx, qstate)
    return torch.sum(torch.linalg.norm(x - dq, ord="fro"))

def test_blocksize(block):
    x = torch.randn(10, 10).cuda()
    qx, qstate = F.quantize_fp4(x, blocksize=block)
    print(x)
    print("---------------")
    print(qx)
    print("---------------")
    print(qstate)

    


for block in [128, 256, 512, 1024, 2048]:
    print(debug_blocksize(block))

#test_blocksize(2048)
