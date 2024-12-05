"""
Extracted from tests/test_functional.py

Usage: pytest benchmarking/matmul_benchmark.py
"""

import time

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F

k = 20

torch.set_printoptions(precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)


@pytest.mark.parametrize(
    ("batch", "seq", "model", "hidden"),
    [
        # pytest.param(1, 128, 6656, 4 * 6656, id="batch=1, seq=128, model=6656, hidden=26k"),
        pytest.param(1, 1, 3584, 512, id="batch=1, seq=128, model=3584, hidden=19k"),
        # pytest.param(4, 128, 6656, 4 * 6656, id="batch=4, seq=128, model=6656, hidden=26k"),
        # pytest.param(16, 256, 6656, 4 * 6656, id="batch=16, seq=256, model=6656, hidden=26k")
    ],
)
@pytest.mark.benchmark
def test_bench_matmul(batch, seq, model, hidden):
    iters = 1000
    formatB = F.get_special_format_str()

    A = torch.randn(batch, seq, model, device="cuda").half()
    B = torch.empty(hidden, model, dtype=torch.float16, device="cuda")
    torch.nn.init.xavier_uniform_(B)

    B_fp4, state = F.quantize_fp4(B)
    B_fp4_c, state_c = F.quantize_fp4(B, compress_statistics=True)

    B_nf4, state_nf4 = F.quantize_nf4(B)
    B_nf4_c, state_nf4_c = F.quantize_nf4(B, compress_statistics=True)

    linear8bit = bnb.nn.Linear8bitLt(model, hidden, False, False).cuda().half()
    linear8bit.eval()

    outliers = torch.randint(0, model, size=(5,)).cuda()
    A[:, :, outliers] = 8.0

    linearMixedBit = bnb.nn.Linear8bitLt(model, hidden, False, False, threshold=6.0).cuda().half()
    # linearMixedBit.eval()

    linear8bit_train = bnb.nn.Linear8bitLt(model, hidden, False).cuda().half()
    linear8bit_train_thresh = bnb.nn.Linear8bitLt(model, hidden, False, threshold=6.0).cuda().half()
    bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)

    # warmup
    for i in range(iters):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print("")

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print(
        f"pytorch fp16: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s",
    )

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    bnb.matmul_4bit(A, B_fp4.t(), quant_state=state)
    # torch.cuda.synchronize()
    # print( f"bnb fp4: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s" )

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    bnb.matmul_4bit(A, B_fp4.t(), quant_state=state_c)
    # torch.cuda.synchronize()
    # print( f"bnb fp4 + compressed stats: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s" )

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)
    torch.cuda.synchronize()
    print(f"bnb nf4: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        bnb.matmul_4bit(A, B_nf4_c.t(), quant_state=state_nf4_c)
    torch.cuda.synchronize()
    print(f"bnb nf4+DQ: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        bnb.matmul(A, B)
    torch.cuda.synchronize()
    print(
        f"B -> CB (each iteration): [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s"
    )

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        bnb.matmul(A, B, threshold=6.0)
    torch.cuda.synchronize()
    print(
        f"B -> CB + threshold: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s"
    )

    CA, SCA, _ = F.int8_vectorwise_quant(A, threshold=0.0)
    CB, SCB, _ = F.int8_vectorwise_quant(B)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        # CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(A, threshold=0.0)
        out32 = F.int8_linear_matmul(CA, CB)
    torch.cuda.synchronize()
    print(
        f"no overhead int8 [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s"
    )

    # C32A, SA = F.transform(CA, "col32")

    # CxB, SB = F.transform(CB, to_order=formatB)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    out32, Sout32 = F.igemmlt(C32A, CxB, SA, SB)
    # torch.cuda.synchronize()
    # print(f"no overhead matmul-lt: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(A, threshold=0.0)
    # C32A, SA = F.transform(CA, "col32")
    # CB, CBt, SCB, SCBt, coo_tensorB = F.double_quant(B)
    # CxB, SB = F.transform(CB, to_order=formatB)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    out32, Sout32 = F.igemmlt(C32A, CxB, SA, SB)
    # torch.cuda.synchronize()
    # print(f"no overhead matmul-lt: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # BA, statsB = F.vectorwise_quant(B, dim=1)
    # CxB, SB = F.nvidia_transform(CB, to_order=formatB)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    A2 = A.view(-1, A.shape[-1]).contiguous()
    #    CA, statsA = F.vectorwise_quant(A2, dim=1)
    #    C32A, SA = F.nvidia_transform(CA, "col32")
    #    out32, Sout32 = F.igemmlt(C32A, CxB, SA, SB)
    #    Cout, Sout = F.nvidia_transform(out32, "row", state=Sout32)
    #    F.vectorwise_mm_dequant(Cout, statsA, statsB.t())
    # torch.cuda.synchronize()
    # print(f"vector pytorch + nvidia: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # BA, statsB = F.vectorwise_quant(B, dim=1, quant_type="linear")
    # CxB, SB = F.nvidia_transform(CB, to_order=formatB)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    A2 = A.view(-1, A.shape[-1]).contiguous()
    #    CA, statsA = F.vectorwise_quant(A2, dim=1, quant_type="linear")
    #    C32A, SA = F.nvidia_transform(CA, "col32")
    #    out32, Sout32 = F.igemmlt(C32A, CxB, SA, SB)
    #    Cout, Sout = F.nvidia_transform(out32, "row", state=Sout32)
    #    out = Cout * statsB * statsA * (1.0 / (127 * 127))
    # torch.cuda.synchronize()
    # print(f"linear pytorch + nvidia: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    linear8bit(A)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        linear8bit(A)
    torch.cuda.synchronize()
    print(
        f"bnb linear8bitlt (eval): [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s"
    )

    linearMixedBit(A)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        linearMixedBit(A)
    torch.cuda.synchronize()
    print(
        f"bnb linear8bitlt with threshold (eval): [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s"
    )

    # linear8bit_train(A)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    linear8bit_train(A)
    # torch.cuda.synchronize()
    # print( f"bnb linear8bitlt (training): [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # linear8bit_train_thresh(A)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    linear8bit_train(A)
    # torch.cuda.synchronize()
    # print( f"bnb linear8bitlt with threshold (training): [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")
