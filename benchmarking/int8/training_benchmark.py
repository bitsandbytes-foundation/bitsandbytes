"""
Extracted from tests/test_functional.py

Usage: pytest benchmarking/int8/training_benchmark.py
"""

import time

import pytest
import torch

from bitsandbytes import functional as F

k = 20

torch.set_printoptions(precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)


@pytest.mark.parametrize(
    ("batch", "seq", "model", "hidden"),
    [
        pytest.param(2, 512, 4 * 1024, 3 * 4 * 1024, id="batch=2, seq=512, model=4k, hidden=12k"),
        pytest.param(2, 512, 5120, 3 * 5120, id="batch=2, seq=512, model=5k, hidden=15k"),
        pytest.param(2, 512, 12 * 1024, 4 * 12 * 1024, id="batch=2, seq=512, model=12k, hidden=48k"),
    ],
)
@pytest.mark.benchmark
def test_bench_8bit_training(batch, seq, model, hidden):
    formatB = F.get_special_format_str()
    A = torch.randn(batch, seq, model, device="cuda").half()
    grad = torch.randn(batch, seq, model, device="cuda").half()
    w1 = torch.randint(-128, 127, size=(hidden, model), device="cuda").half()
    w2 = torch.randint(-128, 127, size=(model, hidden), device="cuda").half()
    print("")

    # torch.cuda.synchronize()
    ## warmup
    # for i in range(100):
    #    torch.matmul(A, w1.t())
    # torch.cuda.synchronize()

    dtype = torch.int8
    A = A.view(-1, A.shape[-1]).contiguous()
    grad = grad.view(-1, grad.shape[-1]).contiguous()
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        out1 = torch.matmul(A, w1.t())  # fc1
        # out2 = torch.matmul(out1, w2.t())# fc2

        # d1 = torch.matmul(grad, w2) # delta1
        # d2 = torch.matmul(d1, w1) # delta2

        # grad1 = torch.einsum('bo,bh->oh', out1, grad) # grad w2
        # grad2 = torch.einsum('bh,bo->ho', A, d2) # grad w1

    torch.cuda.synchronize()
    t16 = time.time() - t0
    print(t16)

    # torch.cuda.empty_cache()

    # Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    # Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)

    # CTw1, Sw1 = F.transform2(Cw1, formatB)
    # CTw2, Sw2 = F.transform2(Cw2, formatB)
    # CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)
    # CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)

    # CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
    # C32A, SA = F.transform2(CA, 'col32')
    ## fc1
    # out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1, dtype=dtype)
    ##out1 = F.mm_dequant(out1_32, Sout1_32, statsAt, statsw1t)

    ## fc2
    # Cout1, Cout1t, statsout1, statsout1t, coo_tensor = F.double_quant(out1)
    # C32out1, Sout1 = F.transform2(Cout1, 'col32')
    # out2_32, Sout2_32 = F.igemmlt(C32out1, CTw2, Sout1, Sw2, dtype=dtype)
    ##out2 = F.mm_dequant(out2_32, Sout2_32, statsout1t, statsw2t)

    ## delta1
    # Cgrad, Cgradt, statsgrad, statsgradt, coo_tensor = F.double_quant(grad)
    # C32grad, Sgrad = F.transform2(Cgrad, 'col32')
    ##d1_32, Sd1_32 = F.igemmlt(C32grad, CTw2t, Sgrad, Sw2t, dtype=dtype)
    ##d1 = F.mm_dequant(d1_32, Sd1_32, statsgradt, statsw2)

    ## delta2
    # Cd1, Cd1t, statsd1, statsd1t, coo_tensor = F.double_quant(d1)
    # C32d1, Sd1 = F.transform2(Cd1, 'col32')
    ##d2_32, Sd2_32 = F.igemmlt(C32d1, CTw1t, Sd1, Sw1t, dtype=dtype)
    ##d2 = F.mm_dequant(d2_32, Sd2_32, statsd1t, statsw1)

    ## grad1
    # C32out1t, Sout1t = F.transform2(Cout1t, 'col32', transpose=True)
    # CTgradt, Sgradt = F.transform2(Cgradt, formatB, transpose=True)
    ##grad1_32, Sgrad1_32 = F.igemmlt(C32out1t, CTgradt, Sout1t, Sgradt, dtype=dtype)
    ##grad1 = F.mm_dequant(grad1_32, Sgrad1_32, statsout1, statsgrad)

    ## grad2
    # C32At, SAt = F.transform2(CAt, 'col32', transpose=True)
    # CTd1t, Sd1t = F.transform2(Cd1t, formatB, transpose=True)
    ##grad2_32, Sgrad2_32 = F.igemmlt(C32At, CTd1t, SAt, Sd1t, dtype=dtype)
    ##grad2 = F.mm_dequant(grad2_32, Sgrad2_32, statsA, statsd1)

    # Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)

    # Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    # Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)

    # CTw1, Sw1 = F.transform2(Cw1, formatB)
    # CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)
    # CTw2, Sw2 = F.transform2(Cw2, formatB)
    # CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(k):
    #    #Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    #    #CTw1, Sw1 = F.transform2(Cw1, formatB)
    #    #Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    #    #CTw1, Sw1 = F.transform2(Cw1, formatB)

    #    #CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A, threshold=3.5)
    #    CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
    #    #CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)
    #    #CTw2, Sw2 = F.transform2(Cw2, formatB)
    #    #CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)

    #    C32A, SA = F.transform2(CA, 'col32')

    #    # fc1
    #    out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1, dtype=dtype)
    #    #out1dn = F.mm_dequant(out1_32, Sout1_32, statsA, statsw1)

    #    #print(coo_tensor.nnz)
    #    #out1sp = F.spmm_coo(coo_tensor, w1.t())
    #    #print(w1.t().shape)
    #    #out1 = out1dn + out1sp

    #    # fc2
    #    Cout1, Cout1t, statsout1, statsout1t, coo_tensor = F.double_quant(out1)
    #    C32out1, Sout1 = F.transform2(Cout1, 'col32')
    #    out2_32, Sout2_32 = F.igemmlt(C32out1, CTw2, Sout1, Sw2, dtype=dtype)
    #    #out2 = F.mm_dequant(out2_32, Sout2_32, statsout1, statsw2)

    #    # delta1
    #    Cgrad, Cgradt, statsgrad, statsgradt, coo_tensor = F.double_quant(grad)
    #    C32grad, Sgrad = F.transform2(Cgrad, 'col32')
    #    d1_32, Sd1_32 = F.igemmlt(C32grad, CTw2t, Sgrad, Sw2t, dtype=dtype)
    #    #d1 = F.mm_dequant(d1_32, Sd1_32, statsgrad, statsw2t)

    #    # delta2
    #    Cd1, Cd1t, statsd1, statsd1t, coo_tensor = F.double_quant(d1)
    #    C32d1, Sd1 = F.transform2(Cd1, 'col32')
    #    d2_32, Sd2_32 = F.igemmlt(C32d1, CTw1t, Sd1, Sw1t, dtype=dtype)
    #    #d2 = F.mm_dequant(d2_32, Sd2_32, statsd1, statsw1t)

    #    # grad1
    #    #C32out1t, Sout1t = F.transform2(Cout1t, 'col32', transpose=True)
    #    #CTgradt, Sgradt = F.transform2(Cgradt, formatB, transpose=True)
    #    #grad1_32, Sgrad1_32 = F.igemmlt(C32out1t, CTgradt, Sout1t, Sgradt, dtype=dtype)
    #    #grad1 = F.mm_dequant(grad1_32, Sgrad1_32, statsout1t, statsgradt)

    #    ## grad2
    #    #C32At, SAt = F.transform2(CAt, 'col32', transpose=True)
    #    #CTd1t, Sd1t = F.transform2(Cd1t, formatB, transpose=True)
    #    #grad2_32, Sgrad2_32 = F.igemmlt(C32At, CTd1t, SAt, Sd1t, dtype=dtype)
    #    #grad2 = F.mm_dequant(grad2_32, Sgrad2_32, statsAt, statsd1t)

    # torch.cuda.synchronize()
    # t8 = time.time() - t0
    # print(t8)
