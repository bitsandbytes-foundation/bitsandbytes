"""
Extracted from tests/test_functional.py

Note: This feature is currently unused! It is kept here for archival purposes.

Usage: pytest benchmarking/int8/row_scale_benchmark.py
"""

import time

import pytest
import torch

from bitsandbytes import functional as F

k = 20
torch.set_printoptions(precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)


@pytest.mark.parametrize(
    ("dim1", "dim4", "inner"),
    [
        pytest.param(1024, 12288 * 4, 12288, id="1024, 12288*4, 12288"),
        pytest.param(2048, 4096 * 4, 4096, id="2048, 4096*4, 4096"),
    ],
)
@pytest.mark.skip("Row scale has some bugs for ampere")
@pytest.mark.benchmark
def test_row_scale_bench(dim1, dim4, inner):
    formatB = F.get_special_format_str()
    err1, err2, err3 = [], [], []
    relerr1, relerr2 = [], []
    scale = 1
    A = torch.randn(dim1, inner, device="cuda").half()
    B = torch.randn(dim4, inner, device="cuda").half()
    torch.nn.init.xavier_uniform_(B)
    # warmpup
    for i in range(k):
        C1 = torch.matmul(A, B.t())

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        C1 = torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print("16", time.time() - t0)

    C1a, C1b, stats1a, stats1b, coo_tensor = F.int8_double_quant(A)
    CB, absmaxB = F.vectorwise_quant(B, quant_type="linear")
    A2, SA = F.nvidia_transform(C1a, "col32")
    B2, SB = F.nvidia_transform(CB, formatB)
    A1, maxA = F.vectorwise_quant(A, dim=1)

    c = 10.0 * inner * scale
    row_scale = maxA / c
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        outC32 = F.int8_linear_matmul(A2, B2, dtype=torch.int8, row_scale=row_scale)
    torch.cuda.synchronize()
    print("row-wise", time.time() - t0)

    C2a, C2b, stats2a, stats2b, coo_tensor = F.int8_double_quant(B)
    B2, SB = F.nvidia_transform(C2a, formatB)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        outC32 = F.int8_linear_matmul(A2, B2)
    torch.cuda.synchronize()
    print("vector-wise", time.time() - t0)
