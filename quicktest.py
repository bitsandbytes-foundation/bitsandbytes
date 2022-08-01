from itertools import product

import torch

import bitsandbytes as bnb
import bitsandbytes.functional as F


def test_igemmlt(dim1, dim2, dim3, dim4, dims, ldb):
    k = 25
    for i in range(k):
        if dims == 2:
            A = torch.randint(-128, 127, size=(dim1, dim3), device="cuda").to(
                torch.int8
            )
        elif dims == 3:
            A = torch.randint(
                -128, 127, size=(dim1, dim2, dim3), device="cuda"
            ).to(torch.int8)
        B = torch.randint(-128, 127, size=(dim4, dim3), device="cuda").to(
            torch.int8
        )
        C1 = torch.matmul(A.float(), B.t().float())

        A2, SA = F.transform(A, "col32")
        B2, SB = F.transform(B, "colx")
        if dims == 2:
            C2, SC = F.transform(
                torch.zeros(
                    A.shape[0], B.shape[0], dtype=torch.int32, device="cuda"
                ),
                "col32",
            )
        else:
            C2, SC = F.transform(
                torch.zeros(
                    A.shape[0],
                    A.shape[1],
                    B.shape[0],
                    dtype=torch.int32,
                    device="cuda",
                ),
                "col32",
            )
        F.igemmlt(A2, B2, C2, SA, SB, SC)
        C3, S = F.transform(C2, "row", state=SC)
        # torch.testing.assert_allclose(C1, C3.float())
        # print(C1)
        # print(C2)
        # print(C3)
        allclose = torch.allclose(C1, C3.float())
        if allclose:
            print(C1)
            print(C2)
            print(C3)

        ## transposed
        # A = torch.randint(-128, 127, size=(dim4, dim3), device='cuda').to(torch.int8)
        # if dims == 2:
        #    B = torch.randint(-128, 127, size=(dim1, dim3), device='cuda').to(torch.int8)
        #    C1 = torch.matmul(A.float(), B.float().t())
        # elif dims == 3:
        #    B = torch.randint(-128, 127, size=(dim1, dim2, dim3), device='cuda').to(torch.int8)
        #    C1 = torch.matmul(B.float(), A.t().float())
        #    C1 = C1.permute([2, 0, 1])

        # A2, SA = F.transform(A, 'col32')
        # B2, SB = F.transform(B, 'colx')
        # if dims == 2:
        #    C2, SC = F.transform(torch.zeros(A.shape[0], B.shape[0], dtype=torch.int32, device='cuda'), 'col32')
        # else:
        #    C2 = torch.zeros(A.shape[0], B.shape[0], B.shape[1], dtype=torch.int32, device='cuda')
        #    state = (C2.shape, 'row', A.shape[0])
        #    C2, SC = F.transform(C2, 'col32', state=state)
        # F.igemmlt(A2, B2, C2, SA, SB, SC)
        # C3, S = F.transform(C2, 'row', state=SC, ld=[0])
        # torch.testing.assert_allclose(C1, C3.float())

        ## weight update
        # if dims == 3:
        #    A = torch.randint(-128, 127, size=(dim1, dim2, dim3), device='cuda').to(torch.int8)
        #    B = torch.randint(-128, 127, size=(dim1, dim2, dim4), device='cuda').to(torch.int8)
        #    C1 = torch.matmul(B.view(-1, B.shape[-1]).t().float(), A.view(-1, A.shape[-1]).float())

        #    A2, SA = F.transform(A.view(-1, A.shape[-1]).t().contiguous(), 'colx')
        #    B2, SB = F.transform(B.view(-1, B.shape[-1]).t().contiguous(), 'col32')
        #    C2 = torch.zeros(B.shape[-1], A.shape[-1], dtype=torch.int32, device='cuda')
        #    C2, SC = F.transform(C2, 'col32')
        #    F.igemmlt(B2, A2, C2, SB, SA, SC)
        #    C3, S = F.transform(C2, 'row', state=SC)
        #    torch.testing.assert_allclose(C1, C3.float())


dims = (2, 3)
ldb = [0]

n = 2
dim1 = torch.randint(1, 256, size=(n,)).tolist()
dim2 = torch.randint(32, 512, size=(n,)).tolist()
dim3 = torch.randint(32, 1024, size=(n,)).tolist()
dim4 = torch.randint(32, 1024, size=(n,)).tolist()
values = list(product(dim1, dim2, dim3, dim4, dims, ldb))

for ldb in range(32, 4096, 32):
    # for ldb in [None]:
    val = test_igemmlt(2, 2, 2, 2, 2, ldb)
    if val:
        print(val, ldb)
    else:
        print("nope", ldb)
# for val in values:
# test_igemmlt(*val)
