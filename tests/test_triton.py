import pytest
import torch

from bitsandbytes.nn.triton_based_modules import SwitchBackLinear, SwitchBackGlobalLinear



@pytest.mark.parametrize("triton_module", [SwitchBackGlobalLinear, SwitchBackLinear])
def test_switchbatch(triton_module):
    for dim in [83, 17, 128]:
        for batch in [13, 128, 256]:

            standard = torch.nn.Linear(dim, 4 * dim).cuda().half()
            switchback = triton_module(dim, 4 * dim).cuda().half()
            switchback.weight.data.copy_(standard.weight)
            switchback.bias.data.copy_(standard.bias)


            for i in range(100):
                x1 = torch.randn(batch, dim).cuda().half().requires_grad_(True)
                x2 = x1.clone().detach().requires_grad_(True)
                print('standard')
                out_standard = standard(x1)
                print('switchback')
                out_sb = switchback(x1)

                (out_standard.abs().mean()).backward()
                (out_sb.abs().mean()).backward()

                err_sb = (out_standard - out_sb).abs().mean()
                print('OUT', err_sb)

                err_sb = (standard.bias.grad - switchback.bias.grad).abs().mean()

                print('GW2', err_sb)

                err_sb = (standard.weight.grad - switchback.weight.grad).abs().mean()

                print('GW1', err_sb)

                #err_sb = (x1.grad - x2.grad).abs().mean()

                #print('GX1', err_sb)

