
import torch
import json
from bitsandbytes.nn.triton_based_modules import SwitchBackGlobalMLP, SwitchBackGlobalLinear, StandardLinear
import time

if __name__ == '__main__':
    
    print('Startin')


    for dim in [1024, 1280, 1408, 1664, 2048]:
        for batch in [2**14, 2**15, 2**16, 2**17]:
            
            x1 = torch.randn(batch, dim).cuda().requires_grad_(True)
            d = 2

            standard = torch.nn.Sequential(
                torch.nn.LayerNorm(dim),
                torch.nn.Linear(dim, 4 * dim),
                torch.nn.GELU(),
                torch.nn.Linear(4 * dim, dim),
            ).cuda()

            my_standard = torch.nn.Sequential(
                torch.nn.LayerNorm(dim),
                StandardLinear(dim, 4 * dim),
                torch.nn.GELU(),
                StandardLinear(4 * dim, dim),
            ).cuda()

            fused_mlp = SwitchBackGlobalMLP(dim, 4 * dim).cuda()

            sb = torch.nn.Sequential(
                torch.nn.LayerNorm(dim),
                SwitchBackGlobalLinear(dim, 4 * dim),
                torch.nn.GELU(),
                SwitchBackGlobalLinear(4 * dim, dim),
            ).cuda()
            
            standard_compiled = torch.compile(standard)

            print('Model part 2')

            repeat = 32
            

            info = {'repeat' : repeat, 'batch_size' : batch, 'dim' : dim}

            k = 'standard'
            for _ in range(repeat // 2):
                with torch.cuda.amp.autocast():
                    out_standard = standard(x1)
                ((2 ** 16) * out_standard).abs().mean().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                with torch.cuda.amp.autocast():
                    out_standard = standard(x1)
                ((2 ** 16) * out_standard).abs().mean().backward()

            torch.cuda.synchronize()
            end = time.time()
            ms = (end - start) / repeat * 1000
            print(f"time {k}: {ms:.3f} ms")
            info[k] = ms


            x1.grad.zero_()
            
            k = 'my_standard'
            for _ in range(repeat // 2):
                with torch.cuda.amp.autocast():
                    out_my_standard = my_standard(x1)
                ((2 ** 16) * out_my_standard).abs().mean().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                with torch.cuda.amp.autocast():
                    out_my_standard = my_standard(x1)
                ((2 ** 16) * out_my_standard).abs().mean().backward()

            torch.cuda.synchronize()
            end = time.time()
            ms = (end - start) / repeat * 1000
            print(f"time {k}: {ms:.3f} ms")
            info[k] = ms

            x1.grad.zero_()

            k = 'standard_compiled'
            for _ in range(repeat // 2):
                with torch.cuda.amp.autocast():
                    out_standard_compiled = standard_compiled(x1)
                ((2 ** 16) * out_standard_compiled).abs().mean().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                with torch.cuda.amp.autocast():
                    out_standard_compiled = standard_compiled(x1)
                ((2 ** 16) * out_standard_compiled).abs().mean().backward()

            torch.cuda.synchronize()
            end = time.time()
            ms = (end - start) / repeat * 1000
            print(f"time {k}: {ms:.3f} ms")
            info[k] = ms

            x1.grad.zero_()

            k = 'sb'
            for _ in range(repeat // 2):
                with torch.cuda.amp.autocast():
                    out_sb = sb(x1)
                ((2 ** 16) * out_sb).abs().mean().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                with torch.cuda.amp.autocast():
                    out_sb = sb(x1)
                ((2 ** 16) * out_sb).abs().mean().backward()

            torch.cuda.synchronize()
            end = time.time()
            ms = (end - start) / repeat * 1000
            print(f"time {k}: {ms:.3f} ms")
            info[k] = ms


            info_json = json.dumps(info)


            with open("tests/triton_tests/info_mlp_autocast_ln.jsonl", "a") as file:
                file.write(info_json + "\n")


        #exit()

    # err_fused = (out_standard - out_fused).abs().mean()
    # err_sb = (out_standard - out_sb).abs().mean()
    # print('OUT', err_fused, err_sb)

    # err_fused = (standard[d].weight.grad - fused_mlp.linear2.weight.grad).abs().mean()
    # err_sb = (standard[d].weight.grad - sb[d].weight.grad).abs().mean()

    # print('GW2', err_fused, err_sb)

    # err_fused = (standard[0].weight.grad - fused_mlp.linear1.weight.grad).abs().mean()
    # err_sb = (standard[0].weight.grad - sb[0].weight.grad).abs().mean()

    # print('GW1', err_fused, err_sb)

    # err_fused = (x1.grad - x2.grad).abs().mean()
    # err_sb = (x1.grad - x3.grad).abs().mean()

    # print('GX1', err_fused, err_sb)

    # import pdb; pdb.set_trace()


    # # NO GELU, ST GRADIENTS, EVERYTHING FINE.
