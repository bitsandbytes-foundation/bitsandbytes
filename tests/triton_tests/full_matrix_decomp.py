import json

import time
import torch
import torch.nn as nn
import bitsandbytes.nn as bnn
from bitsandbytes.nn.triton_based_modules import SwitchBackLinear, SwitchBackGlobalLinear, StandardLinear

from bitsandbytes.nn.triton_utils.v0.quantize_rowwise_nogroup import quantize_rowwise_nogroup
from bitsandbytes.nn.triton_utils.v0.quantize_columnwise_nogroup_transpose import quantize_columnwise_nogroup_transpose
from bitsandbytes.nn.triton_utils.v0.int8_matmul_rowwise_dequantize_bias import int8_matmul_rowwise_dequantize_bias
from bitsandbytes.nn.triton_utils.v0.int8_matmul_rowwise_dequantize import int8_matmul_rowwise_dequantize
from bitsandbytes.nn.triton_utils.v0.quantize_global import quantize_global, quantize_global_transpose
from bitsandbytes.nn.triton_utils.v0.int8_matmul_mixed_dequanitze import int8_matmul_mixed_dequanitze, int8_matmul_mixed_dequanitze_bias

# KNOW ISSUE: need to optimize "w_quantize_colwise_transpose" when embeddim is too large.
# not that big of an issue.

def get_time_standard_fwd(k, v):

    x = torch.randn(batch_size, dim_in, dtype=torch.float16).cuda()
    g = torch.randn(batch_size, dim_out, dtype=torch.float16).cuda()

    ##### time matmul 1
    for _ in range(repeat // 2):
        g.t().matmul(x)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        g.t().matmul(x)

    torch.cuda.synchronize()
    end = time.time()
    print(f"time {k}: {(end - start) / repeat * 1000:.3f} ms")
    return (end - start) / repeat * 1000

if __name__ == '__main__':
    torch.manual_seed(0)
    #for (dim, wm) in [(1024, 4), (1280, 4), (1408, 4.3637), (1664, 4.9231), (2048, 4), (4096, 4), (8096, 4)]
    for (dim, wm) in [(1408, 4), (1664, 4),]:

        for batch_size in [256*32, 256*64, 256*128, 256*256, 256*512]:
            #for batch_size in [256*256, 256*512]:

            for switch in [False, True]:


                # hparams
                repeat = 64
                batch_size = batch_size
                dim_out = dim * wm
                dim_in = dim
                if switch:
                    dim_out = dim
                    dim_in = wm * dim

                dim_in = round(dim_in)
                dim_out = round(dim_out)


                # simulate forward pass
                x = torch.randn(batch_size, dim_in, dtype=torch.float16).cuda()
                g = torch.randn(batch_size, dim_out, dtype=torch.float16).cuda()
                w = torch.randn(dim_out, dim_in, dtype=torch.float16).cuda()
                
                x_int8 = x.clone().to(torch.int8)
                g_int8 = g.clone().to(torch.int8)
                w_int8 = w.clone().to(torch.int8)
                wt_int8 = w.t().contiguous().clone().to(torch.int8)
                state_x_rowwise = x.max(dim=1)[0]
                state_g_rowwise = g.max(dim=1)[0]
                state_w_columnwise = w.max(dim=0)[0]
                state_w_rowwise = w.max(dim=1)[0]
                state_w_global = w.max()

                info = {'repeat' : repeat, 'batch_size' : batch_size, 'dim_out' : dim_out, 'dim_in' : dim_in, 'wm' : wm, 'switch' : switch}

                k = 'standard_fwd'
                for _ in range(repeat // 2):
                    x.matmul(w.t())

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    x.matmul(w.t())

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms

                k = 'standard_gw'
                for _ in range(repeat // 2):
                    g.t().matmul(x)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    g.t().matmul(x)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms


                k = 'standard_gx'
                for _ in range(repeat // 2):
                    g.matmul(w)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    g.matmul(w)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms



                k = 'rowwise_fwd'
                for _ in range(repeat // 2):
                    int8_matmul_rowwise_dequantize(x_int8, w_int8.t(), state_x_rowwise, state_w_columnwise)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    int8_matmul_rowwise_dequantize(x_int8, w_int8.t(), state_x_rowwise, state_w_columnwise)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms

                k = 'rowwise_bwd'
                for _ in range(repeat // 2):
                    int8_matmul_rowwise_dequantize(g_int8, wt_int8.t(), state_x_rowwise, state_w_rowwise)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    int8_matmul_rowwise_dequantize(g_int8, wt_int8.t(), state_x_rowwise, state_w_rowwise)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms


                k = 'global_fwd'
                for _ in range(repeat // 2):
                    int8_matmul_mixed_dequanitze(x_int8, w_int8.t(), state_x_rowwise, state_w_global)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    int8_matmul_mixed_dequanitze(x_int8, w_int8.t(), state_x_rowwise, state_w_global)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms


                k = 'global_bwd'
                for _ in range(repeat // 2):
                    int8_matmul_mixed_dequanitze(g_int8, wt_int8.t(), state_x_rowwise, state_w_global)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    int8_matmul_mixed_dequanitze(g_int8, wt_int8.t(), state_x_rowwise, state_w_global)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms


                k = 'x_quantize_rowwise'
                for _ in range(repeat // 2):
                    quantize_rowwise_nogroup(x)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    quantize_rowwise_nogroup(x)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms

                k = 'g_quantize_rowwise'
                for _ in range(repeat // 2):
                    quantize_rowwise_nogroup(g)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    quantize_rowwise_nogroup(g)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms

                k = 'w_quantize_rowwise'
                for _ in range(repeat // 2):
                    quantize_rowwise_nogroup(w)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    quantize_rowwise_nogroup(w)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms


                k = 'w_quantize_colwise_transpose'
                for _ in range(repeat // 2):
                    quantize_columnwise_nogroup_transpose(w)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    quantize_columnwise_nogroup_transpose(w)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms


                k = 'w_quantize_global'
                for _ in range(repeat // 2):
                    quantize_global(w)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    quantize_global(w)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms

                k = 'w_quantize_global_transpose'
                for _ in range(repeat // 2):
                    quantize_global_transpose(w)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    quantize_global_transpose(w)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms


                k = 'cast_x'
                for _ in range(repeat // 2):
                    newx = x.to(torch.int8)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    newx = x.to(torch.int8)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms



                k = 'cast_g'
                for _ in range(repeat // 2):
                    newx = g.to(torch.int8)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    newx = g.to(torch.int8)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms



                k = 'cast_w'
                for _ in range(repeat // 2):
                    newx = w.to(torch.int8)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeat):
                    newx = w.to(torch.int8)

                torch.cuda.synchronize()
                end = time.time()
                ms = (end - start) / repeat * 1000
                print(f"time {k}: {ms:.3f} ms")
                info[k] = ms


                time_standard = info['standard_fwd'] + info['standard_gx'] + info['standard_gw']
                time_rowwise = info['x_quantize_rowwise'] + info['g_quantize_rowwise']  + info['w_quantize_colwise_transpose'] + info['w_quantize_rowwise'] + info['standard_gw'] + info['rowwise_fwd'] + info['rowwise_bwd']
                time_global = info['x_quantize_rowwise'] + info['g_quantize_rowwise'] + info['w_quantize_global'] + info['w_quantize_global_transpose'] + info['standard_gw'] + info['global_fwd'] + info['global_bwd']

                print('TOTAL STANDARD', time_standard)
                print('TOTAL ROWWISE', time_rowwise)
                print('TOTAL GLOBAL', time_global)

                print('speedup', -100*(time_global - time_standard)/time_standard)

                info['time_standard'] = time_standard
                info['time_rowwise'] = time_rowwise
                info['time_global'] = time_global



                info_json = json.dumps(info)


                with open("tests/triton_tests/info.jsonl", "a") as file:
                    file.write(info_json + "\n")
