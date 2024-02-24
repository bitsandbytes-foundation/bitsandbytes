import json
import time

import torch

from bitsandbytes.triton.int8_matmul_mixed_dequantize import (
    int8_matmul_mixed_dequantize,
)
from bitsandbytes.triton.int8_matmul_rowwise_dequantize import (
    int8_matmul_rowwise_dequantize,
)
from bitsandbytes.triton.quantize_columnwise_and_transpose import (
    quantize_columnwise_and_transpose,
)
from bitsandbytes.triton.quantize_global import (
    quantize_global,
    quantize_global_transpose,
)
from bitsandbytes.triton.quantize_rowwise import quantize_rowwise

# KNOW ISSUE: need to optimize "w_quantize_colwise_transpose" when embeddim is too large.


def get_time(k, fn, info_dict):
    for _ in range(repeat // 2):
        fn()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()

    torch.cuda.synchronize()
    end = time.time()
    ms = (end - start) / repeat * 1000
    print(f"time {k}: {ms:.3f} ms")
    info_dict[k] = ms


if __name__ == "__main__":
    torch.manual_seed(0)
    wm = 4
    for dim in [1024, 1280, 1408, 1664, 2048, 4096]:
        # note "batch_size" is actually "batch_size * embed_dim", which is why it's large
        for batch_size in [256 * 32, 256 * 64, 256 * 128, 256 * 256, 256 * 512]:
            # switch switches dim_in and dim_out
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

                info = {
                    "repeat": repeat,
                    "batch_size": batch_size,
                    "dim_out": dim_out,
                    "dim_in": dim_in,
                    "wm": wm,
                    "switch": switch,
                }

                get_time("standard_fwd", lambda: x.matmul(w.t()), info)
                get_time("standard_gw", lambda: g.t().matmul(x), info)
                get_time("standard_gx", lambda: g.matmul(w), info)
                get_time(
                    "rowwise_fwd",
                    lambda: int8_matmul_rowwise_dequantize(
                        x_int8,
                        w_int8.t(),
                        state_x_rowwise,
                        state_w_columnwise,
                        None,
                    ),
                    info,
                )
                get_time(
                    "rowwise_bwd",
                    lambda: int8_matmul_rowwise_dequantize(
                        g_int8,
                        wt_int8.t(),
                        state_x_rowwise,
                        state_w_rowwise,
                        None,
                    ),
                    info,
                )
                get_time(
                    "global_fwd",
                    lambda: int8_matmul_mixed_dequantize(x_int8, w_int8.t(), state_x_rowwise, state_w_global, None),
                    info,
                )
                get_time(
                    "global_bwd",
                    lambda: int8_matmul_mixed_dequantize(g_int8, wt_int8.t(), state_x_rowwise, state_w_global, None),
                    info,
                )
                get_time("x_quantize_rowwise", lambda: quantize_rowwise(x), info)
                get_time("g_quantize_rowwise", lambda: quantize_rowwise(g), info)
                get_time("w_quantize_rowwise", lambda: quantize_rowwise(w), info)
                get_time("w_quantize_colwise_transpose", lambda: quantize_columnwise_and_transpose(w), info)
                get_time("w_quantize_global", lambda: quantize_global(w), info)
                get_time("w_quantize_global_transpose", lambda: quantize_global_transpose(w), info)

                time_standard = info["standard_fwd"] + info["standard_gx"] + info["standard_gw"]
                time_rowwise = (
                    info["x_quantize_rowwise"]
                    + info["g_quantize_rowwise"]
                    + info["w_quantize_colwise_transpose"]
                    + info["w_quantize_rowwise"]
                    + info["standard_gw"]
                    + info["rowwise_fwd"]
                    + info["rowwise_bwd"]
                )
                time_global = (
                    info["x_quantize_rowwise"]
                    + info["g_quantize_rowwise"]
                    + info["w_quantize_global"]
                    + info["w_quantize_global_transpose"]
                    + info["standard_gw"]
                    + info["global_fwd"]
                    + info["global_bwd"]
                )

                print("TOTAL STANDARD", time_standard)
                print("TOTAL ROWWISE", time_rowwise)
                print("TOTAL GLOBAL", time_global)

                print("speedup", -100 * (time_global - time_standard) / time_standard)

                info["time_standard"] = time_standard
                info["time_rowwise"] = time_rowwise
                info["time_global"] = time_global

                info_json = json.dumps(info)

                # TODO: change this to what you want.
                with open("speed_benchmark/info.jsonl", "a") as file:
                    file.write(info_json + "\n")
