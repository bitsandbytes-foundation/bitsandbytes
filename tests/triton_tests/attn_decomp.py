
import torch
import json
from bitsandbytes.nn.triton_based_modules import SwitchBackGlobalMLP, SwitchBackGlobalLinear, MyLinear
import time

# class AttentionOld(torch.nn.Module):
#     def __init__(
#             self,
#             dim,
#             num_heads=8,
#             qkv_bias=True,
#             scaled_cosine=False,
#             scale_heads=False,
#             attn_drop=0.,
#             proj_drop=0.,
#             linear_module=torch.nn.Linear,
#     ):
#         super().__init__()
#         self.scaled_cosine = scaled_cosine
#         self.scale_heads = scale_heads
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         self.in_proj_linear = linear_module(dim, 3 * dim, bias = qkv_bias)

#         self.attn_drop = torch.nn.Dropout(attn_drop)
#         if self.scale_heads:
#             self.head_scale = torch.nn.Parameter(torch.ones((num_heads, 1, 1)))
#         else:
#             self.head_scale = None
#         self.out_proj = linear_module(dim, dim)
#         self.out_drop = torch.nn.Dropout(proj_drop)

#     def forward(self, x, attn_mask = None):
#         L, N, C = x.shape

#         q, k, v = self.in_proj_linear(x).chunk(3, dim=-1)
            
#         q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
#         k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
#         v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

#         q = q * self.scale
#         attn = torch.bmm(q, k.transpose(-1, -2))

#         if attn_mask is not None:
#             if attn_mask.dtype == torch.bool:
#                 new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
#                 new_attn_mask.masked_fill_(attn_mask, float("-inf"))
#                 attn_mask = new_attn_mask
#             attn += attn_mask
        
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = torch.bmm(attn, v)
#         x = x.transpose(0, 1).reshape(L, N, C)

#         x = self.out_proj(x)
#         x = self.out_drop(x)
#         return x
    
class Attention(torch.nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            attn_drop=0.,
            proj_drop=0.,
            linear_module=torch.nn.Linear,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.ln = torch.nn.LayerNorm(dim)

        self.in_proj_linear = linear_module(dim, 3 * dim, bias = qkv_bias)

        self.attn_drop = torch.nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = torch.nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = linear_module(dim, dim)
        self.out_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x, attn_mask = None):
        q, k, v = self.in_proj_linear(self.ln(x)).chunk(3, dim=-1)
        x = torch.compile(torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask))
        x = self.out_proj(x)
        return x

if __name__ == '__main__':


    for dim in [1024, 1280, 1408, 1664, 2048]:
        for batch in [2**14, 2**15, 2**16, 2**17]:

            # if dim != 4096 or batch != 2**17:
            #     continue

            x1 = torch.randn( batch // 256, 256, dim ).cuda().requires_grad_(True)
            qu = torch.randn( batch // 256, 256, dim ).cuda().requires_grad_(True)
            ke = torch.randn( batch // 256, 256, dim ).cuda().requires_grad_(True)
            va = torch.randn( batch // 256, 256, dim ).cuda().requires_grad_(True)

            standard = Attention(dim).cuda()
            my_standard = Attention(dim, linear_module=MyLinear).cuda()
            sb = Attention(dim, linear_module=SwitchBackGlobalLinear).cuda()
            standard_compiled = torch.compile(standard)
            ln_model = torch.nn.Sequential(
                    torch.nn.LayerNorm(dim),
                    torch.nn.LayerNorm(dim),
                ).cuda()
            ln_model_compiled = torch.compile(
                ln_model
            )
            gelu_model = torch.nn.Sequential(
                    torch.nn.GELU(),
                ).cuda()
            gelu_model_compiled = torch.compile(
                gelu_model
            )


            print('Model part 2')

            repeat = 32
            
            info = {'repeat' : repeat, 'batch_size' : batch, 'dim' : dim}


            k = 'attn'
            for _ in range(repeat // 2):
                with torch.cuda.amp.autocast():
                    out_attn = torch.nn.functional.scaled_dot_product_attention(qu, ke, va)
                ((2 ** 16) * out_attn).abs().mean().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                with torch.cuda.amp.autocast():
                    out_attn = torch.nn.functional.scaled_dot_product_attention(qu, ke, va)
                ((2 ** 16) * out_attn).abs().mean().backward()

            torch.cuda.synchronize()
            end = time.time()
            ms = (end - start) / repeat * 1000
            print(f"time {k}: {ms:.3f} ms")
            info[k] = ms

            k = 'ln'
            for _ in range(repeat // 2):
                with torch.cuda.amp.autocast():
                    out = ln_model(x1)
                ((2 ** 16) * out).abs().mean().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                with torch.cuda.amp.autocast():
                    out = ln_model(x1)
                ((2 ** 16) * out).abs().mean().backward()

            torch.cuda.synchronize()
            end = time.time()
            ms = (end - start) / repeat * 1000
            print(f"time {k}: {ms:.3f} ms")
            info[k] = ms

            x1.grad.zero_()

            k = 'ln_compiled'
            for _ in range(repeat // 2):
                with torch.cuda.amp.autocast():
                    out = ln_model_compiled(x1)
                ((2 ** 16) * out).abs().mean().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                with torch.cuda.amp.autocast():
                    out = ln_model_compiled(x1)
                ((2 ** 16) * out).abs().mean().backward()

            torch.cuda.synchronize()
            end = time.time()
            ms = (end - start) / repeat * 1000
            print(f"time {k}: {ms:.3f} ms")
            info[k] = ms

            k = 'gelu'
            for _ in range(repeat // 2):
                with torch.cuda.amp.autocast():
                    out = gelu_model(x1)
                ((2 ** 16) * out).abs().mean().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                with torch.cuda.amp.autocast():
                    out = gelu_model(x1)
                ((2 ** 16) * out).abs().mean().backward()

            torch.cuda.synchronize()
            end = time.time()
            ms = (end - start) / repeat * 1000
            print(f"time {k}: {ms:.3f} ms")
            info[k] = ms

            x1.grad.zero_()

            k = 'gelu_compiled'
            for _ in range(repeat // 2):
                with torch.cuda.amp.autocast():
                    out = gelu_model_compiled(x1)
                ((2 ** 16) * out).abs().mean().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                with torch.cuda.amp.autocast():
                    out = gelu_model_compiled(x1)
                ((2 ** 16) * out).abs().mean().backward()

            torch.cuda.synchronize()
            end = time.time()
            ms = (end - start) / repeat * 1000
            print(f"time {k}: {ms:.3f} ms")
            info[k] = ms


            x1.grad.zero_()

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
            # 
            # 

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


            with open("tests/triton_tests/attn_info_ln.jsonl", "a") as file:
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