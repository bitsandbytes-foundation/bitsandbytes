import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsandbytes.nn import Linear4bit


class CustomBlock(nn.Module):

    def __init__(self, in_dim, out_dim, bias=False, four_bit=False):
        super().__init__()
        self.layer1 = Linear4bit(in_dim, out_dim, bias) if four_bit else nn.Linear(
            in_dim, out_dim, bias)
        self.layer2 = nn.Linear(out_dim, out_dim, bias)

        # Custom parameter at a non-leaf location
        # self.custom_param = nn.Parameter(torch.randn(out_dim))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x += self.custom_param  # Using the custom parameter in some way
        return x


class HierarchicalModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = CustomBlock(64, 128)
        self.block2 = CustomBlock(128, 256, four_bit=True)
        self.block3 = CustomBlock(256, 512)

        # Custom parameter at a non-leaf location
        self.global_custom_param = nn.Parameter(torch.randn(512))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x += self.global_custom_param  # Using the custom parameter in some way
        return x


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = CustomBlock(128, 256, four_bit=True)
        
    def forward(self, x):
        return self.block1(x)


class MoreComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = CustomBlock(128, 256, four_bit=True)
        self.block2 = CustomBlock(128, 256, four_bit=True)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class LoraModel(nn.Module):
    """This is a toy LoRA decoder model."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([LoraDecoder() for _ in range(4)])
        self.norm = nn.LayerNorm(32)
        self.embed_tokens.weight.requires_grad_(False)
        self.norm.weight.requires_grad_(False)
        self.norm.bias.requires_grad_(False)


class LoraDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = LoraAttention()
        self.mlp = LoraMLP()
        self.inp_layernorm = nn.LayerNorm(32)
        self.post_attn_layernorm = nn.LayerNorm(32)
        self.inp_layernorm.weight.requires_grad_(False)
        self.inp_layernorm.bias.requires_grad_(False)
        self.post_attn_layernorm.weight.requires_grad_(False)
        self.post_attn_layernorm.bias.requires_grad_(False)


class LoraAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 32, bias=False)
        self.lora_A = nn.Linear(32, 8, bias=False)
        self.lora_B = nn.Linear(8, 32, bias=False)
        self.k_proj = nn.Linear(32, 32, bias=False)
        self.v_proj = nn.Linear(32, 32, bias=False)
        self.o_proj = nn.Linear(32, 32, bias=False)
        self.q_proj.weight.requires_grad_(False)
        self.k_proj.weight.requires_grad_(False)
        self.v_proj.weight.requires_grad_(False)
        self.o_proj.weight.requires_grad_(False)


class LoraMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Linear(32, 128, bias=False)
        self.proj2 = nn.Linear(128, 32, bias=False)
        self.proj1.weight.requires_grad_(False)
        self.proj2.weight.requires_grad_(False)