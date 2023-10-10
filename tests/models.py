import torch
import torch.nn as nn
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit


class CustomBlock(nn.Module):

    def __init__(self, in_dim, out_dim, four_bit=False):
        super().__init__()
        self.layer1 = Linear4bit(in_dim, out_dim) if four_bit else nn.Linear(
            in_dim, out_dim)
        self.layer2 = nn.Linear(out_dim, out_dim)

        # Custom parameter at a non-leaf location
        self.custom_param = nn.Parameter(torch.randn(out_dim))

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
