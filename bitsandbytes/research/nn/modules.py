from typing import TypeVar

import torch
from torch import nn

import bitsandbytes as bnb

T = TypeVar("T", bound="torch.nn.Module")


class LinearFP8Mixed(nn.Linear):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__(input_features, output_features, bias)
        self.bw_code = None
        self.fw_code = None
        array = [4096, 2048, 1024, 512, 256, 128, 64, 0]
        for i, k in enumerate(array):
            if input_features > array[i + 1]:
                self.bsz = k
                break
        for i, k in enumerate(array):
            if output_features > array[i + 1]:
                self.bsz2 = k
                break

    def forward(self, x: torch.Tensor):
        if self.fw_code is None:
            self.bw_code = bnb.functional.create_fp8_map(True, 5, 2, 8).to(x.device)
            self.fw_code = bnb.functional.create_fp8_map(True, 4, 3, 8).to(x.device)

        out = bnb.research.matmul_fp8_mixed(
            x,
            self.weight.t(),
            fw_code=self.fw_code,
            bw_code=self.bw_code,
            bsz=self.bsz,
            bsz2=self.bsz2,
        )
        if self.bias is not None:
            out += self.bias

        return out


class LinearFP8Global(nn.Linear):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__(input_features, output_features, bias)
        self.bw_code = None
        self.fw_code = None
        array = [4096, 2048, 1024, 512, 256, 128, 64, 0]
        for i, k in enumerate(array):
            if input_features > array[i + 1]:
                self.bsz = k
                break
        for i, k in enumerate(array):
            if output_features > array[i + 1]:
                self.bsz2 = k
                break

    def forward(self, x: torch.Tensor):
        if self.fw_code is None:
            self.bw_code = bnb.functional.create_fp8_map(True, 5, 2, 8).to(x.device)
            self.fw_code = bnb.functional.create_fp8_map(True, 4, 3, 8).to(x.device)

        out = bnb.matmul_fp8_global(
            x,
            self.weight.t(),
            fw_code=self.fw_code,
            bw_code=self.bw_code,
            bsz=self.bsz,
            bsz2=self.bsz2,
        )
        if self.bias is not None:
            out += self.bias

        return out
