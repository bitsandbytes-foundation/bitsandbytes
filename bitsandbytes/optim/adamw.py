# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import torch
from bitsandbytes.optim.optimizer import Optimizer2State
import bitsandbytes.functional as F

class AdamW(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=1e-2, amsgrad=False, optim_bits=32, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        super(AdamW, self).__init__('adam', params, lr, betas, eps,
                weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise)

class AdamW8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=1e-2, amsgrad=False, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        super(AdamW8bit, self).__init__('adam', params, lr, betas, eps,
                weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise)

class AdamW32bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=1e-2, amsgrad=False, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        super(AdamW32bit, self).__init__('adam', params, lr, betas, eps,
                weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise)

