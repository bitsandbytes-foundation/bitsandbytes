# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
from bitsandbytes.optim.optimizer import Optimizer2State

class LAMB(Optimizer2State):
    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, adam_w_mode=True, optim_bits=32, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=False, max_unorm=1.0):
        super(LAMB, self).__init__('lamb', params, lr, betas, eps,
                weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, max_unorm=1.0)

class LAMB8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, adam_w_mode=True, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=False, max_unorm=1.0):
        super(LAMB8bit, self).__init__('lamb', params, lr, betas, eps,
                weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, max_unorm=1.0)

class LAMB32bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, adam_w_mode=True, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=False, max_unorm=1.0):
        super(LAMB32bit, self).__init__('lamb', params, lr, betas, eps,
                weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise, max_unorm=1.0)


