# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import torch
from bitsandbytes.optim.optimizer import Optimizer1State

torch.optim.Adagrad

class Adagrad(Optimizer1State):
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10,
            optim_bits=32, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if initial_accumulator_value != 0.0:
            raise ValueError('Initial accumulator value != 0.0 not supported!')
        if lr_decay != 0.0:
            raise ValueError('Lr Decay != 0.0 not supported!')
        super(Adagrad, self).__init__('adagrad', params, lr, (0.0, 0.0), eps,
                weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise)

class Adagrad8bit(Optimizer1State):
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10,
            optim_bits=8, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if initial_accumulator_value != 0.0:
            raise ValueError('Initial accumulator value != 0.0 not supported!')
        if lr_decay != 0.0:
            raise ValueError('Lr Decay != 0.0 not supported!')
        assert block_wise
        super(Adagrad8bit, self).__init__('adagrad', params, lr, (0.0, 0.0), eps,
                weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise)

class Adagrad32bit(Optimizer1State):
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10,
            optim_bits=32, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if initial_accumulator_value != 0.0:
            raise ValueError('Initial accumulator value != 0.0 not supported!')
        if lr_decay != 0.0:
            raise ValueError('Lr Decay != 0.0 not supported!')
        super(Adagrad32bit, self).__init__('adagrad', params, lr, (0.0, 0.0), eps,
                weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise)
