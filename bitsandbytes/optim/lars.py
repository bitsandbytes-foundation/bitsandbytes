# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import torch

from torch.optim import Optimizer
from bitsandbytes.optim.optimizer import Optimizer1State

class LARS(Optimizer1State):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, optim_bits=32, args=None,
                 min_8bit_size=4096, percentile_clipping=100, max_unorm=0.02):
        if momentum == 0:
            raise NotImplementError(f'LARS without momentum is not supported!')
        super(LARS, self).__init__('lars', params, lr, (momentum, dampening), 0.0,
                weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, max_unorm=max_unorm, block_wise=False)

class LARS8bit(Optimizer1State):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, args=None,
                 min_8bit_size=4096, percentile_clipping=100, max_unorm=0.02):
        if momentum == 0:
            raise NotImplementError(f'LARS without momentum is not supported!')
        super(LARS8bit, self).__init__('lars', params, lr, (momentum, dampening), 0.0,
                weight_decay, 8, args, min_8bit_size, percentile_clipping, max_unorm=max_unorm, block_wise=False)

class LARS32bit(Optimizer1State):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, args=None,
                 min_8bit_size=4096, percentile_clipping=100, max_unorm=0.02):
        if momentum == 0:
            raise NotImplementError(f'LARS without momentum is not supported!')
        super(LARS32bit, self).__init__('lars', params, lr, (momentum, dampening), 0.0,
                weight_decay, 32, args, min_8bit_size, percentile_clipping, max_unorm=max_unorm, block_wise=False)


class PytorchLARS(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, max_unorm=0.02):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, max_unorm=max_unorm)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PytorchLARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PytorchLARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            max_unorm = group['max_unorm']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None: continue

                state = self.state[p]
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = state.get('momentum_buffer', None)

                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state['momentum_buffer']= buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        update = d_p + buf*momentum
                    else:
                        update = buf

                update_scale = 1.0
                if max_unorm > 0.0:
                    assert p.dtype == torch.float32
                    pnorm = torch.norm(p.detach())
                    unorm = torch.norm(update)
                    if unorm > max_unorm*pnorm:
                        update_scale = max_unorm*pnorm/unorm

                p.add_(update, alpha=-lr*update_scale)

        return loss
