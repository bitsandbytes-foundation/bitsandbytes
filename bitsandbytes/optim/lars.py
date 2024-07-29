# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch.optim import Optimizer

from bitsandbytes.optim.optimizer import Optimizer1State


class LARS(Optimizer1State):
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        max_unorm=0.02,
    ):
        """
        Base LARS optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`):
                The learning rate.
            momentum (`float`, defaults to 0):
                The momentum value speeds up the optimizer by taking bigger steps.
            dampening (`float`, defaults to 0):
                The dampening value reduces the momentum of the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            nesterov (`bool`, defaults to `False`):
                Whether to use Nesterov momentum.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            max_unorm (`float`, defaults to 0.02):
                The maximum gradient norm.
        """
        if momentum == 0:
            raise NotImplementedError("LARS without momentum is not supported!")
        super().__init__(
            "lars",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            percentile_clipping,
            max_unorm=max_unorm,
            block_wise=False,
        )


class LARS8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        max_unorm=0.02,
    ):
        """
        8-bit LARS optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`):
                The learning rate.
            momentum (`float`, defaults to 0):
                The momentum value speeds up the optimizer by taking bigger steps.
            dampening (`float`, defaults to 0):
                The dampening value reduces the momentum of the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            nesterov (`bool`, defaults to `False`):
                Whether to use Nesterov momentum.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            max_unorm (`float`, defaults to 0.02):
                The maximum gradient norm.
        """
        if momentum == 0:
            raise NotImplementedError("LARS without momentum is not supported!")
        super().__init__(
            "lars",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            max_unorm=max_unorm,
            block_wise=False,
        )


class LARS32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        max_unorm=0.02,
    ):
        """
        32-bit LARS optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`):
                The learning rate.
            momentum (`float`, defaults to 0):
                The momentum value speeds up the optimizer by taking bigger steps.
            dampening (`float`, defaults to 0):
                The dampening value reduces the momentum of the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            nesterov (`bool`, defaults to `False`):
                Whether to use Nesterov momentum.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            max_unorm (`float`, defaults to 0.02):
                The maximum gradient norm.
        """
        if momentum == 0:
            raise NotImplementedError("LARS without momentum is not supported!")
        super().__init__(
            "lars",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            32,
            args,
            min_8bit_size,
            percentile_clipping,
            max_unorm=max_unorm,
            block_wise=False,
        )


class PytorchLARS(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        max_unorm=0.02,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            max_unorm=max_unorm,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

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
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            max_unorm = group["max_unorm"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    buf = state.get("momentum_buffer", None)

                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        update = d_p + buf * momentum
                    else:
                        update = buf

                update_scale = 1.0
                if max_unorm > 0.0:
                    assert p.dtype == torch.float32
                    pnorm = torch.norm(p.detach())
                    unorm = torch.norm(update)
                    if unorm > max_unorm * pnorm:
                        update_scale = max_unorm * pnorm / unorm

                p.add_(update, alpha=-lr * update_scale)

        return loss
