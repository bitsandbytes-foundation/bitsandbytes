# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from bitsandbytes.optim.optimizer import Optimizer1State


class RMSprop(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
    ):
        """
        Base RMSprop optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-2):
                The learning rate.
            alpha (`float`, defaults to 0.99):
                The alpha value is the decay rate of the squared gradients of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            momentum (`float`, defaults to 0):
                The momentum value speeds up the optimizer by taking bigger steps.
            centered (`bool`, defaults to `False`):
                Whether the gradients are normalized by the variance. If `True`, it can help training at the expense of additional compute.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
        """
        if alpha == 0:
            raise NotImplementedError("RMSprop with alpha==0.0 is not supported!")
        if centered:
            raise NotImplementedError("Centered RMSprop is not supported!")
        super().__init__(
            "rmsprop",
            params,
            lr,
            (alpha, momentum),
            eps,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
        )


class RMSprop8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        args=None,
        min_8bit_size=4096,
    ):
        """
        8-bit RMSprop optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-2):
                The learning rate.
            alpha (`float`, defaults to 0.99):
                The alpha value is the decay rate of the squared gradients of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            momentum (`float`, defaults to 0):
                The momentum value speeds up the optimizer by taking bigger steps.
            centered (`bool`, defaults to `False`):
                Whether the gradients are normalized by the variance. If `True`, it can help training at the expense of additional compute.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
        """
        if alpha == 0:
            raise NotImplementedError("RMSprop with alpha==0.0 is not supported!")
        if centered:
            raise NotImplementedError("Centered RMSprop is not supported!")
        super().__init__(
            "rmsprop",
            params,
            lr,
            (alpha, momentum),
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
        )


class RMSprop32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        args=None,
        min_8bit_size=4096,
    ):
        """
        32-bit RMSprop optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-2):
                The learning rate.
            alpha (`float`, defaults to 0.99):
                The alpha value is the decay rate of the squared gradients of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            momentum (`float`, defaults to 0):
                The momentum value speeds up the optimizer by taking bigger steps.
            centered (`bool`, defaults to `False`):
                Whether the gradients are normalized by the variance. If `True`, it can help training at the expense of additional compute.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
        """

        if alpha == 0:
            raise NotImplementedError("RMSprop with alpha==0.0 is not supported!")
        if centered:
            raise NotImplementedError("Centered RMSprop is not supported!")
        super().__init__(
            "rmsprop",
            params,
            lr,
            (alpha, momentum),
            eps,
            weight_decay,
            32,
            args,
            min_8bit_size,
        )
