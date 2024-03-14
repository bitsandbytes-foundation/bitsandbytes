# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from bitsandbytes.optim.optimizer import Optimizer1State


class Adagrad(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
    ):
        """
        Base Adagrad optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-2):
                The learning rate.
            lr_decay (`int`, defaults to 0):
                The learning rate decay.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            initial_accumulator_value (`int`, defaults to 0):
                The initial momemtum values.
            eps (`float`, defaults to 1e-10):
                The epsilon value prevents division by zero in the optimizer.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if initial_accumulator_value != 0.0:
            raise ValueError("Initial accumulator value != 0.0 not supported!")
        if lr_decay != 0.0:
            raise ValueError("Lr Decay != 0.0 not supported!")
        super().__init__(
            "adagrad",
            params,
            lr,
            (0.0, 0.0),
            eps,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )


class Adagrad8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        optim_bits=8,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
    ):
        """
        8-bit Adagrad optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-2):
                The learning rate.
            lr_decay (`int`, defaults to 0):
                The learning rate decay.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            initial_accumulator_value (`int`, defaults to 0):
                The initial momemtum values.
            eps (`float`, defaults to 1e-10):
                The epsilon value prevents division by zero in the optimizer.
            optim_bits (`int`, defaults to 8):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if initial_accumulator_value != 0.0:
            raise ValueError("Initial accumulator value != 0.0 not supported!")
        if lr_decay != 0.0:
            raise ValueError("Lr Decay != 0.0 not supported!")
        assert block_wise
        super().__init__(
            "adagrad",
            params,
            lr,
            (0.0, 0.0),
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )


class Adagrad32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
    ):
        """
        32-bit Adagrad optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-2):
                The learning rate.
            lr_decay (`int`, defaults to 0):
                The learning rate decay.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            initial_accumulator_value (`int`, defaults to 0):
                The initial momemtum values.
            eps (`float`, defaults to 1e-10):
                The epsilon value prevents division by zero in the optimizer.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if initial_accumulator_value != 0.0:
            raise ValueError("Initial accumulator value != 0.0 not supported!")
        if lr_decay != 0.0:
            raise ValueError("Lr Decay != 0.0 not supported!")
        super().__init__(
            "adagrad",
            params,
            lr,
            (0.0, 0.0),
            eps,
            weight_decay,
            32,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )
