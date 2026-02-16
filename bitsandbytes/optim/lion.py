# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from bitsandbytes.optim.optimizer import Optimizer1State


class Lion(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    ):
        """
        Base Lion optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-4):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            weight_decay (`float`, defaults to 0):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        super().__init__(
            "lion",
            params,
            lr,
            betas,
            0.0,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            is_paged=is_paged,
        )


class Lion8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    ):
        """
        8-bit Lion optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-4):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            weight_decay (`float`, defaults to 0):
                The weight decay value for the optimizer.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        super().__init__(
            "lion",
            params,
            lr,
            betas,
            0.0,
            weight_decay,
            8,
            args,
            min_8bit_size,
            is_paged=is_paged,
        )


class Lion32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    ):
        """
        32-bit Lion optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-4):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            weight_decay (`float`, defaults to 0):
                The weight decay value for the optimizer.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        super().__init__(
            "lion",
            params,
            lr,
            betas,
            0.0,
            weight_decay,
            32,
            args,
            min_8bit_size,
            is_paged=is_paged,
        )


class PagedLion(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
    ):
        """
        Paged Lion optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-4):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            weight_decay (`float`, defaults to 0):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
        """
        super().__init__(
            "lion",
            params,
            lr,
            betas,
            0.0,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            is_paged=True,
        )


class PagedLion8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        args=None,
        min_8bit_size=4096,
    ):
        """
        Paged 8-bit Lion optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-4):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            weight_decay (`float`, defaults to 0):
                The weight decay value for the optimizer.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
        """
        super().__init__(
            "lion",
            params,
            lr,
            betas,
            0.0,
            weight_decay,
            8,
            args,
            min_8bit_size,
            is_paged=True,
        )


class PagedLion32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        args=None,
        min_8bit_size=4096,
    ):
        """
        Paged 32-bit Lion optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-4):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            weight_decay (`float`, defaults to 0):
                The weight decay value for the optimizer.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
        """
        super().__init__(
            "lion",
            params,
            lr,
            betas,
            0.0,
            weight_decay,
            32,
            args,
            min_8bit_size,
            is_paged=True,
        )
