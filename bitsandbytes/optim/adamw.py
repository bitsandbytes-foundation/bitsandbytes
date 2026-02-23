# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from bitsandbytes.optim.optimizer import Optimizer2State


class AdamW(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    ):
        """
        Base AdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            amsgrad (`bool`, defaults to `False`):
                Whether to use the [AMSGrad](https://hf.co/papers/1904.09237) variant of Adam that uses the maximum of past squared gradients instead.
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
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            is_paged=is_paged,
        )


class AdamW8bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    ):
        """
        8-bit AdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            amsgrad (`bool`, defaults to `False`):
                Whether to use the [AMSGrad](https://hf.co/papers/1904.09237) variant of Adam that uses the maximum of past squared gradients instead.
                Note: This parameter is not supported in AdamW8bit and must be False.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
                Note: This parameter is not used in AdamW8bit as it always uses 8-bit optimization.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        # Validate unsupported parameters
        if amsgrad:
            raise ValueError("AdamW8bit does not support amsgrad=True")

        if optim_bits != 32:
            # We allow the default value of 32 to maintain compatibility with the function signature,
            # but any other value is invalid since AdamW8bit always uses 8-bit optimization
            raise ValueError("AdamW8bit only supports optim_bits=32 (default value for compatibility)")

        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,  # Hardcoded to 8 bits
            args,
            min_8bit_size,
            is_paged=is_paged,
        )


class AdamW32bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    ):
        """
        32-bit AdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            amsgrad (`bool`, defaults to `False`):
                Whether to use the [AMSGrad](https://hf.co/papers/1904.09237) variant of Adam that uses the maximum of past squared gradients instead.
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
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            32,
            args,
            min_8bit_size,
            is_paged=is_paged,
        )


class PagedAdamW(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
    ):
        """
        Paged AdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            amsgrad (`bool`, defaults to `False`):
                Whether to use the [AMSGrad](https://hf.co/papers/1904.09237) variant of Adam that uses the maximum of past squared gradients instead.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
        """
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            is_paged=True,
        )


class PagedAdamW8bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
    ):
        """
        Paged 8-bit AdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            amsgrad (`bool`, defaults to `False`):
                Whether to use the [AMSGrad](https://hf.co/papers/1904.09237) variant of Adam that uses the maximum of past squared gradients instead.
                Note: This parameter is not supported in PagedAdamW8bit and must be False.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
                Note: This parameter is not used in PagedAdamW8bit as it always uses 8-bit optimization.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
        """
        # Validate unsupported parameters
        if amsgrad:
            raise ValueError("PagedAdamW8bit does not support amsgrad=True")

        if optim_bits != 32:
            # We allow the default value of 32 to maintain compatibility with the function signature,
            # but any other value is invalid since PagedAdamW8bit always uses 8-bit optimization
            raise ValueError("PagedAdamW8bit only supports optim_bits=32 (default value for compatibility)")

        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,  # Hardcoded to 8 bits
            args,
            min_8bit_size,
            is_paged=True,
        )


class PagedAdamW32bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
    ):
        """
        Paged 32-bit AdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            amsgrad (`bool`, defaults to `False`):
                Whether to use the [AMSGrad](https://hf.co/papers/1904.09237) variant of Adam that uses the maximum of past squared gradients instead.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
        """
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            32,
            args,
            min_8bit_size,
            is_paged=True,
        )
