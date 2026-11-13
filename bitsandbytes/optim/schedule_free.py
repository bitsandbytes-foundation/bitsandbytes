# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch

import bitsandbytes.functional as F
from bitsandbytes.optim.optimizer import GlobalOptimManager, MockArgs, Optimizer8bit


class AdamWScheduleFree(Optimizer8bit):
    """Schedule-Free AdamW optimizer with optional 8-bit blockwise quantized state.

    Implements Schedule-Free AdamW from *"The Road Less Scheduled"*
    (Defazio et al., 2024), supporting 32-bit and 8-bit blockwise-quantized
    optimizer states (``z`` and ``exp_avg_sq``).

    Call ``optimizer.train()`` before training steps and ``optimizer.eval()``
    before evaluation or checkpointing so that parameters are swapped between the
    interpolated training iterate ``y`` and the averaged evaluation iterate ``x``.
    """

    def __init__(
        self,
        params,
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        optim_bits: int = 32,
        args: Optional[object] = None,
        min_8bit_size: int = 4096,
        blocksize: int = 256,
    ):
        """Initialize the Schedule-Free AdamW optimizer.

        Arguments:
            params (`Iterable`):
                Iterable of parameters or parameter-group dicts to optimize.
            lr (`float`, defaults to `0.0025`):
                The base learning rate.
            betas (`tuple[float, float]`, defaults to `(0.9, 0.999)`):
                Coefficients ``(beta1, beta2)`` for iterate interpolation and
                second-moment estimation. ``beta1`` must be in ``(0, 1)``.
            eps (`float`, defaults to `1e-8`):
                Term added to the denominator to improve numerical stability.
            weight_decay (`float`, defaults to `0.0`):
                Decoupled weight decay coefficient.
            warmup_steps (`int`, defaults to `0`):
                Number of linear warmup steps.
            r (`float`, defaults to `0.0`):
                Polynomial weighting exponent for iterate averaging.
            weight_lr_power (`float`, defaults to `2.0`):
                Power applied to the learning rate in the averaging weight.
            optim_bits (`int`, defaults to `32`):
                Bit width of the optimizer state (``8`` or ``32``).
            args (`object`, defaults to `None`):
                Optional additional arguments object.
            min_8bit_size (`int`, defaults to `4096`):
                Minimum number of elements for a parameter's state to be 8-bit quantized.
            blocksize (`int`, defaults to `256`):
                Block size for 8-bit blockwise quantization.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 < betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps: {warmup_steps}")
        if blocksize not in (64, 128, 256, 512, 1024, 2048, 4096):
            raise ValueError(f"Invalid blocksize: {blocksize}. Must be one of (64, 128, 256, 512, 1024, 2048, 4096).")
        if optim_bits not in (8, 32):
            raise NotImplementedError(f"Amount of optimizer bits not supported: {optim_bits}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            k=0,
            weight_sum=0.0,
            train_mode=True,
        )
        super().__init__(params, defaults, optim_bits=optim_bits, is_paged=False)

        if args is None:
            self.args = MockArgs(
                {
                    "optim_bits": optim_bits,
                    "min_8bit_size": min_8bit_size,
                    "max_unorm": 0.0,
                    "skip_zeros": False,
                }
            )
        else:
            self.args = args

        self.blocksize = blocksize
        self.mng = GlobalOptimManager.get_instance()

    def _is_8bit_param(self, gindex: int, pindex: int, group: dict, p: torch.Tensor) -> bool:
        config = self.get_config(gindex, pindex, group)
        return config["optim_bits"] == 8 and p.numel() >= config["min_8bit_size"]

    def _dequantize_state(self, state: dict, key_state: str, key_absmax: str, key_qmap: str) -> torch.Tensor:
        buf = state[key_state]
        if buf.dtype == torch.uint8:
            return F.dequantize_blockwise(
                buf,
                absmax=state[key_absmax],
                code=state[key_qmap],
                blocksize=self.blocksize,
            )
        return buf

    def _quantize_state(self, tensor: torch.Tensor, state: dict, key_state: str, key_absmax: str, key_qmap: str) -> None:
        if state[key_state].dtype == torch.uint8:
            q, qs = F.quantize_blockwise(
                tensor,
                code=state[key_qmap],
                blocksize=self.blocksize,
            )
            state[key_state] = q
            state[key_absmax] = qs.absmax
        else:
            state[key_state].copy_(tensor)

    @torch.no_grad()
    def eval(self) -> None:
        """Switch parameters to the averaged evaluation iterate ``x``."""
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1 = group["betas"][0]
            for gindex, p in enumerate(group["params"]):
                state = self.state[p]
                if "state1" in state:
                    z = self._dequantize_state(state, "state1", "absmax1", "qmap1")
                    # y = (1 - beta1) * z + beta1 * x  =>  x = lerp(y, z, 1 - 1 / beta1)
                    p.data.lerp_(end=z.to(dtype=p.dtype), weight=1.0 - 1.0 / beta1)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Switch parameters back to the interpolated training iterate ``y``."""
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1 = group["betas"][0]
            for gindex, p in enumerate(group["params"]):
                state = self.state[p]
                if "state1" in state:
                    z = self._dequantize_state(state, "state1", "absmax1", "qmap1")
                    # y = (1 - beta1) * z + beta1 * x  =>  y = lerp(x, z, 1 - beta1)
                    p.data.lerp_(end=z.to(dtype=p.dtype), weight=1.0 - beta1)
            group["train_mode"] = True

    @torch.no_grad()
    def _init_param_state(self, gindex: int, pindex: int, group: dict, p: torch.Tensor) -> dict:
        state = self.state[p]
        state["step"] = 0
        use_8bit = self._is_8bit_param(gindex, pindex, group, p)

        z_init = p.data.detach().to(torch.float32)
        v_init = torch.zeros_like(z_init)

        if use_8bit:
            if "dynamic" not in self.name2qmap:
                self.fill_qmap()
            qmap1 = self.name2qmap["dynamic"].to(p.device)
            qmap2 = self.name2qmap["udynamic"].to(p.device)

            q1, qs1 = F.quantize_blockwise(z_init, code=qmap1, blocksize=self.blocksize)
            q2, qs2 = F.quantize_blockwise(v_init, code=qmap2, blocksize=self.blocksize)

            state["state1"] = q1
            state["absmax1"] = qs1.absmax
            state["qmap1"] = qmap1

            state["state2"] = q2
            state["absmax2"] = qs2.absmax
            state["qmap2"] = qmap2
        else:
            state["state1"] = z_init.clone()
            state["state2"] = v_init

        return state

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single Schedule-Free AdamW optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for gindex, group in enumerate(self.param_groups):
            if not group["train_mode"]:
                raise RuntimeError(
                    "Optimizer is not in train mode. Call optimizer.train() before optimizer.step()."
                )

            beta1, beta2 = group["betas"]
            eps = group["eps"]
            decay = group["weight_decay"]
            k = group["k"]
            warmup_steps = group["warmup_steps"]
            r = group["r"]
            weight_lr_power = group["weight_lr_power"]

            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            bias_correction2 = 1.0 - beta2 ** (k + 1)
            lr = group["lr"] * sched * math.sqrt(bias_correction2)

            weight = ((k + 1) ** r) * (lr ** weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / weight_sum if weight_sum > 0.0 else 0.0

            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamWScheduleFree does not support sparse gradients.")

                state = self.state[p]
                if "state1" not in state:
                    state = self._init_param_state(gindex, pindex, group, p)

                state["step"] += 1

                y = p.data.to(torch.float32)
                grad = p.grad.data.to(torch.float32)
                z = self._dequantize_state(state, "state1", "absmax1", "qmap1").to(torch.float32)
                exp_avg_sq = self._dequantize_state(state, "state2", "absmax2", "qmap2").to(torch.float32)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                grad_norm = grad.div(denom)
                if decay != 0.0:
                    grad_norm.add_(y, alpha=decay)

                y.lerp_(end=z, weight=ckp1)
                y.add_(grad_norm, alpha=lr * (beta1 * (1.0 - ckp1) - 1.0))
                z.sub_(grad_norm, alpha=lr)

                p.data.copy_(y.to(dtype=p.dtype))
                self._quantize_state(z, state, "state1", "absmax1", "qmap1")
                self._quantize_state(exp_avg_sq, state, "state2", "absmax2", "qmap2")

            group["k"] = k + 1

        return loss


class AdamWScheduleFree8bit(AdamWScheduleFree):
    """8-bit blockwise-quantized Schedule-Free AdamW optimizer."""

    def __init__(
        self,
        params,
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        args: Optional[object] = None,
        min_8bit_size: int = 4096,
        blocksize: int = 256,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            optim_bits=8,
            args=args,
            min_8bit_size=min_8bit_size,
            blocksize=blocksize,
        )


class AdamWScheduleFree32bit(AdamWScheduleFree):
    """32-bit Schedule-Free AdamW optimizer."""

    def __init__(
        self,
        params,
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        args: Optional[object] = None,
        min_8bit_size: int = 4096,
        blocksize: int = 256,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            optim_bits=32,
            args=args,
            min_8bit_size=min_8bit_size,
            blocksize=blocksize,
        )
