# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch

import bitsandbytes.functional as F
from bitsandbytes.optim.optimizer import GlobalOptimManager, MockArgs, Optimizer8bit


class CAdamW(Optimizer8bit):
    """Cautious AdamW (C-AdamW) optimizer with optional 8-bit blockwise quantized state.

    Implements Cautious AdamW from *"Cautious Optimizers: Improving Training with
    One Line of Code"* (Liang et al., 2024; https://github.com/kyleliang919/C-Optim),
    masking updates to coordinates where the momentum ``exp_avg`` and current gradient
    ``grad`` share the same sign, and rescaling by ``1 / clamp(mean(mask), min=1e-3)``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        optim_bits: int = 32,
        args: Optional[object] = None,
        min_8bit_size: int = 4096,
        blocksize: int = 256,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        for i in range(len(betas)):
            if not 0.0 <= betas[i] < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {betas[i]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if blocksize not in (64, 128, 256, 512, 1024, 2048, 4096):
            raise ValueError(
                f"Invalid blocksize: {blocksize}. Must be one of (64, 128, 256, 512, 1024, 2048, 4096)."
            )
        if optim_bits not in (8, 32):
            raise NotImplementedError(f"Amount of optimizer bits not supported: {optim_bits}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
    def _init_param_state(self, gindex: int, pindex: int, group: dict, p: torch.Tensor) -> dict:
        state = self.state[p]
        state["step"] = 0
        use_8bit = self._is_8bit_param(gindex, pindex, group, p)

        zeros = torch.zeros_like(p.data, dtype=torch.float32)
        if use_8bit:
            if "dynamic" not in self.name2qmap:
                self.fill_qmap()
            qmap1 = self.name2qmap["dynamic"].to(p.device)
            qmap2 = self.name2qmap["udynamic"].to(p.device)

            q1, qs1 = F.quantize_blockwise(zeros, code=qmap1, blocksize=self.blocksize)
            q2, qs2 = F.quantize_blockwise(zeros, code=qmap2, blocksize=self.blocksize)

            state["state1"] = q1
            state["absmax1"] = qs1.absmax
            state["qmap1"] = qmap1

            state["state2"] = q2
            state["absmax2"] = qs2.absmax
            state["qmap2"] = qmap2
        else:
            state["state1"] = zeros.clone()
            state["state2"] = zeros.clone()

        return state

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single Cautious AdamW optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for gindex, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("CAdamW does not support sparse gradients.")

                state = self.state[p]
                if "state1" not in state:
                    state = self._init_param_state(gindex, pindex, group, p)

                state["step"] += 1
                step = state["step"]

                param = p.data.to(torch.float32)
                grad = p.grad.data.to(torch.float32)
                exp_avg = self._dequantize_state(state, "state1", "absmax1", "qmap1").to(torch.float32)
                exp_avg_sq = self._dequantize_state(state, "state2", "absmax2", "qmap2").to(torch.float32)

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Cautious alignment mask (Liang et al., 2024)
                mask = (exp_avg * grad > 0).to(torch.float32)
                mask.div_(mask.mean().clamp_(min=1e-3))

                norm_grad = (exp_avg * mask) / denom
                param.add_(norm_grad, alpha=-step_size)

                p.data.copy_(param.to(dtype=p.dtype))
                self._quantize_state(exp_avg, state, "state1", "absmax1", "qmap1")
                self._quantize_state(exp_avg_sq, state, "state2", "absmax2", "qmap2")

        return loss


class CAdamW8bit(CAdamW):
    """8-bit blockwise-quantized Cautious AdamW (C-AdamW) optimizer."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
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
            optim_bits=8,
            args=args,
            min_8bit_size=min_8bit_size,
            blocksize=blocksize,
        )


class CAdamW32bit(CAdamW):
    """32-bit Cautious AdamW (C-AdamW) optimizer."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
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
            optim_bits=32,
            args=args,
            min_8bit_size=min_8bit_size,
            blocksize=blocksize,
        )
