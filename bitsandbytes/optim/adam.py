# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

import math
import os

import torch
import torch.distributed as dist
from bitsandbytes.optim.optimizer import Optimizer2State
import bitsandbytes.functional as F

class Adam(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, optim_bits=32, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        super(Adam, self).__init__('adam', params, lr, betas, eps,
                weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise)

class Adam8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        super(Adam8bit, self).__init__('adam', params, lr, betas, eps,
                weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise)

class Adam32bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        super(Adam32bit, self).__init__('adam', params, lr, betas, eps,
                weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise)


class AnalysisAdam(torch.optim.Optimizer):
    """Adam that performs 8-bit vs 32-bit error analysis.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        bnb_analysis='dynamic-blockwise',
        savedir=None
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(AnalysisAdam, self).__init__(params, defaults)
        self.analysis = bnb_analysis
        self.savedir = savedir

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p_id, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group.get("amsgrad", False)
                assert not amsgrad

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    state['abserrors'] = torch.zeros((256, 256), device=p_data_fp32.device)
                    state['relerrors'] = torch.zeros((256, 256), device=p_data_fp32.device)
                    state['counts'] = torch.zeros((256, 256), device=p_data_fp32.device)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            p_data_fp32
                        )

                state["step"] += 1
                beta1, beta2 = group["betas"]
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                e = state['abserrors']
                rele = state['relerrors']
                counts = state['counts']

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]


                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                update_fp32 = exp_avg/denom

                if p_data_fp32.numel() <= 8192 or p_data_fp32.numel() > 50000*1000:
                    # embedding layer or too small
                    p_data_fp32 += -step_size*update_fp32
                else:
                    if self.analysis == 'dynamic-blockwise':
                        code1 = F.create_dynamic_map(signed=True).to(p.device)
                        code2 = F.create_dynamic_map(signed=False).to(p.device)
                        C1, S1 = F.quantize_blockwise(exp_avg, code=code1)
                        state1 = F.dequantize_blockwise(C1, S1)
                        C2, S2 = F.quantize_blockwise(exp_avg_sq, code=code2)
                        state2 = F.dequantize_blockwise(C2, S2)
                    elif self.analysis == 'dynamic':
                        code1 = F.create_dynamic_map(signed=True).to(p.device)
                        code2 = F.create_dynamic_map(signed=False).to(p.device)
                        C1, S1 = F.quantize(exp_avg, code=code1)
                        state1 = F.dequantize(C1, S1)
                        C2, S2 = F.quantize(exp_avg_sq, code=code2)
                        state2 = F.dequantize(C2, S2)
                    elif self.analysis == 'linear':
                        code1 = F.create_linear_map(signed=True).to(p.device)
                        code2 = F.create_linear_map(signed=False).to(p.device)
                        C1, S1 = F.quantize(exp_avg, code=code1)
                        state1 = F.dequantize(C1, S1)
                        C2, S2 = F.quantize(exp_avg_sq, code=code2)
                        state2 = F.dequantize(C2, S2)
                    elif self.analysis == 'quantile':
                        code1 = F.estimate_quantiles(exp_avg)
                        code2 = F.estimate_quantiles(exp_avg_sq)
                        C1 = F.quantize_no_absmax(exp_avg, code=code1)
                        state1 = F.dequantize_no_absmax(C1, code1)
                        C2 = F.quantize_no_absmax(exp_avg_sq, code=code2)
                        state2 = F.dequantize_no_absmax(C2, code2)
                    elif self.analysis == 'my-quantization-routine':
                        pass
                        # 1. get code
                        # 2. quantize
                        # 3. dequantize
                        # Error will be calculated automatically!
                    else:
                        raise ValueError(f'Invalid analysis value: {self.analysis}!')

                    denom = state2.sqrt().add_(group["eps"])
                    update_8bit = state1/denom

                    abserr = torch.abs(update_8bit-update_fp32)
                    relerr = abserr/torch.abs(update_fp32+1e-6)

                    C1, C2 = C1.int(), C2.int()

                    F.histogram_scatter_add_2d(e, C1.int(), C2.int(), abserr)
                    F.histogram_scatter_add_2d(rele, C1.int(), C2.int(), relerr)
                    F.histogram_scatter_add_2d(counts, C1.int(), C2.int(), torch.ones_like(abserr))

                    p_data_fp32 += -step_size*update_fp32


                    if not dist.is_initialized() or dist.get_rank() == 0:
                        if self.savedir != '' and state['step'] % 100 == 0:
                            if not os.path.exists(self.savedir): os.makedirs(self.savedir)
                            shapestr = '_'.join([str(dim) for dim in p_data_fp32.shape])
                            pathe = os.path.join(self.savedir, f'{p_id}_{shapestr}_abserr.pkl')
                            pathrele = os.path.join(self.savedir, f'{p_id}_{shapestr}_relerr.pkl')
                            pathcounts = os.path.join(self.savedir, f'{p_id}_{shapestr}_counts.pkl')
                            torch.save(e, pathe)
                            torch.save(rele, pathrele)
                            torch.save(counts, pathcounts)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)



        return loss
