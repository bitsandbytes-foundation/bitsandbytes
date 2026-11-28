import copy
import math

import pytest
import torch
import torch.nn as nn

import bitsandbytes as bnb
from bitsandbytes.optim import AdamWScheduleFree, AdamWScheduleFree8bit, AdamWScheduleFree32bit


def _reference_schedule_free_adamw(
    params_init: list[torch.Tensor],
    grads_list: list[list[torch.Tensor]],
    lr: float = 0.0025,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    r: float = 0.0,
    weight_lr_power: float = 2.0,
):
    """Pure-PyTorch reference implementation of Schedule-Free AdamW."""
    beta1, beta2 = betas
    ys = [p.clone().to(torch.float32) for p in params_init]
    zs = [p.clone().to(torch.float32) for p in params_init]
    vs = [torch.zeros_like(p, dtype=torch.float32) for p in params_init]
    weight_sum = 0.0

    for k, grads in enumerate(grads_list):
        sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
        step_lr = lr * sched * math.sqrt(1.0 - beta2 ** (k + 1))
        weight = ((k + 1) ** r) * (step_lr ** weight_lr_power)
        weight_sum += weight
        ckp1 = weight / weight_sum if weight_sum > 0.0 else 0.0

        for y, z, v, g in zip(ys, zs, vs, grads):
            g32 = g.to(torch.float32)
            v.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
            denom = v.sqrt().add_(eps)
            grad_norm = g32 / denom
            if weight_decay != 0.0:
                grad_norm.add_(y, alpha=weight_decay)
            y.lerp_(end=z, weight=ckp1)
            y.add_(grad_norm, alpha=step_lr * (beta1 * (1.0 - ckp1) - 1.0))
            z.sub_(grad_norm, alpha=step_lr)

    eval_xs = [y.clone().lerp_(end=z, weight=1.0 - 1.0 / beta1) for y, z in zip(ys, zs)]
    return ys, eval_xs


@pytest.mark.parametrize("warmup_steps", [0, 3])
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_schedule_free_32bit_matches_reference(warmup_steps, weight_decay):
    torch.manual_seed(0)
    p1 = nn.Parameter(torch.randn(64, 64))
    p2 = nn.Parameter(torch.randn(64))
    params_init = [p1.detach().clone(), p2.detach().clone()]

    opt = AdamWScheduleFree32bit(
        [p1, p2],
        lr=1e-2,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
    )

    grads_list = []
    for _ in range(6):
        g1 = torch.randn_like(p1)
        g2 = torch.randn_like(p2)
        grads_list.append([g1.clone(), g2.clone()])
        p1.grad = g1
        p2.grad = g2
        opt.step()

    ref_ys, ref_xs = _reference_schedule_free_adamw(
        params_init,
        grads_list,
        lr=1e-2,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
    )

    torch.testing.assert_close(p1.data, ref_ys[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(p2.data, ref_ys[1], atol=1e-6, rtol=1e-5)

    opt.eval()
    torch.testing.assert_close(p1.data, ref_xs[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(p2.data, ref_xs[1], atol=1e-6, rtol=1e-5)

    opt.train()
    torch.testing.assert_close(p1.data, ref_ys[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(p2.data, ref_ys[1], atol=1e-6, rtol=1e-5)


def test_schedule_free_8bit_convergence_and_memory():
    torch.manual_seed(42)
    x = torch.randn(64, 128)
    target = torch.randn(64, 64)

    model32 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
    model8 = copy.deepcopy(model32)

    opt32 = AdamWScheduleFree32bit(model32.parameters(), lr=5e-3, warmup_steps=5)
    opt8 = AdamWScheduleFree8bit(model8.parameters(), lr=5e-3, warmup_steps=5, min_8bit_size=4096)

    loss_fn = nn.MSELoss()
    init_loss = loss_fn(model8(x), target).item()

    for _ in range(25):
        opt32.zero_grad()
        loss32 = loss_fn(model32(x), target)
        loss32.backward()
        opt32.step()

        opt8.zero_grad()
        loss8 = loss_fn(model8(x), target)
        loss8.backward()
        opt8.step()

    opt32.eval()
    opt8.eval()
    final_loss32 = loss_fn(model32(x), target).item()
    final_loss8 = loss_fn(model8(x), target).item()

    assert final_loss8 < init_loss * 0.5
    assert abs(final_loss8 - final_loss32) < 0.05

    # Verify large weight matrices use uint8 state while small biases stay fp32
    w_state = opt8.state[model8[0].weight]
    b_state = opt8.state[model8[0].bias]
    assert w_state["state1"].dtype == torch.uint8
    assert w_state["state2"].dtype == torch.uint8
    assert "absmax1" in w_state and "absmax2" in w_state
    assert b_state["state1"].dtype == torch.float32
    assert b_state["state2"].dtype == torch.float32


def test_schedule_free_eval_guard_and_state_dict_roundtrip():
    torch.manual_seed(7)
    p = nn.Parameter(torch.randn(64, 65))  # non-multiple of blocksize=256
    opt = AdamWScheduleFree8bit([p], lr=1e-2, min_8bit_size=128)

    p.grad = torch.randn_like(p)
    opt.step()

    opt.eval()
    with pytest.raises(RuntimeError, match="not in train mode"):
        opt.step()
    opt.train()

    sd = opt.state_dict()
    assert bnb.optim.optimizer.Optimizer8bit._FSDP_WRAPPED_QUANT_STATE_KEY in next(iter(sd["state"].values()))

    p_clone = nn.Parameter(p.detach().clone())
    opt2 = AdamWScheduleFree8bit([p_clone], lr=1e-2, min_8bit_size=128)
    opt2.load_state_dict(sd)

    g = torch.randn_like(p)
    p.grad = g.clone()
    p_clone.grad = g.clone()
    opt.step()
    opt2.step()

    torch.testing.assert_close(p.data, p_clone.data, atol=1e-6, rtol=1e-5)


def test_schedule_free_validation():
    p = [nn.Parameter(torch.zeros(4))]
    with pytest.raises(ValueError, match="learning rate"):
        AdamWScheduleFree(p, lr=-1.0)
    with pytest.raises(ValueError, match="epsilon"):
        AdamWScheduleFree(p, eps=-1e-8)
    with pytest.raises(ValueError, match="beta parameter at index 0"):
        AdamWScheduleFree(p, betas=(0.0, 0.999))
    with pytest.raises(ValueError, match="beta parameter at index 1"):
        AdamWScheduleFree(p, betas=(0.9, 1.0))
    with pytest.raises(ValueError, match="weight_decay"):
        AdamWScheduleFree(p, weight_decay=-0.1)
    with pytest.raises(ValueError, match="blocksize"):
        AdamWScheduleFree(p, blocksize=123)
    with pytest.raises(NotImplementedError, match="optimizer bits"):
        AdamWScheduleFree(p, optim_bits=4)
