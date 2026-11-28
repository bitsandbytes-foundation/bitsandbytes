import copy
import math

import pytest
import torch
import torch.nn as nn

import bitsandbytes as bnb
from bitsandbytes.optim import CAdamW, CAdamW8bit, CAdamW32bit


def _reference_c_adamw(
    params_init: list[torch.Tensor],
    grads_list: list[list[torch.Tensor]],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
):
    beta1, beta2 = betas
    ps = [p.clone().to(torch.float32) for p in params_init]
    ms = [torch.zeros_like(p, dtype=torch.float32) for p in params_init]
    vs = [torch.zeros_like(p, dtype=torch.float32) for p in params_init]

    for step, grads in enumerate(grads_list, 1):
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
        step_size = lr / bc1

        for p, m, v, g in zip(ps, ms, vs, grads):
            g32 = g.to(torch.float32)
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)
            m.lerp_(g32, 1.0 - beta1)
            v.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
            denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
            mask = (m * g32 > 0).to(torch.float32)
            mask.div_(mask.mean().clamp_(min=1e-3))
            p.add_((m * mask) / denom, alpha=-step_size)

    return ps


@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_c_adamw_32bit_matches_reference(weight_decay):
    torch.manual_seed(0)
    p1 = nn.Parameter(torch.randn(64, 64))
    p2 = nn.Parameter(torch.randn(64))
    params_init = [p1.detach().clone(), p2.detach().clone()]

    opt = CAdamW32bit([p1, p2], lr=1e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=weight_decay)

    grads_list = []
    for _ in range(6):
        g1 = torch.randn_like(p1)
        g2 = torch.randn_like(p2)
        grads_list.append([g1.clone(), g2.clone()])
        p1.grad = g1
        p2.grad = g2
        opt.step()

    ref_ps = _reference_c_adamw(
        params_init,
        grads_list,
        lr=1e-2,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    torch.testing.assert_close(p1.data, ref_ps[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(p2.data, ref_ps[1], atol=1e-6, rtol=1e-5)


def test_c_adamw_8bit_convergence_and_memory():
    torch.manual_seed(42)
    x = torch.randn(64, 128)
    target = torch.randn(64, 64)

    model32 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
    model8 = copy.deepcopy(model32)

    opt32 = CAdamW32bit(model32.parameters(), lr=5e-3)
    opt8 = CAdamW8bit(model8.parameters(), lr=5e-3, min_8bit_size=4096)

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

    final_loss32 = loss_fn(model32(x), target).item()
    final_loss8 = loss_fn(model8(x), target).item()

    assert final_loss8 < init_loss * 0.5
    assert abs(final_loss8 - final_loss32) < 0.05

    w_state = opt8.state[model8[0].weight]
    b_state = opt8.state[model8[0].bias]
    assert w_state["state1"].dtype == torch.uint8
    assert w_state["state2"].dtype == torch.uint8
    assert "absmax1" in w_state and "absmax2" in w_state
    assert b_state["state1"].dtype == torch.float32
    assert b_state["state2"].dtype == torch.float32


def test_c_adamw_state_dict_roundtrip_and_validation():
    torch.manual_seed(7)
    p = nn.Parameter(torch.randn(64, 65))
    opt = CAdamW8bit([p], lr=1e-2, min_8bit_size=128)

    p.grad = torch.randn_like(p)
    opt.step()

    sd = opt.state_dict()
    assert bnb.optim.optimizer.Optimizer8bit._FSDP_WRAPPED_QUANT_STATE_KEY in next(iter(sd["state"].values()))

    p_clone = nn.Parameter(p.detach().clone())
    opt2 = CAdamW8bit([p_clone], lr=1e-2, min_8bit_size=128)
    opt2.load_state_dict(sd)

    g = torch.randn_like(p)
    p.grad = g.clone()
    p_clone.grad = g.clone()
    opt.step()
    opt2.step()

    torch.testing.assert_close(p.data, p_clone.data, atol=1e-6, rtol=1e-5)

    with pytest.raises(ValueError, match="learning rate"):
        CAdamW([p], lr=-1.0)
    with pytest.raises(ValueError, match="epsilon"):
        CAdamW([p], eps=-1e-8)
    with pytest.raises(ValueError, match="beta parameter"):
        CAdamW([p], betas=(-0.1, 0.999))
    with pytest.raises(ValueError, match="weight_decay"):
        CAdamW([p], weight_decay=-0.1)
    with pytest.raises(ValueError, match="blocksize"):
        CAdamW([p], blocksize=123)
    with pytest.raises(NotImplementedError, match="optimizer bits"):
        CAdamW([p], optim_bits=4)
