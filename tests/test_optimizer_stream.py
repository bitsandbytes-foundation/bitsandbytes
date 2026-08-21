import importlib

import pytest
import torch

import bitsandbytes as bnb

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)


def _optimizer_update_32bit(g, p, state1, state2):
    torch.ops.bitsandbytes.optimizer_update_32bit(
        "adam",
        g,
        p,
        state1,
        state2,
        None,
        0.0,
        0.0,
        0.9,
        0.999,
        0.0,
        0.0,
        1e-8,
        0.0,
        1,
        1e-3,
        1.0,
        False,
    )


def _optimizer_update_8bit(g, p, state1, state2, qmap1, qmap2, absmax1, absmax2):
    torch.ops.bitsandbytes.optimizer_update_8bit_blockwise(
        "adam",
        g,
        p,
        state1,
        state2,
        0.9,
        0.999,
        0.0,
        0.0,
        1e-8,
        1,
        1e-3,
        qmap1,
        qmap2,
        absmax1,
        absmax2,
        0.0,
        1.0,
        False,
    )


def _make_inputs(family):
    torch.manual_seed(0)
    n = 4097
    p = torch.randn(n, device="cuda", dtype=torch.float32)
    g = torch.zeros_like(p)
    final_g = torch.randn_like(p) * 0.01
    if family == "32bit":
        return [g, p, torch.zeros_like(p), torch.zeros_like(p)], final_g

    blocks = (n + 255) // 256
    qmap1 = bnb.functional.create_dynamic_map(signed=True).to("cuda")
    qmap2 = bnb.functional.create_dynamic_map(signed=False).to("cuda")
    return [
        g,
        p,
        torch.zeros(n, device="cuda", dtype=torch.uint8),
        torch.zeros(n, device="cuda", dtype=torch.uint8),
        qmap1,
        qmap2,
        torch.zeros(blocks, device="cuda", dtype=torch.float32),
        torch.zeros(blocks, device="cuda", dtype=torch.float32),
    ], final_g


def _run_update(family, values):
    if family == "32bit":
        _optimizer_update_32bit(*values)
    else:
        _optimizer_update_8bit(*values)


@pytest.mark.parametrize("family", ["32bit", "8bit"])
def test_optimizer_update_uses_current_stream(family):
    for _ in range(2):
        inputs, final_g = _make_inputs(family)
        reference = [value.clone() for value in inputs]
        reference[0].copy_(final_g)
        _run_update(family, reference)
        torch.cuda.synchronize()

        blocker = torch.cuda.Stream()
        caller = torch.cuda.Stream()
        gate = torch.cuda.Event()
        done = torch.cuda.Event()
        with torch.cuda.stream(blocker):
            torch.cuda._sleep(20_000_000)
            gate.record()
        assert not gate.query()

        with torch.cuda.stream(caller):
            caller.wait_event(gate)
            inputs[0].copy_(final_g)
            _run_update(family, inputs)
            done.record()
        done.synchronize()

        for actual, expected in zip(inputs, reference):
            assert torch.equal(actual, expected)


def test_optimizer_stream_symbols_are_additive_and_selected():
    from bitsandbytes.backends.cuda import ops as cuda_ops
    from bitsandbytes.cextension import lib

    symbol_pairs = (
        ("cadam32bit_grad_fp32", "cadam32bit_grad_fp32_with_stream"),
        ("clion32bit_grad_fp16", "clion32bit_grad_fp16_with_stream"),
        ("cadam_8bit_blockwise_grad_fp32", "cadam_8bit_blockwise_grad_fp32_with_stream"),
        ("clion_8bit_blockwise_grad_bf16", "clion_8bit_blockwise_grad_bf16_with_stream"),
    )
    for legacy, stream_aware in symbol_pairs:
        assert getattr(lib, legacy) is not None
        assert getattr(lib, stream_aware) is not None

    assert cuda_ops.str2optimizer32bit["adam"][0] is lib.cadam32bit_grad_fp32_with_stream
    assert cuda_ops.str2optimizer8bit_blockwise["adam"][0] is lib.cadam_8bit_blockwise_grad_fp32_with_stream


@pytest.mark.parametrize("paged,expected_syncs", [(False, 0), (True, 3)])
def test_optimizer_step_sync_policy(monkeypatch, paged, expected_syncs):
    optimizer_module = importlib.import_module("bitsandbytes.optim.optimizer")
    real_sync = optimizer_module.sync_gpu
    calls = []

    def recording_sync(tensor):
        calls.append(tensor)
        real_sync(tensor)

    monkeypatch.setattr(optimizer_module, "sync_gpu", recording_sync)
    params = [torch.nn.Parameter(torch.randn(512, device="cuda")) for _ in range(2)]
    optimizer_cls = bnb.optim.PagedAdamW32bit if paged else bnb.optim.AdamW32bit
    optimizer = optimizer_cls(params, lr=1e-3)
    for param in params:
        param.grad = torch.randn_like(param)

    optimizer.step()
    torch.cuda.synchronize()
    assert len(calls) == expected_syncs
