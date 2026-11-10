from concurrent.futures import ThreadPoolExecutor
import logging
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from bitsandbytes.autograd import _functions as af
from tests.helpers import get_available_devices


@pytest.fixture
def cast_warning(monkeypatch, caplog):
    # Stop after the warning so logging tests need no quantization kernels.
    def stop_quantization(*args, **kwargs):
        raise StopIteration

    monkeypatch.setattr(af.F, "int8_vectorwise_quant", stop_quantization)
    monkeypatch.setattr(af, "_int8_cast_warned_dtypes", set())
    make_record = Mock(wraps=af.logger.makeRecord)
    monkeypatch.setattr(af.logger, "makeRecord", make_record)
    caplog.set_level(logging.WARNING, logger=af.logger.name)

    def forward(dtype=torch.float32):
        with pytest.raises(StopIteration):
            af.MatMul8bitLt.forward(
                SimpleNamespace(needs_input_grad=(False, False)),
                torch.ones(1, 2, dtype=dtype),
                torch.ones(1, 2),
            )

    def reset():
        # The repository's pytest_runtest_call hook also invokes the test body.
        af._int8_cast_warned_dtypes.clear()
        make_record.reset_mock()
        caplog.clear()
        return forward, make_record

    return reset


def test_cast_warning_once_per_dtype(cast_warning, caplog):
    forward, make_record = cast_warning()
    for _ in range(100):
        forward(torch.float32)
        forward(torch.bfloat16)
        forward(torch.float16)
    assert make_record.call_count == 2
    assert len(caplog.records) == 2
    assert "torch.float32" in caplog.messages[0]
    assert "torch.bfloat16" in caplog.messages[1]


def test_cast_warning_concurrent(cast_warning, caplog):
    forward, make_record = cast_warning()
    barrier = Barrier(16)

    def call():
        barrier.wait(timeout=10)
        forward()

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: call(), range(16)))
    assert make_record.call_count == 1
    assert len(caplog.records) == 1


def test_cast_warning_logging_level(cast_warning, caplog):
    forward, make_record = cast_warning()
    with caplog.at_level(logging.ERROR, logger=af.logger.name):
        forward()
    assert make_record.call_count == 0
    forward()
    forward()
    af.logger.warning("unrelated warning")
    af.logger.warning("unrelated warning")
    assert make_record.call_count == 3
    assert caplog.messages[-2:] == ["unrelated warning", "unrelated warning"]


def test_cast_warning_compilation_guard(cast_warning, caplog, monkeypatch):
    forward, make_record = cast_warning()
    monkeypatch.setattr(af, "_is_compiling", lambda: True)
    forward()
    assert make_record.call_count == 0
    monkeypatch.setattr(af, "_is_compiling", lambda: False)
    forward()
    forward()
    assert make_record.call_count == 1


def test_cast_warning_empty_input(cast_warning):
    forward, make_record = cast_warning()
    output = af.MatMul8bitLt.forward(SimpleNamespace(), torch.empty(0, 32), torch.ones(16, 32))
    assert output.shape == (0, 16)
    assert make_record.call_count == 0
    forward()
    assert make_record.call_count == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("threshold", [0.0, 6.0])
@pytest.mark.parametrize("weight_grad", [False, True])
def test_cast_warning_preserves_forward_backward(dtype, device, threshold, weight_grad, monkeypatch):
    if device == "cpu" and weight_grad:
        pytest.skip("The CPU backend does not implement int8_double_quant for weight gradients")
    monkeypatch.setattr(af, "_int8_cast_warned_dtypes", set())
    a = torch.randn(2, 32, device=device, dtype=dtype, requires_grad=True)
    with torch.no_grad():
        a[:, 0] = 8.0
    b = torch.randn(32, 32, device=device, dtype=torch.float16, requires_grad=weight_grad)
    bias = torch.randn(32, device=device, dtype=dtype, requires_grad=True)

    def run():
        state = af.MatmulLtState()
        state.threshold = threshold
        output = af.MatMul8bitLt.apply(a, b, None, bias, state)
        grads = torch.autograd.grad(output.sum(), (a, b, bias) if weight_grad else (a, bias))
        return output.detach(), *grads

    first = run()
    repeated = run()
    with monkeypatch.context() as patch:
        patch.setattr(
            af,
            "_warn_int8_cast_once",
            lambda dtype: af.logger.warning(
                "MatMul8bitLt: inputs will be cast from %s to float16 during quantization", dtype
            ),
        )
        original_logging = run()
    for actual, expected, original in zip(first, repeated, original_logging):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.skipif(torch.__version__ < (2, 5), reason="Fullgraph MatmulLtState tracing requires torch >= 2.5")
@pytest.mark.parametrize("device", get_available_devices())
def test_cast_warning_torch_compile(device, monkeypatch, caplog):
    monkeypatch.setattr(af, "_int8_cast_warned_dtypes", set())
    caplog.set_level(logging.WARNING, logger=af.logger.name)
    caplog.clear()

    def forward(a, b):
        return af.MatMul8bitLt.apply(a, b, None, None, state)

    a = torch.randn(2, 32, device=device, requires_grad=True)
    b = torch.randn(16, 32, device=device, dtype=torch.float16)
    state = af.MatmulLtState()
    state.has_fp16_weights = False
    state.CB, state.SCB, _ = af.F.int8_vectorwise_quant(b)
    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    actual = compiled(a, b)
    actual_grad = torch.autograd.grad(actual.sum(), a)[0]
    repeated = compiled(a, b)
    assert not [r for r in caplog.records if r.name == af.logger.name]
    expected = forward(a, b)
    expected_grad = torch.autograd.grad(expected.sum(), a)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
