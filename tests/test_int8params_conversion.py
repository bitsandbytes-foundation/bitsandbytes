import copy

import pytest
import torch

import bitsandbytes as bnb
from tests.helpers import get_available_devices


def _convert_without_changing_storage(weight, conversion):
    if conversion == "no_args":
        return weight.to()
    if conversion == "dtype_only":
        return weight.to(dtype=torch.int8)
    if conversion == "non_blocking":
        return weight.to(non_blocking=True)
    return weight.to(device=weight.device)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("conversion", ["no_args", "dtype_only", "non_blocking", "explicit_device"])
def test_int8params_to_preserves_scales(device, conversion):
    torch.manual_seed(13)
    weight = bnb.nn.Int8Params(torch.randn(16, 32), requires_grad=False, has_fp16_weights=False).to(device)
    converted = _convert_without_changing_storage(weight, conversion)
    assert isinstance(converted, bnb.nn.Int8Params)
    assert converted.SCB is not None
    assert converted.SCB.dtype == weight.SCB.dtype
    assert converted.SCB.device == converted.device == weight.device
    assert converted.CB.data_ptr() == converted.data.data_ptr()
    assert converted.has_fp16_weights == weight.has_fp16_weights
    torch.testing.assert_close(converted.data, weight.data, rtol=0, atol=0)
    torch.testing.assert_close(converted.SCB, weight.SCB, rtol=0, atol=0)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("conversion", ["no_args", "dtype_only", "non_blocking", "explicit_device"])
@pytest.mark.parametrize("kind", ["embedding", "linear"])
def test_int8params_to_model_continuation(device, conversion, kind):
    torch.manual_seed(17)
    if kind == "embedding":
        original = bnb.nn.Embedding8bit(16, 32).to(device)
        inputs = torch.tensor([[1, 4, 7], [9, 2, 1]], device=device)
    else:
        original = bnb.nn.Linear8bitLt(16, 32, bias=False, has_fp16_weights=False).to(device)
        inputs = torch.randn(2, 3, 16, device=device)

    # Convert before the first forward, so a cached matmul state cannot hide
    # missing quantization statistics on the parameter.
    converted = copy.deepcopy(original)
    converted.weight = _convert_without_changing_storage(converted.weight, conversion)
    original_head = torch.nn.Linear(32, 4, device=device)
    converted_head = copy.deepcopy(original_head)
    original_optimizer = torch.optim.SGD(original_head.parameters(), lr=0.05)
    converted_optimizer = torch.optim.SGD(converted_head.parameters(), lr=0.05)
    for _ in range(2):
        expected = original_head(original(inputs))
        actual = converted_head(converted(inputs))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        original_optimizer.zero_grad()
        converted_optimizer.zero_grad()
        expected.square().mean().backward()
        actual.square().mean().backward()
        for actual_parameter, expected_parameter in zip(converted_head.parameters(), original_head.parameters()):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)
        original_optimizer.step()
        converted_optimizer.step()
        for actual_parameter, expected_parameter in zip(converted_head.parameters(), original_head.parameters()):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)


@pytest.mark.parametrize("conversion", ["no_args", "dtype_only", "non_blocking"])
def test_int8params_to_preserves_meta_scales(conversion):
    weight = bnb.nn.Int8Params(
        torch.empty(16, 32, device="meta", dtype=torch.int8),
        requires_grad=False,
        SCB=torch.empty(16, device="meta"),
    )
    converted = _convert_without_changing_storage(weight, conversion)
    assert converted.SCB is weight.SCB
    assert converted.CB.device.type == "meta"
