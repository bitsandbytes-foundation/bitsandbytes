import copy
from io import BytesIO

import pytest
import torch
from torch import nn

import bitsandbytes as bnb
from tests.helpers import get_available_devices


@pytest.fixture(autouse=True)
def reset_optimizer_overrides():
    manager = bnb.optim.GlobalOptimManager.get_instance()
    manager.initialize()
    yield
    manager.initialize()


def _models(device, dtype):
    actual = nn.Sequential(bnb.nn.StableEmbedding(32, 16, padding_idx=0), nn.Linear(16, 4)).to(device, dtype)
    expected = nn.Sequential(nn.Sequential(nn.Embedding(32, 16, padding_idx=0), nn.LayerNorm(16)), nn.Linear(16, 4))
    expected = expected.to(device, dtype)
    with torch.no_grad():
        expected[0][0].weight.copy_(actual[0].weight)
    expected[0][1].load_state_dict(actual[0].norm.state_dict())
    expected[1].load_state_dict(actual[1].state_dict())
    return actual, expected


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("convenience_method", [False, True])
def test_converted_stable_embedding_training(device, dtype, convenience_method):
    if device == "mps" and dtype == torch.float64:
        pytest.skip("MPS does not support float64")
    actual, expected = _models(device, torch.float32 if convenience_method else dtype)
    if convenience_method:
        method = {torch.float16: "half", torch.bfloat16: "bfloat16", torch.float32: "float", torch.float64: "double"}[
            dtype
        ]
        getattr(actual, method)()
        getattr(expected, method)()
    inputs = torch.tensor([[0, 1, 4, 4], [2, 3, 0, 7]], device=device)
    target = torch.randn(2, 4, 4, device=device)
    optimizer = torch.optim.SGD(actual.parameters(), lr=0.025, momentum=0.9)
    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=0.025, momentum=0.9)
    for _ in range(3):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        output, reference_output = actual(inputs), expected(inputs)
        assert output.dtype == dtype
        torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
        loss = (output.float() - target).square().mean()
        reference_loss = (reference_output.float() - target).square().mean()
        loss.backward()
        reference_loss.backward()
        for parameter, reference_parameter in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)
        assert torch.count_nonzero(actual[0].weight.grad[0]) == 0
        optimizer.step()
        reference_optimizer.step()
        for parameter, reference_parameter in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                reference_optimizer.state[reference_parameter]["momentum_buffer"],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_pretrained_dtype_preserves_full_precision_normalization(dtype):
    layer = bnb.nn.StableEmbedding(32, 16, _weight=torch.randn(32, 16, dtype=dtype))
    assert layer.norm.weight.dtype == torch.get_default_dtype()
    inputs = torch.tensor([[1, 3, 5], [2, 4, 6]])
    embeddings = layer.weight[inputs].to(layer.norm.weight.dtype)
    centered = embeddings - embeddings.mean(dim=-1, keepdim=True)
    expected = centered * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + layer.norm.eps)
    expected = (expected * layer.norm.weight + layer.norm.bias).to(dtype)
    output = layer(inputs)
    assert output.dtype == dtype
    torch.testing.assert_close(
        output, expected, rtol=1e-5 if dtype in (torch.float32, torch.float64) else 1e-2, atol=1e-6
    )


@pytest.mark.parametrize(
    "initial_dtype, later_dtype", [(torch.float32, torch.float64), (torch.float64, torch.float32)]
)
def test_stable_embedding_does_not_follow_later_default_dtype(initial_dtype, later_dtype):
    original_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(initial_dtype)
        layer = bnb.nn.StableEmbedding(32, 16)
        inputs = torch.tensor([1, 3, 7])
        expected = layer(inputs)
        torch.set_default_dtype(later_dtype)
        output = layer(inputs)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert output.dtype == initial_dtype
    finally:
        torch.set_default_dtype(original_default)


def test_converted_stable_embedding_checkpoint_continues_training():
    model, _ = _models("cpu", torch.bfloat16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
    inputs = torch.tensor([[1, 2, 3]])
    model(inputs).float().square().mean().backward()
    optimizer.step()
    stream = BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    restored = copy.deepcopy(model)
    restored.load_state_dict(checkpoint["model"], strict=True)
    resumed_optimizer = torch.optim.SGD(restored.parameters(), lr=0.025, momentum=0.9)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    for current, current_optimizer in [(model, optimizer), (restored, resumed_optimizer)]:
        current_optimizer.zero_grad()
        current(inputs).float().square().mean().backward()
        current_optimizer.step()
    for parameter, restored_parameter in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)
