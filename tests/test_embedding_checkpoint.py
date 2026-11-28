import copy

import pytest
import torch

import bitsandbytes as bnb
from tests.helpers import get_available_devices


def make_embedding(kind, width=64, dtype=torch.float32, storage=torch.uint8):
    if kind == "int8":
        return bnb.nn.Embedding8bit(16, width, dtype=dtype)
    return bnb.nn.Embedding4bit(16, width, dtype=dtype, quant_type=kind, quant_storage=storage)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("kind", ["int8", "fp4", "nf4"])
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("prequantized", [False, True])
def test_embedding_checkpoint_roundtrip(device, kind, width, assign, prequantized, tmp_path):
    torch.manual_seed(29)
    source = torch.nn.Sequential(
        make_embedding(kind, width).to(device),
        torch.nn.Linear(width, 4, device=device),
    )
    inputs = torch.tensor([[1, 3, 1], [7, 9, 11]], device=device)
    optimizer = torch.optim.SGD(source[1].parameters(), lr=0.03)
    optimizer.zero_grad()
    source(inputs).square().mean().backward()
    optimizer.step()
    path = tmp_path / "embedding.pt"
    torch.save({"model": source.state_dict(), "optimizer": optimizer.state_dict()}, path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    original_checkpoint = copy.deepcopy(checkpoint["model"])

    restored = (
        torch.nn.Sequential(
            make_embedding(kind, width),
            torch.nn.Linear(width, 4),
        ).to(device)
        if prequantized
        else torch.nn.Sequential(
            make_embedding(kind, width),
            torch.nn.Linear(width, 4),
        )
    )
    if not prequantized:
        # Keep the embedding weight unquantized until load_state_dict while
        # placing the destination tensors on the requested device.
        restored[0].weight.data = restored[0].weight.data.to(device)
        restored[1].to(device)
    original_parameter = restored[0].weight
    result = restored.load_state_dict(checkpoint["model"], strict=True, assign=assign)
    assert not result.missing_keys and not result.unexpected_keys
    if not assign:
        assert restored[0].weight is original_parameter
        assert restored[0].weight.data_ptr() != checkpoint["model"]["0.weight"].data_ptr()
    expected_type = bnb.nn.Int8Params if kind == "int8" else bnb.nn.Params4bit
    assert isinstance(restored[0].weight, expected_type)
    assert not restored[0].weight.requires_grad
    if kind == "int8":
        assert restored[0].weight.CB.data_ptr() == restored[0].weight.data_ptr()
        assert restored[0].weight.SCB.data_ptr() != checkpoint["model"]["0.SCB"].data_ptr()
    else:
        assert restored[0].weight.module is restored[0]
        assert restored[0].quant_state is restored[0].weight.quant_state
        assert restored[0].weight.quant_state.absmax.data_ptr() != checkpoint["model"]["0.weight.absmax"].data_ptr()
    for key, value in original_checkpoint.items():
        torch.testing.assert_close(checkpoint["model"][key], value, rtol=0, atol=0)

    restored_optimizer = torch.optim.SGD(restored[1].parameters(), lr=0.03)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    for _ in range(2):
        expected = source(inputs)
        actual = restored(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        optimizer.zero_grad()
        restored_optimizer.zero_grad()
        expected.square().mean().backward()
        actual.square().mean().backward()
        for actual_parameter, expected_parameter in zip(restored[1].parameters(), source[1].parameters()):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)
        optimizer.step()
        restored_optimizer.step()
        for actual_parameter, expected_parameter in zip(restored[1].parameters(), source[1].parameters()):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["int8", "fp4", "nf4"])
@pytest.mark.parametrize("keep_vars", [False, True])
def test_embedding_checkpoint_unquantized_state(kind, keep_vars):
    source = make_embedding(kind)
    destination = make_embedding(kind)
    state = source.state_dict(keep_vars=keep_vars)
    assert list(state) == ["weight"]
    destination.load_state_dict(state, strict=True)
    torch.testing.assert_close(destination.weight, source.weight, rtol=0, atol=0)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("kind", ["fp4", "nf4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_embedding_checkpoint_floating_storage(device, kind, dtype):
    source = make_embedding(kind, dtype=dtype, storage=dtype).to(device)
    destination = make_embedding(kind, dtype=dtype, storage=dtype)
    destination.weight.data = destination.weight.data.to(device)
    state = source.state_dict()
    destination.load_state_dict(state, strict=True)
    inputs = torch.tensor([1, 3, 7], device=device)
    torch.testing.assert_close(destination(inputs), source(inputs), rtol=0, atol=0)
    assert destination.weight.quant_storage == dtype
    assert destination.weight.quant_state.dtype == dtype


@pytest.mark.parametrize("kind", ["int8", "fp4", "nf4"])
def test_embedding_checkpoint_rejects_wrong_logical_shape(kind):
    source = make_embedding(kind, width=64).to("cpu")
    destination = make_embedding(kind, width=32)
    original_weight = destination.weight.detach().clone()
    with pytest.raises(RuntimeError, match="Invalid quantized embedding shape"):
        destination.load_state_dict(source.state_dict())
    torch.testing.assert_close(destination.weight, original_weight, rtol=0, atol=0)


def test_embedding_checkpoint_requires_int8_scales():
    source = make_embedding("int8").to("cpu")
    state = source.state_dict()
    del state["SCB"]
    destination = make_embedding("int8")
    original_weight = destination.weight.detach().clone()
    with pytest.raises(RuntimeError, match="Missing quantization scales"):
        destination.load_state_dict(state)
    torch.testing.assert_close(destination.weight, original_weight, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["int8", "fp4", "nf4"])
@pytest.mark.parametrize("width", [32, 64])
def test_embedding_checkpoint_assign_materializes_meta(kind, width):
    source = make_embedding(kind, width=width).to("cpu")
    with torch.device("meta"):
        destination = make_embedding(kind, width=width)
    assert destination.weight.device.type == "meta"
    destination.load_state_dict(source.state_dict(), assign=True, strict=True)
    assert destination.weight.device.type == "cpu"
    inputs = torch.tensor([[1, 7], [4, 9]])
    torch.testing.assert_close(destination(inputs), source(inputs), rtol=0, atol=0)
