import io

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F


def _metadata_view(values, stride=2):
    # Keep every native pointer read inside owned storage, including on the
    # unpatched implementation that incorrectly ignores the view's stride.
    storage = torch.full((values.numel() * stride + 3,), 0.25, dtype=values.dtype)
    view = storage[3::stride][: values.numel()]
    view.copy_(values.reshape(-1))
    return view, storage


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("blocksize", [64, 128])
@pytest.mark.parametrize("stride", [1, 2])
def test_cpu_blockwise_quantization_uses_codebook_values(dtype, blocksize, stride):
    code = torch.linspace(-1.0, 1.0, 256)
    view, storage = _metadata_view(code, stride)
    before = storage.clone()
    # Binary fractions stay away from the LUT discretization boundaries.
    values = torch.tensor([-1.0, -0.75, -0.5, -0.25, 0.125, 0.5, 0.75, 1.0], dtype=dtype)
    values = values.repeat(33)[:257]
    expected_codes, expected_state = F.quantize_blockwise(values, code=code, blocksize=blocksize)

    actual_codes, state = F.quantize_blockwise(values, code=view, blocksize=blocksize)

    torch.testing.assert_close(actual_codes, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(state.absmax, expected_state.absmax, rtol=0, atol=0)
    torch.testing.assert_close(state.code, code, rtol=0, atol=0)
    torch.testing.assert_close(storage, before, rtol=0, atol=0)
    assert view.stride() == (stride,)
    assert view.storage_offset() == 3
    # Independent nearest-code reference, using the actual rounded inputs.
    scales = expected_state.absmax.repeat_interleave(blocksize)[: values.numel()]
    distances = (values.float().div(scales)[:, None] - code).abs()
    nearest = distances.argmin(dim=1).to(torch.uint8)
    torch.testing.assert_close(actual_codes, nearest, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["code", "scales", "both", "offset"])
def test_cpu_blockwise_dequantization_reads_strided_metadata(dtype, layout):
    codes = (torch.arange(257) % 256).to(torch.uint8)
    code = F.create_dynamic_map()
    scales = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    code_view, code_storage = _metadata_view(code, 2 if layout in ("code", "both") else 1)
    scales_view, scale_storage = _metadata_view(scales, 3 if layout in ("scales", "both") else 1)
    original_code = code_storage.clone()
    original_scale = scale_storage.clone()
    state = F.QuantState(absmax=scales_view, code=code_view, blocksize=64, dtype=dtype)
    expected = (code[codes.long()] * scales.repeat_interleave(64)[: codes.numel()]).to(dtype)

    actual = F.dequantize_blockwise(codes, state)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(code_storage, original_code, rtol=0, atol=0)
    torch.testing.assert_close(scale_storage, original_scale, rtol=0, atol=0)
    assert state.code is code_view
    assert state.absmax is scales_view


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_blockwise_caller_supplied_scale_and_output_views(dtype):
    values = torch.linspace(-2.0, 3.0, 257, dtype=dtype)
    expected_codes, expected_state = F.quantize_blockwise(values, blocksize=64)
    scales, _ = _metadata_view(torch.zeros(5))
    encoded_storage = torch.full((514,), 173, dtype=torch.uint8)
    encoded = encoded_storage[::2]
    actual_codes, state = F.quantize_blockwise(values, absmax=scales, out=encoded, blocksize=64)
    assert actual_codes is encoded
    assert state.absmax is scales
    torch.testing.assert_close(actual_codes, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_state.absmax, rtol=0, atol=0)
    assert torch.all(encoded_storage[1::2] == 173)

    expected = F.dequantize_blockwise(expected_codes, expected_state)
    actual = F.dequantize_blockwise(actual_codes, state)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cpu_blockwise_explicit_metadata_survives_weight_serialization():
    weights = torch.linspace(-1.0, 2.0, 512).reshape(32, 16)
    code = F.create_dynamic_map()
    code_view, _ = _metadata_view(code)
    encoded, state = F.quantize_blockwise(weights, code=code_view, blocksize=64)
    scales_view, _ = _metadata_view(state.absmax)
    buffer = io.BytesIO()
    torch.save({"weights": encoded, "code": code_view, "scales": scales_view}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    actual = F.dequantize_blockwise(
        restored["weights"], code=restored["code"], absmax=restored["scales"], blocksize=64
    )
    reference_codes, reference_state = F.quantize_blockwise(weights, code=code, blocksize=64)
    expected = F.dequantize_blockwise(reference_codes, reference_state)
    inputs = torch.linspace(-2.0, 1.0, 48).reshape(3, 16)
    torch.testing.assert_close(
        torch.nn.functional.linear(inputs, actual),
        torch.nn.functional.linear(inputs, expected),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("metadata", ["code", "scales", "both"])
def test_cpu_adam8bit_continuation_with_strided_state(metadata):
    model = torch.nn.Linear(16, 32, bias=False)
    reference = torch.nn.Linear(16, 32, bias=False)
    reference.load_state_dict(model.state_dict())
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.01, min_8bit_size=0)
    reference_optimizer = bnb.optim.Adam8bit(reference.parameters(), lr=0.01, min_8bit_size=0)
    inputs = torch.linspace(-1.0, 1.0, 64).reshape(4, 16)
    target = torch.linspace(0.5, -0.5, 128).reshape(4, 32)
    initial = model.weight.detach().clone()
    for step in range(4):
        for network, opt in ((model, optimizer), (reference, reference_optimizer)):
            opt.zero_grad()
            torch.nn.functional.mse_loss(network(inputs), target).backward()
            opt.step()
        torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
        state = optimizer.state[model.weight]
        expected_state = reference_optimizer.state[reference.weight]
        for key in ("state1", "state2", "absmax1", "absmax2"):
            torch.testing.assert_close(state[key], expected_state[key], rtol=0, atol=0)
        if step == 0:
            assert state["state1"].dtype == torch.uint8
            keys = []
            if metadata in ("code", "both"):
                keys += ["qmap1", "qmap2"]
            if metadata in ("scales", "both"):
                keys += ["absmax1", "absmax2"]
            for key in keys:
                state[key], _ = _metadata_view(state[key])
    assert not torch.equal(model.weight, initial)
