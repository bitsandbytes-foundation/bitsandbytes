import io

import pytest
import torch

from bitsandbytes import functional as F


def _values(dtype):
    # These binary fractions avoid the existing LUT discretization boundaries.
    return torch.tensor([-1.0, -0.75, -0.5, -0.25, 0.125, 0.5, 0.75, 1.0], dtype=dtype).repeat(33)[:257]


def _change_interior(code, half):
    start, end = (1, 127) if half == "negative" else (127, 255)
    position = torch.arange(1, end - start, dtype=code.dtype) / (end - start)
    code[start + 1 : end] = code[start] + (code[end] - code[start]) * position.square()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("blocksize", [64, 256])
@pytest.mark.parametrize("half", ["negative", "positive"])
def test_cpu_quantization_refreshes_changed_codebook_interior(dtype, blocksize, half):
    code = torch.linspace(-1.0, 1.0, 256)
    values = _values(dtype)
    original = code.clone()
    pointer = code.data_ptr()
    warmed, warm_state = F.quantize_blockwise(values, code=code, blocksize=blocksize)
    _change_interior(code, half)
    assert code.data_ptr() == pointer
    torch.testing.assert_close(code[[0, 1, 127, 255]], original[[0, 1, 127, 255]], rtol=0, atol=0)
    assert not torch.equal(code, original)
    assert torch.all(code[1:] > code[:-1])

    actual, state = F.quantize_blockwise(values, code=code, blocksize=blocksize)
    fresh_code = code.clone()
    expected, expected_state = F.quantize_blockwise(values, code=fresh_code, blocksize=blocksize)

    assert not torch.equal(warmed, expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state.absmax, expected_state.absmax, rtol=0, atol=0)
    torch.testing.assert_close(state.code, code, rtol=0, atol=0)
    torch.testing.assert_close(warm_state.code, original, rtol=0, atol=0)
    # An independent nearest-code oracle checks the changed values themselves.
    scales = state.absmax.repeat_interleave(blocksize)[: values.numel()]
    nearest = (values.float().div(scales)[:, None] - code).abs().argmin(dim=1).to(torch.uint8)
    torch.testing.assert_close(actual, nearest, rtol=0, atol=0)
    decoded = F.dequantize_blockwise(actual, state)
    torch.testing.assert_close(decoded, (code[nearest.long()] * scales).to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_codebook_cache_preserves_unchanged_and_sampled_entry_updates(dtype):
    code = torch.linspace(-1.0, 1.0, 256)
    values = _values(dtype)
    first, state = F.quantize_blockwise(values, code=code, blocksize=64)
    repeated, repeated_state = F.quantize_blockwise(values, code=code, blocksize=64)
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)
    torch.testing.assert_close(repeated_state.absmax, state.absmax, rtol=0, atol=0)
    code[127] = 0.0
    actual, actual_state = F.quantize_blockwise(values, code=code, blocksize=64)
    expected, expected_state = F.quantize_blockwise(values, code=code.clone(), blocksize=64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state.code, expected_state.code, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_codebook_cache_keeps_alternating_and_evicted_maps_distinct(dtype):
    values = _values(dtype)
    codebooks = [torch.linspace(-1.0, 1.0, 256) for _ in range(7)]
    for index, code in enumerate(codebooks):
        positions = torch.linspace(0.0, 1.0, 256)
        code.copy_(2.0 * positions.pow(1.0 + index / 10.0) - 1.0)
    expected = [F.quantize_blockwise(values, code=code, blocksize=64)[0] for code in codebooks]
    for index in (0, 1, 0, 1, 6, 5, 4, 3, 2, 1, 0):
        actual, _ = F.quantize_blockwise(values, code=codebooks[index], blocksize=64)
        torch.testing.assert_close(actual, expected[index], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_updated_cpu_codebook_persists_consistent_linear_weights(dtype):
    code = torch.linspace(-1.0, 1.0, 256)
    weights = _values(dtype)[:256].reshape(16, 16)
    F.quantize_blockwise(weights, code=code, blocksize=64)
    _change_interior(code, "negative")
    actual, state = F.quantize_blockwise(weights, code=code, blocksize=64)
    expected, expected_state = F.quantize_blockwise(weights, code=code.clone(), blocksize=64)
    buffer = io.BytesIO()
    torch.save({"weights": actual, "code": state.code, "absmax": state.absmax}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    decoded = F.dequantize_blockwise(
        restored["weights"], code=restored["code"], absmax=restored["absmax"], blocksize=64
    )
    reference = F.dequantize_blockwise(expected, code=expected_state.code, absmax=expected_state.absmax, blocksize=64)
    inputs = torch.linspace(-1.0, 1.0, 48).reshape(3, 16)
    torch.testing.assert_close(
        torch.nn.functional.linear(inputs, decoded),
        torch.nn.functional.linear(inputs, reference),
        rtol=0,
        atol=0,
    )
