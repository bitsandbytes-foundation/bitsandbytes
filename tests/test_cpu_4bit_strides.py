import copy

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.cextension import ErrorHandlerMockBNBNativeLibrary, lib

pytestmark = pytest.mark.skipif(
    isinstance(lib, ErrorHandlerMockBNBNativeLibrary), reason="Native CPU library is required"
)


def _buffer_view(tensor, stride):
    buffer = torch.zeros(tensor.numel() * stride + 1, dtype=tensor.dtype)
    view = buffer[1::stride].reshape(tensor.shape)
    view.copy_(tensor)
    return buffer, view


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
@pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 64), (5, 48), (4, 33)])
@pytest.mark.parametrize("layout", ["packed", "scales", "both", "offset"])
def test_cpu_dequantize_4bit_strides(dtype, quant_type, storage_dtype, shape, layout):
    packed, state = bnb.functional.quantize_4bit(
        torch.randn(shape, dtype=dtype), quant_type=quant_type, quant_storage=storage_dtype
    )
    expected = bnb.functional.dequantize_4bit(packed, state)
    packed_buffer, packed_view = _buffer_view(packed, 2 if layout in ("packed", "both") else 1)
    scale_buffer, scale_view = _buffer_view(state.absmax, 2 if layout in ("scales", "both") else 1)
    packed_before, scales_before = packed_buffer.clone(), scale_buffer.clone()
    view_state = copy.copy(state)
    view_state.absmax = scale_view

    actual = bnb.functional.dequantize_4bit(packed_view, view_state)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(packed_buffer.view(torch.uint8), packed_before.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(scale_buffer, scales_before, rtol=0, atol=0)
    assert view_state.absmax is scale_view


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
@pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16])
def test_cpu_matmul_4bit_strided_quantization_outputs(dtype, quant_type, storage_dtype):
    weight = torch.randn(8, 64, dtype=dtype)
    packed, state = bnb.functional.quantize_4bit(weight, quant_type=quant_type, quant_storage=storage_dtype)
    _, packed_out = _buffer_view(packed, 2)
    _, scales_out = _buffer_view(state.absmax, 2)
    actual_packed, actual_state = bnb.functional.quantize_4bit(
        weight, out=packed_out, absmax=scales_out, quant_type=quant_type, quant_storage=storage_dtype
    )
    assert actual_packed is packed_out
    assert actual_state.absmax is scales_out
    expected_weight = bnb.functional.dequantize_4bit(packed, state)
    inputs = torch.randn(3, 64, dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()

    actual = bnb.matmul_4bit(inputs, actual_packed, actual_state)
    expected = torch.nn.functional.linear(reference_inputs, expected_weight)
    gradient = torch.randn_like(expected)
    actual.backward(gradient)
    expected.backward(gradient)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
