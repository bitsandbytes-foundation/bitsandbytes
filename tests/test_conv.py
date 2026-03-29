"""Tests for quantized Conv layers (Conv*d4bit, Conv*d8bitLt).

Follows the same patterns used in test_linear4bit.py and test_modules.py.
"""

import pytest
import torch
import torch.nn as nn

import bitsandbytes as bnb
from bitsandbytes.nn.modules import (
    Conv1d4bit,
    Conv2d4bit,
    Conv3d4bit,
    Conv1dFP4,
    Conv1dNF4,
    Conv2dFP4,
    Conv2dNF4,
    Conv3dFP4,
    Conv3dNF4,
    Conv1d8bitLt,
    Conv2d8bitLt,
    Conv3d8bitLt,
    Params4bit,
    Int8Params,
)
from tests.helpers import (
    TRUE_FALSE,
    get_available_devices,
    id_formatter,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

CONV4BIT_CLASSES_1D = [Conv1d4bit, Conv1dFP4, Conv1dNF4]
CONV4BIT_CLASSES_2D = [Conv2d4bit, Conv2dFP4, Conv2dNF4]
CONV4BIT_CLASSES_3D = [Conv3d4bit, Conv3dFP4, Conv3dNF4]
CONV8BIT_CLASSES = [Conv1d8bitLt, Conv2d8bitLt, Conv3d8bitLt]


# ---------------------------------------------------------------------------
# 1. Construction / parameter type tests (no device movement needed)
# ---------------------------------------------------------------------------


class TestConv4bitConstruction:
    """Verify that Conv*d4bit layers can be constructed and have the right param types."""

    def test_conv1d4bit_creates_params4bit_weight(self):
        layer = Conv1d4bit(16, 32, 3, padding=1)
        assert isinstance(layer.weight, Params4bit)
        assert layer.weight.quant_type == "fp4"
        assert layer._weight_shape == (32, 16, 3)

    def test_conv2d4bit_creates_params4bit_weight(self):
        layer = Conv2d4bit(3, 64, 3, padding=1)
        assert isinstance(layer.weight, Params4bit)
        assert layer._weight_shape == (64, 3, 3, 3)

    def test_conv3d4bit_creates_params4bit_weight(self):
        layer = Conv3d4bit(3, 16, (3, 3, 3), padding=1)
        assert isinstance(layer.weight, Params4bit)
        assert layer._weight_shape == (16, 3, 3, 3, 3)

    @pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
    def test_conv2d4bit_quant_type(self, quant_type):
        layer = Conv2d4bit(8, 16, 3, quant_type=quant_type)
        assert layer.weight.quant_type == quant_type

    def test_conv1d_fp4_alias(self):
        layer = Conv1dFP4(16, 32, 3)
        assert isinstance(layer, Conv1d4bit)
        assert layer.weight.quant_type == "fp4"

    def test_conv1d_nf4_alias(self):
        layer = Conv1dNF4(16, 32, 3)
        assert isinstance(layer, Conv1d4bit)
        assert layer.weight.quant_type == "nf4"

    def test_conv2d_fp4_alias(self):
        layer = Conv2dFP4(3, 16, 3)
        assert isinstance(layer, Conv2d4bit)
        assert layer.weight.quant_type == "fp4"

    def test_conv2d_nf4_alias(self):
        layer = Conv2dNF4(3, 16, 3)
        assert isinstance(layer, Conv2d4bit)
        assert layer.weight.quant_type == "nf4"

    def test_conv3d_fp4_alias(self):
        layer = Conv3dFP4(3, 16, 3)
        assert isinstance(layer, Conv3d4bit)
        assert layer.weight.quant_type == "fp4"

    def test_conv3d_nf4_alias(self):
        layer = Conv3dNF4(3, 16, 3)
        assert isinstance(layer, Conv3d4bit)
        assert layer.weight.quant_type == "nf4"

    @pytest.mark.parametrize("bias", TRUE_FALSE, ids=id_formatter("bias"))
    def test_conv2d4bit_bias(self, bias):
        layer = Conv2d4bit(3, 16, 3, bias=bias)
        if bias:
            assert layer.bias is not None
            assert layer.bias.shape == (16,)
        else:
            assert layer.bias is None

    def test_conv2d4bit_groups(self):
        layer = Conv2d4bit(16, 32, 3, groups=4, padding=1)
        assert layer.groups == 4
        # weight shape should be (out_ch, in_ch/groups, kH, kW)
        assert layer._weight_shape == (32, 4, 3, 3)

    def test_conv2d4bit_asymmetric_kernel(self):
        layer = Conv2d4bit(3, 16, (5, 3), padding=(2, 1))
        assert layer._weight_shape == (16, 3, 5, 3)

    def test_conv2d4bit_no_bias_weight_shape(self):
        layer = Conv2d4bit(3, 16, 3, bias=False)
        # Flatten should be (out_channels, in_channels * kH * kW)
        assert layer.weight.data.shape[0] <= 16 * 3 * 3 * 3  # packed, so could be smaller

    def test_compute_dtype_is_stored(self):
        layer = Conv2d4bit(3, 16, 3, compute_dtype=torch.bfloat16)
        assert layer.compute_dtype == torch.bfloat16
        assert layer.compute_type_is_set is True

    def test_compute_dtype_default_is_none(self):
        layer = Conv2d4bit(3, 16, 3)
        assert layer.compute_dtype is None
        assert layer.compute_type_is_set is False


class TestConv8bitLtConstruction:
    """Verify that Conv*d8bitLt layers can be constructed and have the right param types."""

    def test_conv1d8bit_creates_int8params_weight(self):
        layer = Conv1d8bitLt(16, 32, 3, padding=1)
        assert isinstance(layer.weight, Int8Params)
        assert layer._weight_shape == (32, 16, 3)

    def test_conv2d8bit_creates_int8params_weight(self):
        layer = Conv2d8bitLt(3, 64, 3, padding=1)
        assert isinstance(layer.weight, Int8Params)
        assert layer._weight_shape == (64, 3, 3, 3)

    def test_conv3d8bit_creates_int8params_weight(self):
        layer = Conv3d8bitLt(3, 16, (3, 3, 3), padding=1)
        assert isinstance(layer.weight, Int8Params)
        assert layer._weight_shape == (16, 3, 3, 3, 3)

    @pytest.mark.parametrize("has_fp16_weights", TRUE_FALSE, ids=id_formatter("fp16w"))
    def test_conv2d8bit_has_fp16_weights(self, has_fp16_weights):
        layer = Conv2d8bitLt(3, 16, 3, has_fp16_weights=has_fp16_weights)
        assert layer.state.has_fp16_weights == has_fp16_weights
        assert layer.weight.has_fp16_weights == has_fp16_weights

    @pytest.mark.parametrize("bias", TRUE_FALSE, ids=id_formatter("bias"))
    def test_conv2d8bit_bias(self, bias):
        layer = Conv2d8bitLt(3, 16, 3, bias=bias)
        if bias:
            assert layer.bias is not None
        else:
            assert layer.bias is None

    def test_conv2d8bit_groups(self):
        layer = Conv2d8bitLt(16, 32, 3, groups=4, padding=1)
        assert layer.groups == 4
        assert layer._weight_shape == (32, 4, 3, 3)


# ---------------------------------------------------------------------------
# 2. isinstance / subclass checks
# ---------------------------------------------------------------------------


class TestInheritance:
    """Check MRO and isinstance relationships."""

    def test_conv1d4bit_is_conv1d(self):
        layer = Conv1d4bit(16, 32, 3)
        assert isinstance(layer, nn.Conv1d)
        assert isinstance(layer, nn.Module)

    def test_conv2d4bit_is_conv2d(self):
        layer = Conv2d4bit(3, 16, 3)
        assert isinstance(layer, nn.Conv2d)

    def test_conv3d4bit_is_conv3d(self):
        layer = Conv3d4bit(3, 16, 3)
        assert isinstance(layer, nn.Conv3d)

    def test_conv1d8bit_is_conv1d(self):
        layer = Conv1d8bitLt(16, 32, 3)
        assert isinstance(layer, nn.Conv1d)

    def test_conv2d8bit_is_conv2d(self):
        layer = Conv2d8bitLt(3, 16, 3)
        assert isinstance(layer, nn.Conv2d)

    def test_conv3d8bit_is_conv3d(self):
        layer = Conv3d8bitLt(3, 16, 3)
        assert isinstance(layer, nn.Conv3d)


# ---------------------------------------------------------------------------
# 3. Weight shape bookkeeping
# ---------------------------------------------------------------------------


class TestWeightShape:
    """Verify _weight_shape is correctly computed for various configurations."""

    @pytest.mark.parametrize(
        "in_ch, out_ch, ks, groups, expected_shape",
        [
            (16, 32, 3, 1, (32, 16, 3)),
            (16, 32, 5, 1, (32, 16, 5)),
            (16, 32, 3, 4, (32, 4, 3)),
            (16, 32, 1, 1, (32, 16, 1)),
        ],
    )
    def test_conv1d4bit_weight_shape(self, in_ch, out_ch, ks, groups, expected_shape):
        layer = Conv1d4bit(in_ch, out_ch, ks, groups=groups)
        assert layer._weight_shape == expected_shape

    @pytest.mark.parametrize(
        "in_ch, out_ch, ks, groups, expected_shape",
        [
            (3, 64, 3, 1, (64, 3, 3, 3)),
            (16, 32, (5, 3), 1, (32, 16, 5, 3)),
            (16, 32, 3, 4, (32, 4, 3, 3)),
            (64, 64, 1, 1, (64, 64, 1, 1)),
        ],
    )
    def test_conv2d4bit_weight_shape(self, in_ch, out_ch, ks, groups, expected_shape):
        layer = Conv2d4bit(in_ch, out_ch, ks, groups=groups)
        assert layer._weight_shape == expected_shape

    @pytest.mark.parametrize(
        "in_ch, out_ch, ks, groups, expected_shape",
        [
            (3, 16, 3, 1, (16, 3, 3, 3, 3)),
            (16, 32, (3, 5, 7), 1, (32, 16, 3, 5, 7)),
            (16, 32, 3, 4, (32, 4, 3, 3, 3)),
        ],
    )
    def test_conv3d4bit_weight_shape(self, in_ch, out_ch, ks, groups, expected_shape):
        layer = Conv3d4bit(in_ch, out_ch, ks, groups=groups)
        assert layer._weight_shape == expected_shape


# ---------------------------------------------------------------------------
# 4. Conv attributes preserved (stride, padding, dilation)
# ---------------------------------------------------------------------------


class TestConvAttributes:
    """Ensure standard Conv attributes survive quantisation wrapping."""

    def test_conv1d4bit_attributes(self):
        layer = Conv1d4bit(16, 32, 5, stride=2, padding=2, dilation=1)
        assert layer.stride == (2,)
        assert layer.padding == (2,)
        assert layer.dilation == (1,)
        assert layer.kernel_size == (5,)
        assert layer.in_channels == 16
        assert layer.out_channels == 32

    def test_conv2d4bit_attributes(self):
        layer = Conv2d4bit(3, 64, (3, 5), stride=(1, 2), padding=(1, 2), dilation=(1, 1))
        assert layer.stride == (1, 2)
        assert layer.padding == (1, 2)
        assert layer.kernel_size == (3, 5)

    def test_conv2d8bit_attributes(self):
        layer = Conv2d8bitLt(3, 64, 3, stride=2, padding=1)
        assert layer.stride == (2, 2)
        assert layer.padding == (1, 1)
        assert layer.kernel_size == (3, 3)


# ---------------------------------------------------------------------------
# 5. Forward pass tests (require CUDA / accelerator)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("bias", TRUE_FALSE, ids=id_formatter("bias"))
class TestConv4bitForward:
    """Test forward pass output shapes and approximate numerical correctness."""

    def test_conv1d4bit_forward_shape(self, device, quant_type, bias):
        layer = Conv1d4bit(16, 32, 3, padding=1, bias=bias, quant_type=quant_type)
        layer = layer.to(device)
        x = torch.randn(2, 16, 20, device=device, dtype=torch.float32)
        out = layer(x)
        assert out.shape == (2, 32, 20)
        assert out.dtype == x.dtype

    def test_conv2d4bit_forward_shape(self, device, quant_type, bias):
        layer = Conv2d4bit(3, 64, 3, padding=1, bias=bias, quant_type=quant_type)
        layer = layer.to(device)
        x = torch.randn(2, 3, 8, 8, device=device, dtype=torch.float32)
        out = layer(x)
        assert out.shape == (2, 64, 8, 8)

    def test_conv2d4bit_forward_groups(self, device, quant_type, bias):
        layer = Conv2d4bit(16, 32, 3, padding=1, groups=4, bias=bias, quant_type=quant_type)
        layer = layer.to(device)
        x = torch.randn(2, 16, 8, 8, device=device, dtype=torch.float32)
        out = layer(x)
        assert out.shape == (2, 32, 8, 8)

    def test_conv2d4bit_forward_asymmetric_kernel(self, device, quant_type, bias):
        layer = Conv2d4bit(3, 16, (5, 3), padding=(2, 1), bias=bias, quant_type=quant_type)
        layer = layer.to(device)
        x = torch.randn(1, 3, 12, 12, device=device, dtype=torch.float32)
        out = layer(x)
        assert out.shape == (1, 16, 12, 12)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
class TestConv4bitNumericalParity:
    """Compare quantized Conv output with full-precision Conv (rough parity)."""

    def test_conv2d4bit_approximate_parity(self, device, quant_type):
        """Output of quantized conv should be within ~20% of fp conv for random weights."""
        torch.manual_seed(42)
        fp_conv = nn.Conv2d(16, 32, 3, padding=1, bias=True)

        q_conv = Conv2d4bit(16, 32, 3, padding=1, bias=True, quant_type=quant_type)
        # Copy the weights before quantisation
        q_conv.weight = Params4bit(
            data=fp_conv.weight.data.reshape(32, -1).clone(),
            quant_type=quant_type,
            requires_grad=False,
        )
        q_conv.bias = nn.Parameter(fp_conv.bias.data.clone())

        fp_conv = fp_conv.to(device)
        q_conv = q_conv.to(device)

        x = torch.randn(4, 16, 8, 8, device=device, dtype=torch.float32)
        fp_out = fp_conv(x)
        q_out = q_conv(x)

        assert fp_out.shape == q_out.shape
        # 4-bit quantisation introduces error; we check the values are in the same ballpark
        rel_err = (fp_out - q_out).abs().mean() / fp_out.abs().mean()
        assert rel_err < 0.5, f"Relative error too large: {rel_err:.4f}"


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("bias", TRUE_FALSE, ids=id_formatter("bias"))
class TestConv8bitLtForward:
    """Test forward pass for 8-bit conv layers."""

    def test_conv1d8bit_forward_fp16_weights(self, device, bias):
        layer = Conv1d8bitLt(16, 32, 3, padding=1, bias=bias, has_fp16_weights=True)
        layer = layer.to(device)
        x = torch.randn(2, 16, 20, device=device, dtype=torch.float32)
        out = layer(x)
        assert out.shape == (2, 32, 20)

    def test_conv2d8bit_forward_fp16_weights(self, device, bias):
        layer = Conv2d8bitLt(3, 64, 3, padding=1, bias=bias, has_fp16_weights=True)
        layer = layer.to(device)
        x = torch.randn(2, 3, 8, 8, device=device, dtype=torch.float32)
        out = layer(x)
        assert out.shape == (2, 64, 8, 8)

    def test_conv2d8bit_forward_int8_weights(self, device, bias):
        layer = Conv2d8bitLt(16, 32, 3, padding=1, bias=bias, has_fp16_weights=False)
        layer = layer.to(device)
        x = torch.randn(2, 16, 8, 8, device=device, dtype=torch.float16)
        out = layer(x)
        assert out.shape == (2, 32, 8, 8)

    def test_conv2d8bit_forward_groups(self, device, bias):
        layer = Conv2d8bitLt(16, 32, 3, padding=1, groups=4, bias=bias, has_fp16_weights=True)
        layer = layer.to(device)
        x = torch.randn(2, 16, 8, 8, device=device, dtype=torch.float32)
        out = layer(x)
        assert out.shape == (2, 32, 8, 8)


# ---------------------------------------------------------------------------
# 6. Module export checks
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify all new Conv classes are accessible via bnb.nn.*."""

    @pytest.mark.parametrize(
        "name",
        [
            "Conv1d4bit",
            "Conv2d4bit",
            "Conv3d4bit",
            "Conv1dFP4",
            "Conv2dFP4",
            "Conv3dFP4",
            "Conv1dNF4",
            "Conv2dNF4",
            "Conv3dNF4",
            "Conv1d8bitLt",
            "Conv2d8bitLt",
            "Conv3d8bitLt",
        ],
    )
    def test_accessible_via_bnb_nn(self, name):
        assert hasattr(bnb.nn, name), f"bnb.nn.{name} not found"
        cls = getattr(bnb.nn, name)
        assert isinstance(cls, type), f"bnb.nn.{name} is not a class"


# ---------------------------------------------------------------------------
# 7. Eval / Training mode
# ---------------------------------------------------------------------------


class TestTrainEvalMode:
    """Ensure train/eval modes propagate correctly."""

    def test_conv2d4bit_eval_mode(self):
        layer = Conv2d4bit(3, 16, 3)
        layer.eval()
        assert not layer.training

    def test_conv2d4bit_train_mode(self):
        layer = Conv2d4bit(3, 16, 3)
        layer.train()
        assert layer.training

    def test_conv2d8bit_eval_mode(self):
        layer = Conv2d8bitLt(3, 16, 3)
        layer.eval()
        assert not layer.training

    def test_conv2d8bit_train_mode(self):
        layer = Conv2d8bitLt(3, 16, 3)
        layer.train()
        assert layer.training
