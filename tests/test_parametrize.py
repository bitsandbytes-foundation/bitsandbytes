import pytest
import torch
import torch.nn as nn

from bitsandbytes import functional as F
from bitsandbytes.cextension import ROCM_WARP_SIZE_64
from bitsandbytes.nn.parametrize import (
    Bnb4bitParametrization,
    replace_parameter_4bit,
    replace_parameter_4bit_prequantized,
)
from tests.helpers import (
    TRUE_FALSE,
    describe_dtype,
    get_available_devices,
    id_formatter,
    is_supported_on_hpu,
)


class ParametrizeTestModule(nn.Module):
    """Test module with different parameter shapes for testing parametrization."""

    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        # 2D parameter (typical weight matrix)
        self.weight_2d = nn.Parameter(torch.randn(1024, 1024, device=device, dtype=dtype))
        # 3D parameter (MoE expert weights - the main use case for this feature)
        self.expert_weights = nn.Parameter(torch.randn(8, 512, 256, device=device, dtype=dtype))
        # 1D parameter (bias-like)
        self.bias_1d = nn.Parameter(torch.randn(1024, device=device, dtype=dtype))
        # Non-parameter attribute (should not be quantizable)
        self.not_param = torch.randn(32, device=device, dtype=dtype)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
@pytest.mark.parametrize(
    "blocksize",
    [64, 128, 256] if not ROCM_WARP_SIZE_64 else [128, 256],
)
def test_replace_parameter_4bit(device, dtype, quant_type, compress_statistics, blocksize):
    """Test basic parameter replacement with 4-bit quantization on different dtypes."""
    if device == "hpu" and not is_supported_on_hpu(quant_type, dtype):
        pytest.skip("This configuration is not supported on HPU.")

    # Create module directly on target device to avoid unnecessary transfers
    module = ParametrizeTestModule(device=device, dtype=dtype)
    original_param = module.weight_2d.clone()

    # Apply 4-bit quantization parametrization to the weight parameter
    replace_parameter_4bit(
        module, "weight_2d", compress_statistics=compress_statistics, quant_type=quant_type, blocksize=blocksize
    )

    # Verify that parametrization was applied correctly
    assert hasattr(module, "parametrizations"), "Module should have parametrizations attribute"
    assert "weight_2d" in module.parametrizations, "weight_2d should be parametrized"

    # Test that accessing the parameter returns dequantized version with correct properties
    reconstructed = module.weight_2d
    assert reconstructed.shape == original_param.shape, "Shape should be preserved"
    assert reconstructed.dtype == dtype, "dtype should match original"
    assert reconstructed.device.type == device, "Device should match target"

    # Verify quantization quality using same approach as functional tests
    err = (original_param - reconstructed.detach()).abs().float()
    relerr = (err / (original_param.abs().float() + 1e-8)).mean()
    err_mean = err.mean()

    # Expected error bounds from test_functional.py
    expected_errors = {
        "nf4": {
            64: {"abs": 0.072792, "rel": 0.203299},
            128: {"abs": 0.076835, "rel": 0.215252},
            256: {"abs": 0.080326, "rel": 0.226044},
        },
        "fp4": {
            64: {"abs": 0.096545, "rel": 0.260130},
            128: {"abs": 0.102947, "rel": 0.275734},
            256: {"abs": 0.108685, "rel": 0.289842},
        },
    }

    assert err_mean < expected_errors[quant_type][blocksize]["abs"] + 1e-3, f"Mean abs error {err_mean:.6f} too high"
    assert relerr < expected_errors[quant_type][blocksize]["rel"] + 1e-3, f"Mean rel error {relerr:.6f} too high"


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
def test_moe_parameter_shape(device, dtype):
    """Test parametrization with MoE-style parameter shape"""
    if device == "hpu" and not is_supported_on_hpu("nf4", dtype):
        pytest.skip("This configuration is not supported on HPU.")

    param_shape = (8, 64, 64)

    # Create module with custom parameter shape directly on target device
    class MoEModule(nn.Module):
        def __init__(self, device, dtype):
            super().__init__()
            self.param = nn.Parameter(torch.randn(*param_shape, dtype=dtype, device=device))

    module = MoEModule(device=device, dtype=dtype)
    original_param = module.param.clone()

    # Apply quantization parametrization
    replace_parameter_4bit(module, "param", quant_type="nf4")

    # Verify reconstruction maintains all properties
    reconstructed = module.param
    assert reconstructed.shape == param_shape, f"Shape should be preserved: {reconstructed.shape} vs {param_shape}"
    assert reconstructed.dtype == dtype, "dtype should match original"
    assert reconstructed.device.type == device, "Device should match target"

    # Verify quantization quality using error calculation approach from functional tests
    err = (original_param - reconstructed.detach()).abs().float()
    relerr = (err / (original_param.abs().float() + 1e-8)).mean()
    err_mean = err.mean()

    # Use slightly looser bounds for higher dimensional tensors
    abs_bound = 0.085  # NF4 baseline + margin
    rel_bound = 0.25  # NF4 baseline + margin

    assert err_mean < abs_bound, f"Mean abs error {err_mean:.6f} too high for shape {param_shape}"
    assert relerr < rel_bound, f"Mean rel error {relerr:.6f} too high for shape {param_shape}"


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_prequantized_replacement(device, dtype, quant_type):
    """Test applying parametrization to already quantized parameters."""
    if device == "hpu" and not is_supported_on_hpu(quant_type, dtype):
        pytest.skip("Configuration not supported on HPU.")

    module = ParametrizeTestModule(device=device, dtype=dtype)
    original_param = module.weight_2d.clone()

    # Manually quantize the parameter data first (simulates loading pre-quantized weights)
    quantized_data, quant_state = F.quantize_4bit(original_param.data, quant_type=quant_type)

    # Replace parameter with quantized data (what would happen during model loading)
    module.weight_2d = nn.Parameter(quantized_data, requires_grad=False)

    # Apply parametrization to handle dequantization on access
    replace_parameter_4bit_prequantized(
        module, "weight_2d", quant_state.as_dict(packed=True), device=torch.device(device)
    )

    # Test that parameter access properly dequantizes
    reconstructed = module.weight_2d
    assert reconstructed.shape == original_param.shape, "Shape should be preserved"
    assert reconstructed.dtype == dtype, "dtype should match original"
    assert reconstructed.device.type == device, "Device should match target"


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
@pytest.mark.skipif(torch.__version__ < (2, 5), reason="state dict hook requires torch >= 2.5.0")
def test_state_dict_functionality(device, dtype, quant_type, compress_statistics):
    """Test that state dict saving works with quantized parameters."""
    if device == "hpu" and not is_supported_on_hpu(quant_type, dtype):
        pytest.skip("Configuration not supported on HPU.")

    module = ParametrizeTestModule(device=device, dtype=dtype)

    # Apply parametrization to expert weights (main MoE use case)
    replace_parameter_4bit(module, "expert_weights", quant_type=quant_type, compress_statistics=compress_statistics)

    # Save state dict - should include quantization state, not parametrization internals
    state_dict = module.state_dict()

    # Verify state dict structure: quantized param + quantization metadata
    assert "expert_weights" in state_dict, "Quantized parameter should be in state dict"
    assert "expert_weights.absmax" in state_dict, "Quantization absmax should be saved"
    assert "expert_weights.quant_map" in state_dict, "Quantization map should be saved"
    assert f"expert_weights.quant_state.bitsandbytes__{quant_type}" in state_dict, "Quant state should be saved"

    # Verify parametrization internals are NOT saved (clean state dict)
    assert "parametrizations.expert_weights.original" not in state_dict, (
        "Internal parametrization keys should not be saved"
    )

    # Test that the parameter can be accessed after state dict creation
    reconstructed = module.expert_weights
    assert reconstructed.shape == (8, 512, 256), "Shape should be preserved"
    assert reconstructed.dtype == dtype, "dtype should match"


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
def test_moe_realistic_forward(device, dtype):
    """Test realistic MoE forward computation with quantized expert weights."""
    if device == "hpu" and not is_supported_on_hpu("nf4", dtype):
        pytest.skip("Configuration not supported on HPU.")

    class SimpleMoE(nn.Module):
        def __init__(self, device, dtype):
            super().__init__()
            # Expert weights: [num_experts, input_dim, output_dim]
            self.expert_weights = nn.Parameter(torch.randn(4, 32, 64, dtype=dtype, device=device))

        def forward(self, x, expert_idx=0):
            # Select and use specific expert weight matrix
            expert_weight = self.expert_weights[expert_idx]  # Shape: [input_dim, output_dim]
            return torch.matmul(x, expert_weight)

    module = SimpleMoE(device=device, dtype=dtype)
    x = torch.randn(8, 32, dtype=dtype, device=device)

    # Get reference output before quantization
    with torch.no_grad():
        reference_output = module(x, expert_idx=1)

    # Apply 4-bit quantization to expert weights
    replace_parameter_4bit(module, "expert_weights", quant_type="nf4")

    # Get output after quantization - should be very close to original
    with torch.no_grad():
        quantized_output = module(x, expert_idx=1)

    # Verify outputs match within quantization tolerance
    assert quantized_output.shape == reference_output.shape, "Output shape should be preserved"

    # Calculate error like functional tests (matrix ops may amplify quantization errors)
    err = (reference_output - quantized_output).abs().float()
    relerr = (err / (reference_output.abs().float() + 1e-8)).mean()
    err_mean = err.mean()

    # Allow for error amplification through matrix multiplication
    assert err_mean < 0.5, f"Forward pass mean abs error {err_mean:.6f} too high"
    assert relerr < 2.0, f"Forward pass mean rel error {relerr:.6f} too high"


def test_error_conditions():
    """Test that proper errors are raised for invalid inputs."""
    module = ParametrizeTestModule()

    # Test AttributeError for non-existent parameter
    with pytest.raises(AttributeError, match="Module does not have parameter 'nonexistent'"):
        replace_parameter_4bit(module, "nonexistent")

    # Test TypeError for non-Parameter attribute
    with pytest.raises(TypeError, match="Parameter 'not_param' is not an instance of nn\\.Parameter"):
        replace_parameter_4bit(module, "not_param")

    # Test same errors for prequantized version
    with pytest.raises(AttributeError, match="Module does not have parameter 'nonexistent'"):
        replace_parameter_4bit_prequantized(module, "nonexistent", {}, torch.device("cpu"))

    with pytest.raises(TypeError, match="Parameter 'not_param' is not an instance of nn\\.Parameter"):
        replace_parameter_4bit_prequantized(module, "not_param", {}, torch.device("cpu"))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.skipif(torch.__version__ < (2, 5), reason="state dict hook requires torch >= 2.5.0")
def test_quant_state_preservation(device, dtype):
    """Test that quantization state is properly preserved and accessible."""
    if device == "hpu" and not is_supported_on_hpu("nf4", dtype):
        pytest.skip("Configuration not supported on HPU.")

    module = ParametrizeTestModule(device=device, dtype=dtype)

    blocksize = 128 if ROCM_WARP_SIZE_64 else 64

    # Apply parametrization with specific settings
    replace_parameter_4bit(module, "weight_2d", quant_type="nf4", compress_statistics=True, blocksize=blocksize)

    # Verify that quantization state is accessible through parametrization
    parametrization = module.parametrizations.weight_2d[0]
    assert isinstance(parametrization, Bnb4bitParametrization), "Should be Bnb4bitParametrization instance"

    # Check quantization state properties
    quant_state = parametrization.quant_state
    assert isinstance(quant_state, F.QuantState), "Should have QuantState"
    assert quant_state.quant_type == "nf4", "Quant type should be preserved"
    assert quant_state.blocksize == blocksize, "Block size should be preserved"

    # Verify that state dict includes all necessary quantization metadata
    state_dict = module.state_dict()
    quant_state_dict = quant_state.as_dict(packed=True)

    for key in quant_state_dict.keys():
        full_key = f"weight_2d.{key}"
        assert full_key in state_dict, f"Quantization metadata '{full_key}' should be in state dict"


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.skipif(torch.__version__ < (2, 5), reason="state dict hook requires torch >= 2.5.0")
def test_multiple_parameters(device, dtype):
    """Test applying parametrization to multiple parameters in the same module."""
    if device == "hpu" and not is_supported_on_hpu("nf4", dtype):
        pytest.skip("Configuration not supported on HPU.")

    module = ParametrizeTestModule(device=device, dtype=dtype)
    original_2d = module.weight_2d.clone()
    original_3d = module.expert_weights.clone()

    # Apply parametrization to multiple parameters, with varying configurations
    replace_parameter_4bit(module, "weight_2d", quant_type="nf4", blocksize=128)
    replace_parameter_4bit(module, "expert_weights", quant_type="fp4", blocksize=256)

    # Verify both parameters are parametrized and work correctly
    reconstructed_2d = module.weight_2d
    reconstructed_3d = module.expert_weights

    assert reconstructed_2d.shape == original_2d.shape, "2D parameter shape should be preserved"
    assert reconstructed_3d.shape == original_3d.shape, "3D parameter shape should be preserved"

    # Check that state dict includes quantization info for both parameters
    state_dict = module.state_dict()
    assert "weight_2d" in state_dict, "2D parameter should be in state dict"
    assert "expert_weights" in state_dict, "3D parameter should be in state dict"
    assert "weight_2d.absmax" in state_dict, "2D parameter quantization metadata should be saved"
    assert "expert_weights.absmax" in state_dict, "3D parameter quantization metadata should be saved"


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize(
    "blocksize",
    [64, 128, 256] if not ROCM_WARP_SIZE_64 else [128, 256],
)
def test_different_blocksizes(device, dtype, blocksize):
    """Test parametrization with different block sizes to verify flexibility."""
    if device == "hpu" and not is_supported_on_hpu("nf4", dtype):
        pytest.skip("Configuration not supported on HPU.")

    module = ParametrizeTestModule(device=device, dtype=dtype)
    original_param = module.expert_weights.clone()

    # Apply parametrization with specified block size
    replace_parameter_4bit(module, "expert_weights", quant_type="nf4", blocksize=blocksize)

    # Verify reconstruction works with different block sizes
    reconstructed = module.expert_weights
    assert reconstructed.shape == original_param.shape, "Shape should be preserved"
    assert reconstructed.device.type == device, "Device should match"

    # Verify quantization quality using error calculation approach from functional tests
    err = (original_param - reconstructed.detach()).abs().float()
    relerr = (err / (original_param.abs().float() + 1e-8)).mean()
    err_mean = err.mean()

    # Expected error bounds from functional tests (using NF4 bounds since that's what we're testing)
    expected_abs = {64: 0.072792, 128: 0.076835, 256: 0.080326}
    expected_rel = {64: 0.203299, 128: 0.215252, 256: 0.226044}

    assert err_mean < expected_abs[blocksize] + 0.01, (
        f"Mean abs error {err_mean:.6f} too high for blocksize {blocksize}"
    )
    assert relerr < expected_rel[blocksize] + 0.02, f"Mean rel error {relerr:.6f} too high for blocksize {blocksize}"


def test_parametrization_forward_method():
    """Test the Bnb4bitParametrization forward method directly."""
    device = "cpu"

    # Create test tensor and manually quantize it
    original_tensor = torch.randn(64, 64, dtype=torch.float32, device=device)
    quantized_data, quant_state = F.quantize_4bit(original_tensor, quant_type="nf4")

    # Create parametrization instance
    parametrization = Bnb4bitParametrization(quant_state)

    # Test forward pass (dequantization)
    dequantized = parametrization.forward(quantized_data)

    # Verify dequantization produces correct output
    assert dequantized.shape == original_tensor.shape, "Shape should be preserved during dequantization"
    assert dequantized.dtype == torch.float32, "dtype should be preserved"
    assert dequantized.device == original_tensor.device, "Device should be preserved"

    # Check that dequantization approximates original using mean error calculation
    err = (original_tensor - dequantized.detach()).abs().float()
    relerr = (err / (original_tensor.abs().float() + 1e-8)).mean()
    err_mean = err.mean()

    # Use NF4 bounds from functional tests with small margin
    assert err_mean < 0.08, f"Mean abs error {err_mean:.6f} too high"
    assert relerr < 0.25, f"Mean rel error {relerr:.6f} too high"


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
def test_gradient_behavior(device, dtype):
    """Test that quantized parameters have proper gradient behavior."""
    if device == "hpu" and not is_supported_on_hpu("nf4", dtype):
        pytest.skip("Configuration not supported on HPU.")

    module = ParametrizeTestModule(device=device, dtype=dtype)

    # Ensure original parameter requires gradients
    module.weight_2d.requires_grad_(True)
    assert module.weight_2d.requires_grad, "Original parameter should require gradients"

    # Apply quantization parametrization
    replace_parameter_4bit(module, "weight_2d", quant_type="nf4")

    # Verify that quantized parameters don't require gradients (expected behavior)
    # The underlying quantized parameter should have requires_grad=False
    # The dequantized output should also not require gradients
    reconstructed = module.weight_2d
    assert not reconstructed.requires_grad, "Dequantized parameter should not require gradients"
