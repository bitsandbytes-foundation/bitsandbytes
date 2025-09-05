from functools import partial
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from .. import functional as F


class Bnb4bitParametrization(nn.Module):
    """
    A parametrization module that handles dequantization of a 4-bit quantized parameter.

    The parameter data is expected to be already quantized when this parametrization is applied.
    This module will dequantize the parameter data to its original floating-point representation
    when the forward method is called (i.e. when the parameter is accessed).

    Args:
        quant_state (`F.QuantState`):
            The quantization state containing the necessary information for dequantization.
    """

    def __init__(self, quant_state: F.QuantState):
        super().__init__()
        self.quant_state = quant_state

    @torch.no_grad()
    def forward(self, quantized_param: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to dequantize the parameter.

        Args:
            quantized_param (`torch.Tensor`): The quantized parameter tensor (from .original)

        Returns:
            `torch.Tensor`: The dequantized parameter tensor in the original shape and dtype.
        """
        return F.dequantize_4bit(quantized_param, self.quant_state)


def replace_parameter_4bit_prequantized(
    module: nn.Module, param_name: str, qs_dict: dict[str, Any], device: torch.device
):
    if not hasattr(module, param_name):
        raise AttributeError(f"Module does not have parameter '{param_name}'")

    original_param = getattr(module, param_name)

    if not isinstance(original_param, nn.Parameter):
        raise TypeError(f"Parameter '{param_name}' is not an instance of nn.Parameter")

    quant_state = F.QuantState.from_dict(qs_dict, device=device)

    # Apply a parametrization to the module to handle dequantization.
    P.register_parametrization(module, param_name, Bnb4bitParametrization(quant_state), unsafe=True)

    # Next, register hooks.
    _register_parametrization_hooks(module, param_name)


def replace_parameter_4bit(
    module: nn.Module,
    param_name: str,
    compress_statistics: bool = False,
    quant_type: Literal["nf4", "fp4"] = "nf4",
    blocksize: Optional[int] = None,
):
    """
    Replace a module parameter with a 4-bit quantized version using parametrization.

    This function quantizes an existing parameter in a PyTorch module to 4-bit precision
    and sets up parametrization to handle automatic dequantization during forward passes.
    The original parameter is replaced with quantized data, and a parametrization layer
    is registered to manage the quantization state and dequantization process.

    Additional, it registers a state dict post-hook to ensure that the quantization state
    is saved correctly when the model's state dict is saved.

    It is useful for MoE models or other scenarios where you want to quantize parameters
    outside of nn.Linear layers without changing the model's architecture.

    <Tip warning={true}>This feature is experimental and may change in future releases.</Tip>

    Args:
        module (`nn.Module`):
            The PyTorch module containing the parameter to be quantized.
        param_name (`str`):
            The name of the parameter within the module to quantize.
        compress_statistics (`bool`, *optional*, defaults to `False`):
            Whether to compress quantization statistics to reduce memory usage.
        quant_type (`Literal["nf4", "fp4"]`, *optional*, defaults to `"nf4"`):
            The quantization format to use.
        blocksize (`int`, *optional*, defaults to `None`):
            The block size for quantization. If None, uses the default block size.

    Raises:
        AttributeError: If the module does not have the specified parameter.
        TypeError: If the specified attribute is not an instance of nn.Parameter.
    """

    if not hasattr(module, param_name):
        raise AttributeError(f"Module does not have parameter '{param_name}'")

    original_param = getattr(module, param_name)

    if not isinstance(original_param, nn.Parameter):
        raise TypeError(f"Parameter '{param_name}' is not an instance of nn.Parameter")

    # Quantize the original parameter.
    quantized_data, quant_state = F.quantize_4bit(
        original_param.data,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
    )

    # Replace the parameter with the quantized data.
    setattr(module, param_name, nn.Parameter(quantized_data, requires_grad=False))
    del original_param

    # Apply a parametrization to the module to handle dequantization.
    P.register_parametrization(module, param_name, Bnb4bitParametrization(quant_state), unsafe=True)

    # Next, register hooks.
    _register_parametrization_hooks(module, param_name)


def _disable_parametrization_cache(module: nn.Module, inputs: tuple[Any, ...], output: Any):
    P._cache_enabled -= 1
    if not P._cache_enabled:
        P._cache = {}


def _enable_parametrization_cache(module: nn.Module, inputs: tuple[Any, ...]):
    P._cache_enabled += 1


def _register_parametrization_hooks(module: nn.Module, param_name: str):
    # Register a state dict hook for saving. Note that this requires torch >= 2.5.0.
    if torch.__version__ >= (2, 5):
        module.register_state_dict_post_hook(
            partial(
                _parametrized_state_dict_post_hook,
                param_name=param_name,
            )
        )

    # Register hooks to enable caching for the dequantization parametrization.
    # This helps preserve time and memory when the same quantized parameter
    # is accessed multiple times in the forward computation.
    module.register_forward_pre_hook(_enable_parametrization_cache)
    module.register_forward_hook(_disable_parametrization_cache)


def _parametrized_state_dict_post_hook(
    module: nn.Module,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: Any,
    *,
    param_name: str = "weight",
    **kwargs: dict[str, Any],
) -> None:
    """
    Hook to modify the state dict to include the quantization state.
    """

    original_key = f"{prefix}parametrizations.{param_name}.original"

    if original_key in state_dict:
        # Create a clean entry.
        # The `parametrizations.{param_name}.original` key will have the quantized data,
        # but we would like it to keep it in the state_dict as `{param_name}`.
        clean_key = f"{prefix}{param_name}"
        state_dict[clean_key] = state_dict.pop(original_key)

        assert P.is_parametrized(module, param_name)

        # Find the parametrization, which should have the quantization state.
        parametrization: Bnb4bitParametrization = next(
            filter(lambda x: isinstance(x, Bnb4bitParametrization), module.parametrizations[param_name]), None
        )

        assert parametrization is not None, "Parametrization not found for the parameter."

        quant_state = parametrization.quant_state

        # Next, we need to store the quantization state.
        if quant_state is not None:
            for k, v in quant_state.as_dict(packed=True).items():
                state_dict[f"{prefix}{param_name}.{k}"] = v
