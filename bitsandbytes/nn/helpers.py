from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import torch
from torch.nn import Module


def detach_tensors_or_pass_value(value: Any) -> Any:
    """
    Detach a tensor or all tensors in a list, leaving other types as is.

    Args:
        value (Any): A value that could be a tensor, a list of tensors, or another type.

    Returns:
        Any: The detached tensor(s) or the original value if it's not a tensor.
    """
    if isinstance(value, torch.Tensor):
        return value.detach()
    elif isinstance(value, list):
        return [v.detach() if isinstance(v, torch.Tensor) else v for v in value]
    else:
        return value


@contextmanager
def suspend_nn_inits() -> Iterator[None]:
    """
    Context manager to suspend all PyTorch initialization functions.

    This context manager designed to temporarily disable the initialization routines of PyTorch neural network layers. When you enter the context (i.e., the block of code under the `with suspend_nn_inits():` statement), the standard initialization functions like `kaiming_uniform_`, `uniform_`, and `normal_` are replaced with a `skip` function that does nothing.
    
    This is useful for cases where you don't want the layers to be automatically initialized with random values, such as when you're about to load pre-trained or quantized weights into a model.
    
    Once you exit the context, the original initialization functions are restored, ensuring that the changes are temporary and confined to the specific block of code.
    """

    def skip(*args, **kwargs):
        return None

    # backing up
    backed_up_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_

    # replacing, using temporary monkey patching
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip

    try:
        yield

    finally:
        # restoring
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = backed_up_inits


def read_state_dict_from_file(
        load_path: str, layer_idx: int, sublayer_name: str) -> dict:
    """
    Read the quantized parameters for a given layer from a file.

    Args:
        load_path (str): The path to the saved quantized model.
        layer_idx (int): The index of the layer in the model.
        sublayer_name (str): The name of the sublayer.

    Returns:
        dict: A dictionary containing the quantized parameters.
    """
    return torch.load(f"{load_path}/{layer_idx}/{sublayer_name}")


def explore_state_dict(state_dict: dict):
    """
    Explore the state dictionary of a PyTorch layer.

    Parameters:
        state_dict (dict): The state dictionary.

    Prints:
        Descriptive statistics for each sublayer.
    """
    from collections import Counter

    for key, value in state_dict.items():
        if isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                content = Counter(
                    (tensor.layout, tensor.shape, tensor.dtype) for tensor in value)

            else:
                content = Counter(type(item) for item in value)

        elif isinstance(value, torch.Tensor):
            content = (value.layout, value.shape, value.dtype)

        else:
            content = (type(value), value)

        print(f'{key}: {content}')


def display_quantization(start: float, stop: float, increment: float, maxq: int):
    """
    Displays the quantization process for a range of values.

    Args:
        start (float): The starting value of the range to be quantized.
        stop (float): The ending value of the range to be quantized.
        increment (float): The step size for generating values within the range.
        maxq (int): The maximum quantization level.

    Returns:
        list: A list of tuples, each containing the original value, the scaled value,
              and the quantized value.

    Notes:
        This function prints the scale and zero-point used for quantization, the
        original values to be quantized, and their quantized versions.
    """
    to_quantize = torch.arange(start, stop, increment).tolist()
    scale = (max(to_quantize) - min(to_quantize)) / maxq
    zero = -min(to_quantize) / scale

    # TODO: use actual quantization function from bitsandbytes.nn.spqr.quantization
    scaled = (torch.arange(start, stop, increment) / scale + zero).tolist()
    quantized = (torch.arange(start, stop, increment) / scale + zero).round().tolist()

    print(f'{scale = }\n{zero = }\n{to_quantize = }\n{quantized = }')
    return list(zip(to_quantize, scaled, quantized))


def list_layers(model, indent=0):
    """Recursively list all layer types in a PyTorch model."""
    for name, module in model.named_children():
        # Check if the module has any parameters that require gradients
        requires_grad = any(param.requires_grad for param in module.parameters())

        print(
            "    " * indent, name, ":",
            type(module).__name__, f"- {requires_grad = }")

        # Recursively list the layers for child modules
        if isinstance(module, torch.nn.Module):
            list_layers(module, indent + 1)


def list_layers_and_parameters(
        model: Module,
        indent: int = 0,
        non_leaf_params: Optional[Dict[str, str]] = None) -> None:
    """
    Recursively list all layer types and their parameters in a PyTorch model.
    
    Parameters:
        model (Module): The PyTorch model to explore.
        indent (int): Indentation level for printing.
        non_leaf_params (Dict[str, str]): Dictionary to collect non-leaf node parameters.
    """
    if non_leaf_params is None:
        non_leaf_params = {}

    for name, module in model.named_children():
        print("    " * indent, name, ":", type(module).__name__)

        for param_name, param in module.named_parameters():
            param_is_local_to_module = '.' not in param_name

            if param_is_local_to_module:
                non_leaf_params[
                    f"{name}.{param_name}"] = f"{param.requires_grad=}, {param.dtype=}, {param.shape=}"
                print(
                    "    " * (indent + 1), param_name,
                    f"- {param.requires_grad=}, {param.dtype=}")

        # Recursively list the layers and parameters for child modules
        if isinstance(module, torch.nn.Module):
            list_layers_and_parameters(module, indent + 1, non_leaf_params)

    if indent == 0:
        print("\n\n\nNon-leaf node parameters:")
        for param_name, properties in non_leaf_params.items():
            print("    ", param_name, "-", properties)


def list_cuda_devices():
    """List all available CUDA devices, with id and device names."""
    if torch.cuda.is_available():
        # Get the total number of available CUDA devices
        num_cuda_devices = torch.cuda.device_count()

        # Iterate through each CUDA device and print its name
        for i in range(num_cuda_devices):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available.")


import functools


def debug(func):
    """Print the function signature and return value."""

    def format_arg(arg):
        if isinstance(arg, bool):
            return str(arg)
        elif isinstance(arg, int):
            return f"{arg:,}"
        else:
            return repr(arg)

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [format_arg(a) for a in args]
        kwargs_repr = [f"{k}={format_arg(v)}" for k, v in kwargs.items()]
        signature = ",\n".join(args_repr + kwargs_repr)
        print(f"{func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"returned {format_arg(value)}\n\n")
        return value

    return wrapper_debug
