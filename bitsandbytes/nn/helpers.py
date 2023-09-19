import torch

from contextlib import contextmanager
from typing import Any, Iterator


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