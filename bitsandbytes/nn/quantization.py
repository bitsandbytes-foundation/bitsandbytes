import torch
import torch.nn as nn
from typing import Any, Dict, List, Union

import bitsandbytes as bnb
from .helpers import read_state_dict_from_file


def dequantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        eps: float = 1e-9) -> torch.Tensor:
    """
    Dequantizes an input tensor based on the provided scale and zero-point values.
    
    Parameters:
        x (torch.Tensor): The quantized input tensor to be dequantized.
        scale (torch.Tensor): The scale tensor, used for the dequantization process.
        zero (torch.Tensor): The zero-point tensor, used for the dequantization process.
        eps (float, optional): A small constant added to avoid division by zero. 
                                Currently not used in the function. Default is 1e-9.
    
    Returns:
        torch.Tensor: The dequantized tensor.
        
    Example:
        >>> x = torch.tensor([2, 3, 4])
        >>> scale = torch.tensor(0.5)
        >>> zero = torch.tensor(1)
        >>> dequantize(x, scale, zero)
        tensor([0.5, 1.0, 1.5])
    """
    # TODO: eps: float=1e-9 is not even used --> Tim: why is it there?
    return scale * (x - zero)


def quantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        maxq: int,
        eps: float = 1e-9) -> Union[torch.Tensor, list]:
    """
    Quantizes an input tensor based on the provided scale, zero-point, and maximum quantized value.
    
    Parameters:
        x (torch.Tensor): The input tensor to be quantized.
        scale (torch.Tensor): The scale tensor, used for the quantization process.
        zero (torch.Tensor): The zero-point tensor, used for the quantization process.
        maxq (int): The maximum value after quantization.
        eps (float, optional): A small constant added to avoid division by zero. Default is 1e-9.
    
    Returns:
        torch.Tensor: The quantized tensor.
        
    Example:
        >>> x = torch.tensor([-3, -2, -1, 0, 1, 2, 3]).tolist()
        >>> scale = (max(to_quantize) - min(to_quantize)) / maxq  # 0.857...
        >>> zero = -min(to_quantize) / scale  # 3.5
        >>> maxq = 7  # 3-bit quantization: 2^3 = 8 --> [0..7]
        >>> quantize(x, scale, zero, maxq)
        tensor([0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0])
    """
    return torch.clamp(torch.round(x / scale.clamp_min(eps) + zero), 0, maxq)


def layer_weight_dequantization(quantized_params_dict: Dict[str, Any]) -> torch.Tensor:
    _, in_dim = quantized_params_dict["weight_shape"]  # torch.Size([4544, 4544])
    blocksize = quantized_params_dict["blocksize"]  # 128
    keep_last_columns = quantized_params_dict["keep_last_columns"]  # 0
    reconstructed_weight = torch.zeros(quantized_params_dict["weight_shape"])
    block_start_iter = range(0, in_dim - keep_last_columns, blocksize)
    current_ind = 0

    for block_start in block_start_iter:
        block_end = min(block_start + blocksize, in_dim)

        for column_index in range(block_start, block_end):

            if column_index % quantized_params_dict["groupsize"] == 0:

                if quantized_params_dict["quant_layer_scale_qq_scale"]:

                    dequantize_zeros = dequantize(
                        quantized_params_dict["quant_layer_zeros"][current_ind],
                        quantized_params_dict["quant_layer_zero_qq_scale"][current_ind],
                        quantized_params_dict["quant_layer_zero_qq_zero"][current_ind],
                    )
                    dequantize_scale = dequantize(
                        quantized_params_dict["quant_layer_scale"][current_ind],
                        quantized_params_dict["quant_layer_scale_qq_scale"]
                        [current_ind],
                        quantized_params_dict["quant_layer_scale_qq_zero"][current_ind],
                    )

                else:
                    dequantize_zeros = quantized_params_dict["quant_layer_zeros"][
                        current_ind]
                    dequantize_scale = quantized_params_dict["quant_layer_scale"][
                        current_ind]
                current_ind += 1

            reconstructed_weight[:, column_index] = dequantize(
                quantized_params_dict["quant_weights"][:, column_index].unsqueeze(1),
                dequantize_scale.reshape(-1, 1),
                dequantize_zeros.reshape(-1, 1),
            ).reshape_as(reconstructed_weight[:, column_index])

    reconstructed_weight = (
        reconstructed_weight *
        (quantized_params_dict["outliers_matrix"].to_dense().cpu() == 0) +
        quantized_params_dict["outliers_matrix"].to_dense().cpu())

    # weight tensor has a permutation order, which was needed due to a helpful computational
    # optimization during quantization, but now has to be inverted to get the correct order
    index_to_invert_permutation = torch.argsort(quantized_params_dict["perm"]).cpu()

    reconstructed_weight = reconstructed_weight[:, index_to_invert_permutation]
    return reconstructed_weight


def find_sublayers_to_dequantize(
    layer, layer_types_to_filter_by=(nn.Conv2d, nn.Linear)):
    sublayers = {}

    for name, sublayer in layer.named_modules():

        if isinstance(sublayer, layer_types_to_filter_by):
            sublayers[name] = sublayer

    return sublayers


def dequantize_layer(load_path, layer, layer_idx):
    sublayers = find_sublayers_to_dequantize(layer)

    for name_of_sublayer, sublayer in sublayers.items():
        quantized_params_dict = read_state_dict_from_file(
            load_path, layer_idx, name_of_sublayer)
        sublayer.weight = nn.Parameter(
            layer_weight_dequantization(quantized_params_dict).to(
                sublayer.weight.data.dtype))

    return layer


def apply_packing(
    layer_state_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    keys: Union[str, List[str]],
    debug: bool = False,
) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Applies packing, mutating a given layer state dictionary's given `keys` to tensors of equivalent 64-bit integers that are a container to a maximum of 23 packed 3-bit integers.

    Args:
        layer_state_dict (Dict): The state dictionary of the layer.
        keys (Union[str, List[str]]): The keys of the layer state dictionary.

    Returns:
        Dict: The packed layer state dictionary.
    """

    def prepare_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Converts a tensor to half-precision, as that's what `the pack_3bits()` function expects, and moves it to the CUDA device."""
        return tensor.half().cuda()

    def pack_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Packs a tensor into 3-bits and prepares it beforehand."""
        if debug:
            print(f"Packing tensor for `{key}` of shape {tensor.shape}...")
        return prepare_tensor(bnb.functional.pack_3bits(tensor))

    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        value = layer_state_dict[key]

        if torch.is_tensor(value):
            layer_state_dict[key] = pack_tensor(value)

        elif isinstance(value, list) and all(torch.is_tensor(x) for x in value):
            layer_state_dict[key] = [pack_tensor(tensor) for tensor in value]

        else:
            raise ValueError(f"Unsupported value type for key {key}: {type(value)}")

    return layer_state_dict


def load_sublayer(
    sublayer_name: str,
    sublayer: nn.Linear,
    load_path: str,
    layer_idx: int,
) -> bnb.nn.Linear3BitSpQR:
    """
    Loads the weights of a SpQR-quantized sublayer.

    Args:
        sublayer_name (str): The name of the sublayer.
        sublayer (nn.Linear): The PyTorch sublayer to be re-quantized.

    Returns:
        bnb.nn.Linear3BitSpQR: The re-quantized sublayer.
    """
    new_sublayer = bnb.nn.Linear3BitSpQR(sublayer.in_features, sublayer.out_features)

    quantized_params_dict = read_state_dict_from_file(
        load_path, layer_idx, sublayer_name)
    new_sublayer.weight = bnb.nn.SpQRParameter(**quantized_params_dict)

    return new_sublayer


def load_layer(load_path: str, layer, layer_idx: int):
    """
    Loads a given SpQR-quantized layer.

    Args:
        load_path (str): The path to the saved quantized model.
        layer: The PyTorch layer to be dequantized.
        layer_idx (int): The index of the layer in the model.
    """
    sublayers = find_sublayers_to_dequantize(layer)

    for sublayer_name, sublayer in sublayers.items():
        load_sublayer(sublayer_name, sublayer)

    return layer
