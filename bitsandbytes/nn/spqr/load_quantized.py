from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from tqdm import trange

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama' and 'falcon' supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")


def dequantize(x, scale, zero, eps=1e-9):
    return scale * (x - zero)


@contextmanager
def suspend_nn_inits():

    def skip(*args, **kwargs):
        return None

    # saving
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_

    # replacing
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip

    try:
        yield

    finally:
        # restoring
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits


def get_layers(model):
    if model.config.model_type == "llama":
        return model.model.layers
    elif model.config.model_type.lower() in FALCON_TYPES:
        return model.transformer.h
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def find_sublayers(module, layers=(nn.Conv2d, nn.Linear)):
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


# TODO: might remove: for the debugging purposes
def explore_state_dict(layer):
    from collections import Counter

    for layer_name, layer in example_layer.items():
        if type(layer) == list:
            if isinstance(layer[0], torch.Tensor
                          ):  # Using `isinstance` is more Pythonic than `type() ==`
                content = Counter([(tensor.shape, tensor.dtype) for tensor in layer])
            else:
                content = type(layer[0])
        elif isinstance(layer, torch.Tensor):
            content = (layer.shape, layer.dtype)
        else:
            content = layer

        print(f'{layer_name}: {content}')


def layer_weight_dequantization(quantized_params_dict):
    _, in_dim = quantized_params_dict["weight_shape"]
    blocksize = quantized_params_dict["blocksize"]
    keep_last_columns = quantized_params_dict["keep_last_columns"]
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

    invperm = torch.argsort(quantized_params_dict["perm"]).cpu()
    reconstructed_weight = reconstructed_weight[:, invperm]
    return reconstructed_weight


def read_state_dict_from_file(load_path, block_idx, layer_name):
    return torch.load(load_path + "/" + str(block_idx) + "/" + layer_name)


def dequantize_layer(load_path, layer, layer_idx):
    sub_layers = find_sublayers(layer)

    for name_of_sub_layer in sub_layers:
        quantized_params_dict = read_state_dict_from_file(
            load_path, layer_idx, name_of_sub_layer)
        sub_layers[name_of_sub_layer].weight = nn.Parameter(
            layer_weight_dequantization(quantized_params_dict).to(
                sub_layers[name_of_sub_layer].weight.data.dtype))

    return layer


def load_quantized_model(model, load_path):
    layers = get_layers(model)

    for layer_idx in trange(len(layers)):
        layers[layer_idx] = dequantize_layer(load_path, layers[layer_idx], layer_idx)

    model.load_state_dict(
        torch.load(load_path + "/not_quantized_weights.pt"), strict=False)

    return model


def get_model(model_path, load_quantized=None, dtype="auto"):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=True).torch_dtype
            or "auto")  # force transformers 4.29.2 to follow the same rules as 4.30.x

    else:
        dtype = getattr(torch, dtype)

    with suspend_nn_inits():
        if load_quantized:
            print("Initializing model with random weights...")
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True)  # consider trust_remote_code=True

            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, torch_dtype=dtype).eval()

            print("Loading quantized model ...")
            model = load_quantized_model(model, load_quantized)

        else:
            print("Loading pretrained model ...")
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )

    model.seqlen = 2048

    print("Model loaded sucessfully ...")
    return model
