from contextlib import contextmanager

import bitsandbytes as bnb
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

    #[0 ...
    #[1 ...
    #[2 ...
    invperm = torch.argsort(quantized_params_dict["perm"]).cpu()
    # A[:, invperm]
    # A[:, [17, 3 5, 16]]
    #[2 ...
    #[0 ...
    #[1 ...
    # invperm (index to transfer wrong into right order)
    reconstructed_weight = reconstructed_weight[:, invperm]
    return reconstructed_weight


def read_state_dict_from_file(load_path, block_idx, layer_name):
    return torch.load(load_path + "/" + str(block_idx) + "/" + layer_name)


def dequantize_layer(load_path, layer, layer_idx):
    sub_layers = find_sublayers(layer)

    for name_of_sub_layer in sub_layers:
        quantized_params_dict = read_state_dict_from_file(
            load_path, layer_idx, name_of_sub_layer)
        # init
        sublayer_new = bnb.nn.spqr.Linear3BitSpQR(sub_layers.in_features, out_features)
        # Solution A:
        # 1. dequantize weight (to get right permutation order)
        # 2. quantize weight again
        # (2b write test to test if 2 == 1)
        #   (a) we want to have the quantized data, but we cannot because its permuted
        #   (b) so we need to unpermute, because its complicated, we dequantize first (which unpermutes the data)
        #   (c) now to use this, we need to requantized the dequantized (and unpermuted) data
        #   (d) to test if (c) is correct, we need to ensure its the same as permuted (d)
        #       this means, we take (d), permute it with the permutation order of (a) and then quantize it again
        #       now if (d)[perm] is equal to (a) we know our implementation is correct
        #   quantized_data == quantize(dequantize(data)[perm])
        # 3. now pack the weight+scales+zeros from (2) into class Linear3bitSpqr

        # SpQR
        # weights [17, 83].T -> [cols, row]
        # H -> torch.argsort(H) -> idx   argsort([5, 7, 3]) -> index(sort([5, 7, 3])) -> index([3, 5, 7] -> [2, 0, 1]
        # weights_perm = A[:, idx]
        # quant(A_perm) -> quant_weight, quant_zeros, quant_scales ...
        # checkpoint = torch.save(quant_state)

        # weight_row_indicies_partial_3_first_values = [5, 7, 3]
        # perm_3_first_values = [2, 0, 1]
        # weight_row_inicidies_partial_3_first_values = weight[:, perm] = [3, 5, 7]
        # inverted_perm_3_first_values = torch.argsort(perm) = [1, 2, 0]
        # weight_row_inicides_partcial_3_first_value_of_permuted[inv_perm] -> [5, 7, 3]

        # bitsandbytes
        # state_dict = torch.load(quant_path)
        # weights = dequantize(weights_perm, weights_scales, weight_zeros ...) 
        # quantize(weight) -> quant_weight, quant_zeros, quant_scales
        # quantize: q = torch.clamp(torch.round(x / scale.clamp_min(eps) + zero), 0, maxq) (in bitsandbytes x is not permuted)
        sublayer_new.weight = bnb.nn.spqr.Params3bit(pack((quantized_params_dict["quant_weights"][:, 'perm']).half))
        sublayer_new.scales = bnb.nn.spqr.Params3bit(pack((quantized_params_dict["quant_layer_zeros"]).half))
        sublayer_new.scales = bnb.nn.spqr.Params3bit(pack((quantized_params_dict["quant_layer_scales"]).half))

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
