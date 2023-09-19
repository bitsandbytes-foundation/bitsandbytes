import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from tqdm import trange

from .helpers import suspend_nn_inits
from .quantization import dequantize_layer

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama' and 'falcon' supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")


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


def load_quantized_model(model, load_path):
    layers = get_layers(model)

    for layer_idx in trange(len(layers)):
        layers[layer_idx] = dequantize_layer(load_path, layers[layer_idx], layer_idx)

    model.load_state_dict(
        torch.load(f"{load_path}/not_quantized_weights.pt"), strict=False)

    return model


def get_model(model_path, load_quantized=None, dtype="auto"):
    # TODO: `model_path` and `load_quantized` are both paths, but they are the same path. This is confusing. Fix this.
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
