from bitsandbytes.bitsandbytes.nn.load_quantized import get_model
from bitsandbytes.nn.quantization import find_sublayers_to_dequantize

# model_path = '/mnt/D/data/falcon7b/'  # permuted
model_path = '/mnt/D/data/test_falcon2/'  # unpermuted
model = get_model(model_path, model_path)

# quantized = torch.load('./tests/test_layer_quantized.pth')
layers = model.transformer.h
sublayers = [sublayer for sublayer in layers[0].named_modules()]

test_layer = find_sublayers_to_dequantize(layers[0])['self_attention.dense']
