import torch

from bitsandbytes.functional import pack_3bits
from bitsandbytes.nn.helpers import explore_state_dict
from bitsandbytes.nn.quantization import apply_packing

layer_path = '/mnt/D/data/falcon7b/5/self_attention.dense'
example_layer = torch.load(layer_path)

to_pack_layers = ['quant_weights', 'quant_layer_scale', 'quant_layer_zeros']
packed_state_dict = apply_packing(example_layer, to_pack_layers, debug=True)

explore_state_dict(example_layer)
print()
explore_state_dict(packed_state_dict)
