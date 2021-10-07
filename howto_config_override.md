# How to override config hyperparameters for particular weights/parameters

If you want to optimize some unstable parameters with 32-bit Adam and others with 8-bit Adam, you can use the `GlobalOptimManager`. With this, we can also configure specific hyperparameters for particular layers, such as embedding layers. To do that, we need two things: (1) register the parameter while they are still on the CPU, (2) override the config with the new desired hyperparameters (anytime, anywhere). See our [guide](howto_config_override.md) for more details

```python
import torch
import bitsandbytes as bnb

mng = bnb.optim.GlobalOptimManager.get_instance()

model = MyModel()
mng.register_parameters(model.parameters()) # 1. register parameters while still on CPU

model = model.cuda()
# use 8-bit optimizer states for all parameters
adam = bnb.optim.Adam(model.parameters(), lr=0.001, optim_bits=8) 

# 2a. override: the parameter model.fc1.weight now uses 32-bit Adam
mng.override_config(model.fc1.weight, 'optim_bits', 32) 

# 2b. override: the two special layers use
# sparse optimization + different learning rate + different Adam betas
mng.override_config([model.special.weight, model.also_special.weight],
                    key_value_dict ={'is_sparse': True, 'lr': 1e-5, 'betas'=(0.9, 0.98)}) 
``` 
Possible options for the config override are: `betas, eps, weight_decay, lr, optim_bits, min_8bit_size, percentile_clipping, block_wise, max_unorm`
