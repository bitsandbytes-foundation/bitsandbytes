# bitsandbytes

Bitsandbytes is a lightweight wrapper around CUDA custom functions, in particular 8-bit optimizers and quantization functions.

[Paper](https://arxiv.org/abs/2110.02861) -- [Video](https://www.youtube.com/watch?v=IxrlHAJtqKE) -- [Docs](https://bitsandbytes.readthedocs.io/en/latest/)

## TL;DR
**Installation**:
1. Note down version: ``conda list | grep cudatoolkit``
2. Replace 111 with the version that you see: ``pip install bitsandbytes-cuda111``

**Usage**:
1. Comment out optimizer: ``#torch.optim.Adam(....)``
2. Add 8-bit optimizer of your choice ``bnb.optim.Adam8bit(....)`` (arguments stay the same)
3. Replace embedding layer if necessary: ``torch.nn.Embedding(..) -> bnb.nn.Embedding(..)``


## Features
- 8-bit Optimizers: Adam, AdamW, RMSProp, LARS, LAMB (saves 75% memory)
- Stable Embedding Layer: Improved stability through better initialization, and normalization
- 8-bit quantization: Quantile, Linear, and Dynamic quantization
- Fast quantile estimation: Up to 100x faster than other algorithms

## Requirements & Installation

Requirements: anaconda, cudatoolkit, pytorch
Hardware requirements: NVIDIA Maxwell GPU or newer (>=GTX 9XX)
Supported CUDA versions: 9.2 - 11.3

The requirements can best be fulfilled by installing pytorch via anaconda. You can install PyTorch by following the ["Get Started"](https://pytorch.org/get-started/locally/) instructions on the official website.

bitsandbytes is compatible with all major PyTorch releases and cudatoolkit versions, but for now, you need to select the right version manually. To do this run:

```conda list | grep cudatoolkit```

and take note of the Cuda version that you have installed. Then you can install bitsandbytes via:
```bash
# choices: {cuda92, cuda 100, cuda101, cuda102, cuda110, cuda111, cuda113}
# replace XXX with the respective number
pip install bitsandbytes-cudaXXX
```

To check if your installation was successful, you can execute the following command, which runs a single bnb Adam update.
```
wget https://gist.githubusercontent.com/TimDettmers/1f5188c6ee6ed69d211b7fe4e381e713/raw/4d17c3d09ccdb57e9ab7eca0171f2ace6e4d2858/check_bnb_install.py && python check_bnb_install.py
```

## Using bitsandbytes

### Using the 8-bit Optimizers

With bitsandbytes 8-bit optimizers can be used by changing a single line of code in your codebase. For NLP models we recommend also to use the StableEmbedding layers (see below) which improves results and helps with stable 8-bit optimization.  To get started with 8-bit optimizers, it is sufficient to replace your old optimizer with the 8-bit optimizer in the following way:
```python
import bitsandbytes as bnb

# adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # comment out old optimizer
adam = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # add bnb optimizer
adam = bnb.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995), optim_bits=8) # equivalent


torch.nn.Embedding(...) ->  bnb.nn.StableEmbedding(...) # recommended for NLP models
```

Note that by default all parameter tensors with less than 4096 elements are kept at 32-bit even if you initialize those parameters with 8-bit optimizers. This is done since such small tensors do not save much memory and often contain highly variable parameters (biases) or parameters that require high precision (batch norm, layer norm). You can change this behavior like so:
```
# parameter tensors with less than 16384 values are optimized in 32-bit
# it is recommended to use multiplies of 4096
adam = bnb.optim.Adam8bit(model.parameters(), min_8bit_size=16384) 
```

### Change Bits and other Hyperparameters for Individual Parameters

If you want to optimize some unstable parameters with 32-bit Adam and others with 8-bit Adam, you can use the `GlobalOptimManager`. With this, we can also configure specific hyperparameters for particular layers, such as embedding layers. To do that, we need two things: (1) register the parameter while they are still on the CPU, (2) override the config with the new desired hyperparameters (anytime, anywhere). See our [guide](howto_config_override.md) for more details

### Fairseq Users

To use the Stable Embedding Layer, override the respective `build_embedding(...)` function of your model. Make sure to also use the `--no-scale-embedding` flag to disable scaling of the word embedding layer (nor replaced with layer norm). You can use the optimizers by replacing the optimizer in the respective file (`adam.py` etc.).

## Release and Feature History

For upcoming features and changes and full history see [Patch Notes](CHANGELOG.md).

## Errors

1. RuntimeError: CUDA error: no kernel image is available for execution on the device. [Solution](errors_and_solutions.md#No-kernel-image-available)
2. __fatbinwrap_.. [Solution](errors_and_solutions.md#fatbinwrap_)

## Compile from source

To compile from source, please follow the [compile_from_source.md](compile_from_source.md) instructions.

## License

The majority of bitsandbytes is licensed under MIT, however portions of the project are available under separate license terms: Pytorch is licensed under the BSD license.

We thank Fabio Cannizzo for his work on [FastBinarySearch](https://github.com/fabiocannizzo/FastBinarySearch) which we use for CPU quantization.

## Citation
If you found this library and 8-bit optimizers or quantization routines useful, please consider citing out work.
```
@misc{dettmers2021optim8bit,
      title={8-bit Optimizers via Block-wise Quantization},
      author={Tim Dettmers and Mike Lewis and Sam Shleifer and Luke Zettlemoyer},
      year={2021},
      eprint={2110.02861},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
