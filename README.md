# `bitsandbytes`

[![Downloads](https://static.pepy.tech/badge/bitsandbytes)](https://pepy.tech/project/bitsandbytes) [![Downloads](https://static.pepy.tech/badge/bitsandbytes/month)](https://pepy.tech/project/bitsandbytes) [![Downloads](https://static.pepy.tech/badge/bitsandbytes/week)](https://pepy.tech/project/bitsandbytes)

The `bitsandbytes` library is a lightweight Python wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and 8 & 4-bit quantization functions.

The library includes quantization primitives for 8-bit & 4-bit operations, through `bitsandbytes.nn.Linear8bitLt` and `bitsandbytes.nn.Linear4bit` and 8-bit optimizers through `bitsandbytes.optim` module.

**Installation for ROCm:**

To install develop version:
```bash
git clone --recurse https://github.com/ROCm/bitsandbytes
cd bitsandbytes
git checkout rocm_enabled
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=hip -S . (Use -DBNB_ROCM_ARCH="gfx90a;gfx942" to target specific gpu arch)
make
pip install .
```

For ROCm specific versions:

Install Dependencies:
```bash
# hipblaslt installation needed only for rocm<6.0
apt install hipblaslt
pip install --upgrade pip
pip install einops lion_pytorch accelerate
pip install git+https://github.com/ROCm/transformers.git
```
Install Bitsandbytes:
```bash
git clone --recurse https://github.com/ROCm/bitsandbytes
cd bitsandbytes
# Checkout branch as needed
# for rocm 5.7 - rocm5.7_internal_testing
# for rocm 6.2 - rocm6.2_internal_testing
git checkout <branch>
make hip
python setup.py install
```

**For more details, please head to the official documentation page:**

**[https://huggingface.co/docs/bitsandbytes/main](https://huggingface.co/docs/bitsandbytes/main)**

## License

The majority of bitsandbytes is licensed under MIT, however small portions of the project are available under separate license terms, as the parts adapted from Pytorch are licensed under the BSD license.

We thank Fabio Cannizzo for his work on [FastBinarySearch](https://github.com/fabiocannizzo/FastBinarySearch) which we use for CPU quantization.
