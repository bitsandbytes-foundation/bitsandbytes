# `bitsandbytes`

The `bitsandbytes` library is a lightweight Python wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and 8 & 4-bit quantization functions.

The library includes quantization primitives for 8-bit & 4-bit operations, through `bitsandbytes.nn.Linear8bitLt` and `bitsandbytes.nn.Linear4bit` and 8-bit optimizers through `bitsandbytes.optim` module.

There are ongoing efforts to support further hardware backends, i.e. Intel CPU + GPU, AMD GPU, Apple Silicon. Windows support is quite far along and is on its way as well.

**Please head to the official documentation page:**

**[https://huggingface.co/docs/bitsandbytes/main](https://huggingface.co/docs/bitsandbytes/main)**

## License

The majority of bitsandbytes is licensed under MIT, however small portions of the project are available under separate license terms, as the parts adapted from Pytorch are licensed under the BSD license.

We thank Fabio Cannizzo for his work on [FastBinarySearch](https://github.com/fabiocannizzo/FastBinarySearch) which we use for CPU quantization.
