# `bitsandbytes`

[![Downloads](https://static.pepy.tech/badge/bitsandbytes)](https://pepy.tech/project/bitsandbytes) [![Downloads](https://static.pepy.tech/badge/bitsandbytes/month)](https://pepy.tech/project/bitsandbytes) [![Downloads](https://static.pepy.tech/badge/bitsandbytes/week)](https://pepy.tech/project/bitsandbytes)

The `bitsandbytes` library is a lightweight Python wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and 8 & 4-bit quantization functions.

The library includes quantization primitives for 8-bit & 4-bit operations, through `bitsandbytes.nn.Linear8bitLt` and `bitsandbytes.nn.Linear4bit` and 8-bit optimizers through `bitsandbytes.optim` module.

There are ongoing efforts to support further hardware backends, i.e. Intel CPU + GPU, AMD GPU, Apple Silicon. Windows support is quite far along and is on its way as well.

**Please head to the official documentation page:**

**[https://huggingface.co/docs/bitsandbytes/main](https://huggingface.co/docs/bitsandbytes/main)**

## `bitsandbytes` multi-backend _alpha_ release is out!

🚀 Big news! After months of hard work and incredible community contributions, we're thrilled to announce the **bitsandbytes multi-backend _alpha_ release**! 💥

Now supporting:
- 🔥 **AMD GPUs** (ROCm)
- ⚡ **Intel CPUs** & **GPUs**

We’d love your early feedback! 🙏

👉 [Instructions for your `pip install` here](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend)

We're super excited about these recent developments and grateful for any constructive input or support that you can give to help us make this a reality (e.g. helping us with the upcoming Apple Silicon backend or reporting bugs). BNB is a community project and we're excited for your collaboration 🤗

## License

`bitsandbytes` is MIT licensed.

We thank Fabio Cannizzo for his work on [FastBinarySearch](https://github.com/fabiocannizzo/FastBinarySearch) which we use for CPU quantization.
