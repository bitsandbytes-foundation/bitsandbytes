# `bitsandbytes`

[![Downloads](https://static.pepy.tech/badge/bitsandbytes)](https://pepy.tech/project/bitsandbytes) [![Downloads](https://static.pepy.tech/badge/bitsandbytes/month)](https://pepy.tech/project/bitsandbytes) [![Downloads](https://static.pepy.tech/badge/bitsandbytes/week)](https://pepy.tech/project/bitsandbytes)

The `bitsandbytes` library is a lightweight Python wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and 8 & 4-bit quantization functions.

The library includes quantization primitives for 8-bit & 4-bit operations, through `bitsandbytes.nn.Linear8bitLt` and `bitsandbytes.nn.Linear4bit` and 8-bit optimizers through `bitsandbytes.optim` module.

There are ongoing efforts to support further hardware backends, i.e. Intel CPU + GPU, AMD GPU, Apple Silicon. Windows support is quite far along and is on its way as well.

**Please head to the official documentation page:**

**[https://huggingface.co/docs/bitsandbytes/main](https://huggingface.co/docs/bitsandbytes/main)**

## ALPHA TESTERS WANTED: `multi-backend-refactor` AMD GPU + Intel CPU/GPU specific BNB backend implementations

We're in the process of a complex refactor in order to allow the support of additional hardware backends, other than CUDA, in BNB. The efforts around this are already quite far along and there's plenty of functionality already in place that is in need for users to take a hands-on approach! Mac support will likely soon also see progress. However, I recommend waiting 2 weeks until the device abstraction has further consolidated (**breaking changes upcoming**).

Currently, you still need to compile from source, after checking out the `multi-backend-refactor` branch (instructions WIP, but [the current docs on the compilation from source](https://huggingface.co/docs/bitsandbytes/main/en/installation#compile-from-source) are a good starting point; [feel free to share tips / input in this Github discussion](https://github.com/TimDettmers/bitsandbytes/discussions/1219). We'll soon enable nightly releases to make this much easier for you!

Please give feedback to us in [this dedicated Github Discussion space](https://github.com/TimDettmers/bitsandbytes/discussions/categories/catch-all-alpha-testing-the-multi-backend-refactor)!

We're super excited about these recent developments and grateful for any constructive input or support that you can give to help us make this a reality. BNB is a community project and we're excited for your collaboration 🤗

## License

`bitsandbytes` is MIT licensed.

We thank Fabio Cannizzo for his work on [FastBinarySearch](https://github.com/fabiocannizzo/FastBinarySearch) which we use for CPU quantization.
