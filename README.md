# `bitsandbytes`

[![Downloads](https://static.pepy.tech/badge/bitsandbytes)](https://pepy.tech/project/bitsandbytes) [![Downloads](https://static.pepy.tech/badge/bitsandbytes/month)](https://pepy.tech/project/bitsandbytes) [![Downloads](https://static.pepy.tech/badge/bitsandbytes/week)](https://pepy.tech/project/bitsandbytes)

The `bitsandbytes` library is a lightweight Python wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and 8 & 4-bit quantization functions.

The library includes quantization primitives for 8-bit & 4-bit operations, through `bitsandbytes.nn.Linear8bitLt` and `bitsandbytes.nn.Linear4bit` and 8-bit optimizers through `bitsandbytes.optim` module.

There are ongoing efforts to support further hardware backends, i.e. Intel CPU + GPU, AMD GPU, Apple Silicon. Windows support is quite far along and is on its way as well.

**Please head to the official documentation page:**

**[https://huggingface.co/docs/bitsandbytes/main](https://huggingface.co/docs/bitsandbytes/main)**

## `𝗯𝗶𝘁𝘀𝗮𝗻𝗱𝗯𝘆𝘁𝗲𝘀` 𝗺𝘂𝗹𝘁𝗶-𝗯𝗮𝗰𝗸𝗲𝗻𝗱 𝙖𝙡𝙥𝙝𝙖 𝗿𝗲𝗹𝗲𝗮𝘀𝗲 is out!

🚀 Big news! After months of hard work and incredible community contributions, we're thrilled to announce the 𝗯𝗶𝘁𝘀𝗮𝗻𝗱𝗯𝘆𝘁𝗲𝘀 𝗺𝘂𝗹𝘁𝗶-𝗯𝗮𝗰𝗸𝗲𝗻𝗱 𝙖𝙡𝙥𝙝𝙖 𝗿𝗲𝗹𝗲𝗮𝘀𝗲! 💥

Now supporting:
- 🔥 𝗔𝗠𝗗 𝗚𝗣𝗨𝘀 (ROCm)
- ⚡ 𝗜𝗻𝘁𝗲𝗹 𝗖𝗣𝗨𝘀 & 𝗚𝗣𝗨𝘀

We’d love your early feedback! 🙏

👉 [Instructions for your `𝚙𝚒𝚙 𝚒𝚗𝚜𝚝𝚊𝚕𝚕` here](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend)

We're super excited about these recent developments and grateful for any constructive input or support that you can give to help us make this a reality (e.g. helping us with the upcoming Apple Silicon backend or reporting bugs). BNB is a community project and we're excited for your collaboration 🤗

## License

`bitsandbytes` is MIT licensed.

We thank Fabio Cannizzo for his work on [FastBinarySearch](https://github.com/fabiocannizzo/FastBinarySearch) which we use for CPU quantization.
