<p align="center"><img src="https://avatars.githubusercontent.com/u/175231607?s=200&v=4" alt=""></p>
<h1 align="center">bitsandbytes</h1>
<p align="center">
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/bitsandbytes-foundation/bitsandbytes.svg?color=blue"></a>
    <a href="https://pepy.tech/project/bitsandbytes"><img alt="Downloads" src="https://static.pepy.tech/badge/bitsandbytes/month"></a>
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes/actions/workflows/tests.yml"><img alt="Nightly Unit Tests" src="https://img.shields.io/github/actions/workflow/status/bitsandbytes-foundation/bitsandbytes/tests.yml?logo=github&label=Nightly%20Tests"></a>
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/bitsandbytes-foundation/bitsandbytes"></a>
    <a href="https://pypi.org/project/bitsandbytes/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/bitsandbytes"></a>
</p>

`bitsandbytes` enables accessible large language models via k-bit quantization for PyTorch. We provide three main features for dramatically reducing memory consumption for inference and training:

* 8-bit optimizers uses block-wise quantization to maintain 32-bit performance at a small fraction of the memory cost.
* LLM.int8() or 8-bit quantization enables large language model inference with only half the required memory and without any performance degradation. This method is based on vector-wise quantization to quantize most features to 8-bits and separately treating outliers with 16-bit matrix multiplication.
* QLoRA or 4-bit quantization enables large language model training with several memory-saving techniques that don't compromise performance. This method quantizes a model to 4-bits and inserts a small set of trainable low-rank adaptation (LoRA) weights to allow training.

The library includes quantization primitives for 8-bit & 4-bit operations, through `bitsandbytes.nn.Linear8bitLt` and `bitsandbytes.nn.Linear4bit` and 8-bit optimizers through `bitsandbytes.optim` module.

## System Requirements
bitsandbytes has the following minimum requirements for all platforms:

* Python 3.9+
* [PyTorch](https://pytorch.org/get-started/locally/) 2.2+
  * _Note: While we aim to provide wide backwards compatibility, we recommend using the latest version of PyTorch for the best experience._

#### Accelerator support:

<table>
  <thead>
    <tr>
      <th>Platform</th>
      <th>Accelerator</th>
      <th>Hardware Requirements</th>
      <th>Support Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="4">🐧 <strong>Linux, glibc >= 2.24</strong></td>
    </tr>
    <tr>
      <td align="right">x86-64</td>
      <td>◻️ CPU</td>
      <td>AVX2</td>
      <td>〰️ Partial Support</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM50+ minimum<br>SM75+ recommended</td>
      <td>✅ Full Support</td>
    </tr>
    <tr>
      <td></td>
      <td>🟥 AMD GPU <br><code>cuda</code></td>
      <td>
        CDNA: gfx90a, gfx942<br>
        RDNA: gfx1100, gfx1200
      </td>
      <td>🚧 In Development</td>
    </tr>
    <tr>
      <td></td>
      <td>🟦 Intel GPU <br><code>xpu</code></td>
      <td>
        Data Center GPU Max Series<br>
        Arc A-Series (Alchemist)<br>
        Arc B-Series (Battlemage)
      </td>
      <td>🚧 In Development</td>
    </tr>
    <tr>
      <td></td>
      <td>🟪 Intel Gaudi <br><code>hpu</code></td>
      <td>Gaudi1, Gaudi2, Gaudi3</td>
      <td>🚧 In Development</td>
    </tr>
    <tr>
      <td align="right">aarch64</td>
      <td>◻️ CPU</td>
      <td></td>
      <td>〰️ Partial Support</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM75, SM80, SM90, SM100</td>
      <td>✅ Full Support</td>
    </tr>
    <tr>
      <td colspan="4">🪟 <strong>Windows 11 / Windows Server 2019+</strong></td>
    </tr>
    <tr>
      <td align="right">x86-64</td>
      <td>◻️ CPU</td>
      <td>AVX2</td>
      <td>〰️ Partial Support</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM50+ minimum<br>SM75+ recommended</td>
      <td>✅ Full Support</td>
    </tr>
    <tr>
      <td></td>
      <td>🟦 Intel GPU <br><code>xpu</code></td>
      <td>
        Arc A-Series (Alchemist) <br>
        Arc B-Series (Battlemage)
      </td>
      <td>🚧 In Development</td>
    </tr>
    <tr>
      <td colspan="4">🍎 <strong>macOS 13.1+</strong></td>
    </tr>
    <tr>
      <td align="right">arm64</td>
      <td>◻️ CPU</td>
      <td>Apple M1+</td>
      <td>🚧 In Development</td>
    </tr>
    <tr>
      <td></td>
      <td>⬜ Metal <br><code>mps</code></td>
      <td>Apple M1+</td>
      <td>🚧 In Development</td>
  </tbody>
</table>

## :book: Documentation
* [Official Documentation](https://huggingface.co/docs/bitsandbytes/main)
* 🤗 [Transformers](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
* 🤗 [Diffusers](https://huggingface.co/docs/diffusers/quantization/bitsandbytes)
* 🤗 [PEFT](https://huggingface.co/docs/peft/developer_guides/quantization#quantize-a-model)

## :heart: Sponsors
The continued maintenance and development of `bitsandbytes` is made possible thanks to the generous support of our sponsors. Their contributions help ensure that we can keep improving the project and delivering valuable updates to the community.

<a href="https://hf.co" target="_blank"><img width="100" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" alt="Hugging Face"></a>

## License
`bitsandbytes` is MIT licensed.

We thank Fabio Cannizzo for his work on [FastBinarySearch](https://github.com/fabiocannizzo/FastBinarySearch) which we use for CPU quantization.

## How to cite us
If you found this library useful, please consider citing our work:

### QLoRA

```bibtex
@article{dettmers2023qlora,
  title={Qlora: Efficient finetuning of quantized llms},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

### LLM.int8()

```bibtex
@article{dettmers2022llmint8,
  title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2208.07339},
  year={2022}
}
```

### 8-bit Optimizers

```bibtex
@article{dettmers2022optimizers,
  title={8-bit Optimizers via Block-wise Quantization},
  author={Dettmers, Tim and Lewis, Mike and Shleifer, Sam and Zettlemoyer, Luke},
  journal={9th International Conference on Learning Representations, ICLR},
  year={2022}
}
```
