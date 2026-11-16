<p align="center"><img src="https://avatars.githubusercontent.com/u/175231607?s=200&v=4" alt=""></p>
<h1 align="center">bitsandbytes</h1>
<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>
<p align="center">
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/bitsandbytes-foundation/bitsandbytes.svg?color=blue"></a>
    <a href="https://pepy.tech/project/bitsandbytes"><img alt="Downloads" src="https://static.pepy.tech/badge/bitsandbytes/month"></a>
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes/actions/workflows/tests-nightly.yml"><img alt="Nightly Unit Tests" src="https://img.shields.io/github/actions/workflow/status/bitsandbytes-foundation/bitsandbytes/tests-nightly.yml?logo=github&label=Nightly%20Tests"></a>
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/bitsandbytes-foundation/bitsandbytes"></a>
    <a href="https://pypi.org/project/bitsandbytes/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/bitsandbytes"></a>
</p>

`bitsandbytes` 通过面向 PyTorch 的 k-bit 量化技术，大幅降低了大语言模型（LLM）的使用与微调门槛。该库提供了三大核心特性，可显著削减推理与训练期间的显存占用：

* **8-bit 优化器**：采用分块量化（block-wise quantization）技术，在显存开销仅为原有极小比例的前提下，完整维持 32-bit 的优化器精度与收敛性能。
* **LLM.int8() 或 8-bit 量化**：仅需约一半显存即可实现大语言模型推理，且无任何性能退化。该方法基于逐向量量化（vector-wise quantization），将绝大多数特征量化至 8-bit，同时对异常离群值（outliers）使用 16-bit 矩阵乘法单独计算。
* **QLoRA 或 4-bit 量化**：结合多项创新的显存节省技术，在不妥协模型能力的前提下实现大语言模型的高效微调。该方法将基座模型权重量化为 4-bit，并注入极小规模的可训练低秩自适应（LoRA）适配层进行训练。

该库通过 `bitsandbytes.nn.Linear8bitLt` 与 `bitsandbytes.nn.Linear4bit` 提供了 8-bit 与 4-bit 量化基础算子，并通过 `bitsandbytes.optim` 模块提供了全套 8-bit 优化器。

## 系统环境要求 (System Requirements)
bitsandbytes 在所有支持平台上的最低环境要求如下：

* Python 3.10+
* [PyTorch](https://pytorch.org/get-started/locally/) 2.4+
  * _注意：虽然开发团队致力于保持广泛的向下兼容性，但仍推荐使用最新版本的 PyTorch 以获取最佳体验。_

#### 硬件加速器支持矩阵 (Accelerator support):

<small>注意：本表格反映当前开发分支的状态。若查阅最新的稳定版本，请参考 [0.50.0 标签文档](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.0/README.md#accelerator-support)。
</small>

##### 图例说明：
🚧 = 计划支持 |
〰️ = 部分支持 |
✅ = 完整支持 |
❌ = 暂不支持

<table>
  <thead>
    <tr>
      <th>操作系统平台</th>
      <th>加速器类型</th>
      <th>硬件规格要求</th>
      <th>LLM.int8()</th>
      <th>QLoRA 4-bit</th>
      <th>8-bit 优化器</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="6">🐧 <strong>Linux, glibc >= 2.24</strong></td>
    </tr>
    <tr>
      <td align="right">x86-64</td>
      <td>◻️ CPU</td>
      <td>最低：AVX2<br>针对性优化：AVX512F, AVX512BF16</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>最低要求 SM60+<br>推荐 SM75+</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟥 AMD GPU <br><code>cuda</code></td>
      <td>
        CDNA: gfx908, gfx90a, gfx942, gfx950, gfx1250<br>
        RDNA: gfx101X, gfx103X, gfx110X, gfx115X, gfx120X
      </td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟦 Intel GPU <br><code>xpu</code></td>
      <td>
        Data Center GPU Max 系列<br>
        Arc A 系列 (Alchemist)<br>
        Arc B 系列 (Battlemage)
      </td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟪 Intel Gaudi <br><code>hpu</code></td>
      <td>Gaudi2, Gaudi3</td>
      <td>✅</td>
      <td>〰️</td>
      <td>❌</td>
    </tr>
    <tr>
      <td align="right">aarch64</td>
      <td>◻️ CPU</td>
      <td></td>
      <td>✅ *</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM75+</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td colspan="6">🪟 <strong>Windows 11 / Windows Server 2022+</strong></td>
    </tr>
    <tr>
      <td align="right">x86-64</td>
      <td>◻️ CPU</td>
      <td>AVX2</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>最低要求 SM60+<br>推荐 SM75+</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟥 AMD GPU <br><code>cuda</code></td>
      <td>
        CDNA: gfx908, gfx90a<br>
        RDNA: gfx101X, gfx103X, gfx110X, gfx115X, gfx120X
      </td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟦 Intel GPU <br><code>xpu</code></td>
      <td>
        Arc A 系列 (Alchemist) <br>
        Arc B 系列 (Battlemage)
      </td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td align="right">arm64</td>
      <td>◻️ CPU</td>
      <td></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM121</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td colspan="6">🍎 <strong>macOS 14+</strong></td>
    </tr>
    <tr>
      <td align="right">arm64</td>
      <td>◻️ CPU</td>
      <td>Apple M1+</td>
      <td>✅ *</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>⬜ Metal <br><code>mps</code></td>
      <td>Apple M1+</td>
      <td>✅ *</td>
      <td>✅</td>
      <td>🚧</td>
  </tbody>
</table>
<sup>* 标注星号的功能特性虽然已支持，但可能尚缺少极致的性能优化。</sup>

## :book: 官方文档与生态集成 (Documentation)
* [官方文档中心](https://huggingface.co/docs/bitsandbytes/main)
* 🤗 [Transformers 官方量化集成指南](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
* 🤗 [Diffusers 官方量化集成指南](https://huggingface.co/docs/diffusers/quantization/bitsandbytes)
* 🤗 [PEFT 官方量化适配器指南](https://huggingface.co/docs/peft/developer_guides/quantization#quantize-a-model)

## :heart: 赞助支持 (Sponsors)
`bitsandbytes` 的持续维护与特性演进离不开赞助商的慷慨支持。他们的持续资助确保了开发团队能够不断改进项目，并向开源社区提供高质量的更新。

<kbd><a href="https://hf.co" target="_blank"><img width="100" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" alt="Hugging Face"></a></kbd>

## 开源许可证 (License)
`bitsandbytes` 遵循 MIT 开源许可证。

## 论文引用 (How to cite us)
如果您在学术研究中使用了本库，请考虑引用以下论文：

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

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年11月16日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
