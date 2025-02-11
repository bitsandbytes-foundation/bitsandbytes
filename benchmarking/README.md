# Benchmarking

## Inference
End-to-end inference benchmarking can be performed using the 🤗 [`optimum-benchmark`](https://github.com/huggingface/optimum-benchmark) library.

See the example script in
[inference_benchmark.py](inference_benchmark.py).

### Results (as of v0.45.0)

Our overall benchmarking results compared with v0.44.1 provide the following insights:
#### LLM.int8()
* **Turing/Ampere/Ada**: The observed per-token throughput is improved by 60-85%, while latency is decreased by 40-45%.
* **H100**: With our benchmarking of Llama 3.1 70B, we observed the new LLM.int8() to consistently outperform NF4 at batch size >= 8.

#### NF4/FP4
* **Turing/Ampere/Ada**: With batch size of 1, per-token throughput is _improved by 10-25%_ and per-token latency is _decreased by 10-20%_.
* **H100**: Across all batch sizes, per-token throughput is _improved by up to 28%_ and per-token latency is _decreased by up to 22%_.

Summaries with the benchmarking results are provided below.

#### NVIDIA T4 16GB
<details>
<summary>Qwen 2.5 3B Instruct</summary>

|                      | Batch Size | Mean Latency (s) <sub>v0.45.0.dev</sub> | Throughput <sub>v0.45.0.dev</sub> | Mean Latency (s) <sub>v0.44.1</sub> | Latency Improvement | Throughput <sub>v0.44.1</sub> | Throughput Improvement |
|----------------------|------------|------------------------------|------------------------|--------------------------|---------------------|--------------------|------------------------|
| FP16                 | 1          | 0.0390                       | 25.66                  | 0.0390                   | 1.00                | 25.66              | 1.000x                 |
| NF4                  | 1          | 0.0608                       | 16.45                  | 0.0710                   | 1.14                | 14.08              | 1.168x                 |
| NF4+DQ               | 1          | 0.0736                       | 13.58                  | 0.0905                   | 1.19                | 11.05              | 1.229x                 |
| INT8                 | 1          | 0.0902                       | 11.08                  | 0.1609                   | 1.44                | 6.21               | 1.784x                 |
| INT8+Decomp          | 1          | 0.1672                       | 5.98                   | 0.2994                   | 1.44                | 3.34               | 1.790x                 |
| FP16                 | 8          | 0.0422                       | 189.56                 | 0.0422                   | 1.00                | 189.56             | 1.000x                 |
| NF4                  | 8          | 0.0960                       | 83.37                  | 0.1010                   | 1.05                | 79.17              | 1.053x                 |
| NF4+DQ               | 8          | 0.1042                       | 76.80                  | 0.1156                   | 1.10                | 69.18              | 1.110x                 |
| INT8                 | 8          | 0.0919                       | 87.01                  | 0.1640                   | 1.44                | 48.78              | 1.784x                 |
| INT8+Decomp          | 8          | 0.1812                       | 44.15                  | 0.3296                   | 1.45                | 24.28              | 1.818x                 |
| FP16                 | 32         | 0.0601                       | 532.30                 | 0.0601                   | 1.00                | 532.30             | 1.000x                 |
| NF4                  | 32         | 0.1150                       | 278.32                 | 0.1182                   | 1.03                | 270.71             | 1.028x                 |
| NF4+DQ               | 32         | 0.1215                       | 263.36                 | 0.1297                   | 1.06                | 246.76             | 1.067x                 |
| INT8                 | 32         | 0.0943                       | 339.21                 | 0.1640                   | 1.42                | 195.14             | 1.738x                 |
| INT8+Decomp          | 32         | 0.1912                       | 167.37                 | 0.3413                   | 1.44                | 93.75              | 1.785x                 |
</details>

#### NVIDIA RTX 4090 24GB
<details>
<summary>Llama 3.1 8B</summary>

|                      | Batch Size | Mean Latency (s) <sub>v0.45.0.dev</sub> | Throughput <sub>v0.45.0.dev</sub> | Mean Latency (s) <sub>v0.44.1</sub> | Latency Improvement | Throughput <sub>v0.44.1</sub> | Throughput Improvement |
|----------------------|------------|------------------------------|------------------------|--------------------------|---------------------|--------------------|------------------------|
| BF16        | 1  | 0.0211 | 47.46   | 0.0211 | 1.00 | 47.46   | 1.000x |
| NF4         | 1  | 0.0148 | 67.71   | 0.0164 | 1.10 | 61.08   | 1.109x |
| NF4+DQ      | 1  | 0.0175 | 57.08   | 0.0208 | 1.16 | 48.15   | 1.185x |
| INT8        | 1  | 0.0220 | 45.39   | 0.0395 | 1.44 | 25.32   | 1.793x |
| INT8+Decomp | 1  | 0.0449 | 22.26   | 0.0743 | 1.40 | 13.45   | 1.655x |
| BF16        | 8  | 0.0239 | 334.64  | 0.0239 | 1.00 | 334.64  | 1.000x |
| NF4         | 8  | 0.0425 | 188.08  | 0.0422 | 0.99 | 189.50  | 0.993x |
| NF4+DQ      | 8  | 0.0443 | 180.68  | 0.0437 | 0.99 | 183.02  | 0.987x |
| INT8        | 8  | 0.0221 | 361.61  | 0.0389 | 1.43 | 205.82  | 1.757x |
| INT8+Decomp | 8  | 0.0478 | 164.55  | 0.0777 | 1.38 | 103.01  | 1.597x |
| BF16        | 32 | 0.0304 | 1054.35 | 0.0304 | 1.00 | 1054.35 | 1.000x |
| NF4         | 32 | 0.0461 | 694.60  | 0.0466 | 1.01 | 686.90  | 1.011x |
| NF4+DQ      | 32 | 0.0471 | 678.73  | 0.0480 | 1.02 | 666.33  | 1.019x |
| INT8        | 32 | 0.0230 | 1390.54 | 0.0390 | 1.41 | 819.99  | 1.696x |
| INT8+Decomp | 32 | 0.0512 | 624.94  | 0.0835 | 1.39 | 383.18  | 1.631x |
</details>

<details>
<summary>Qwen 2.5 14B Instruct</summary>

|                      | Batch Size | Mean Latency (s) <sub>v0.45.0.dev</sub> | Throughput <sub>v0.45.0.dev</sub> | Mean Latency (s) <sub>v0.44.1</sub> | Latency Improvement | Throughput <sub>v0.44.1</sub> | Throughput Improvement |
|----------------------|------------|------------------------------|------------------------|--------------------------|---------------------|--------------------|------------------------|
| NF4         | 1 | 0.0214 | 46.74  | 0.0256 | 1.16 | 39.10  | 1.195x |
| NF4+DQ      | 1 | 0.0256 | 39.03  | 0.0318 | 1.19 | 31.46  | 1.241x |
| INT8        | 1 | 0.0326 | 30.68  | 0.0596 | 1.45 | 16.79  | 1.827x |
| INT8+Decomp | 1 | 0.0648 | 15.44  | 0.1105 | 1.41 | 9.05   | 1.706x |
| NF4         | 8 | 0.0696 | 114.95 | 0.0697 | 1.00 | 114.78 | 1.001x |
| NF4+DQ      | 8 | 0.0719 | 111.29 | 0.0723 | 1.01 | 110.70 | 1.005x |
| INT8        | 8 | 0.0325 | 246.22 | 0.0596 | 1.45 | 134.21 | 1.835x |
| INT8+Decomp | 8 | 0.0721 | 110.95 | 0.1201 | 1.40 | 66.62  | 1.665x |
</details>


#### NVIDIA H100 80GB SXM
<details>
<summary>Llama 3.1 8B</summary>

|                      | Batch Size | Mean Latency (s) <sub>v0.45.0.dev</sub> | Throughput <sub>v0.45.0.dev</sub> | Mean Latency (s) <sub>v0.44.1</sub> | Latency Improvement | Throughput <sub>v0.44.1</sub> | Throughput Improvement |
|----------------------|------------|------------------------------|------------------------|--------------------------|---------------------|--------------------|------------------------|
| BF16        | 1  | 0.0244 | 40.99   | 0.0244 | 1.00 | 40.99   | 1.000x |
| NF4         | 1  | 0.0331 | 30.14   | 0.0391 | 1.15 | 25.60   | 1.177x |
| NF4+DQ      | 1  | 0.0411 | 24.34   | 0.0528 | 1.22 | 18.92   | 1.286x |
| INT8        | 1  | 0.0522 | 19.17   | N/A    | N/A  | N/A     | N/A    |
| INT8+Decomp | 1  | 0.0817 | 12.24   | N/A    | N/A  | N/A     | N/A    |
| BF16        | 8  | 0.0255 | 313.90  | 0.0255 | 1.00 | 313.90  | 1.000x |
| NF4         | 8  | 0.0476 | 168.05  | 0.0551 | 1.14 | 145.13  | 1.158x |
| NF4+DQ      | 8  | 0.0566 | 141.27  | 0.0663 | 1.15 | 120.67  | 1.171x |
| INT8        | 8  | 0.0515 | 155.44  | N/A    | N/A  | N/A     | N/A    |
| INT8+Decomp | 8  | 0.0853 | 93.79   | N/A    | N/A  | N/A     | N/A    |
| BF16        | 32 | 0.0261 | 1227.96 | 0.0261 | 1.00 | 1227.96 | 1.000x |
| NF4         | 32 | 0.0486 | 658.65  | 0.0546 | 1.11 | 585.91  | 1.124x |
| NF4+DQ      | 32 | 0.0577 | 555.06  | 0.0665 | 1.13 | 481.04  | 1.154x |
| INT8        | 32 | 0.0545 | 586.26  | N/A    | N/A  | N/A     | N/A    |
| INT8+Decomp | 32 | 0.0864 | 370.51  | N/A    | N/A  | N/A     | N/A    |
</details>

<details>
<summary>Qwen 2.5 32B Instruct</summary>

|             | Batch Size | Mean Latency (s) <sub>v0.45.0.dev</sub> | Throughput <sub>v0.45.0.dev</sub> |
|-------------|------------|-----------------------------------------|-----------------------------------|
| BF16        | 1  | 0.0508 | 19.67  |
| NF4         | 1  | 0.0707 | 14.14  |
| NF4+DQ      | 1  | 0.0860 | 11.63  |
| INT8        | 1  | 0.1031 | 9.70   |
| INT8+Decomp | 1  | 0.1820 | 5.49   |
| BF16        | 8  | 0.0525 | 152.50 |
| NF4         | 8  | 0.1154 | 69.35  |
| NF4+DQ      | 8  | 0.1209 | 66.19  |
| INT8        | 8  | 0.1078 | 74.24  |
| INT8+Decomp | 8  | 0.1958 | 40.87  |
| BF16        | 32 | 0.0547 | 584.54 |
| NF4         | 32 | 0.1246 | 256.84 |
| NF4+DQ      | 32 | 0.1298 | 246.47 |
| INT8        | 32 | 0.1056 | 302.96 |
| INT8+Decomp | 32 | 0.2027 | 157.83 |
</details>

<details>
<summary>Llama 3.1 70B</summary>

|             | Batch Size | Mean Latency (s) <sub>v0.45.0.dev</sub> | Throughput <sub>v0.45.0.dev</sub> |
|-------------|------------|-----------------------------------------|-----------------------------------|
| NF4         | 1  | 0.0833 | 12.00  |
| NF4+DQ      | 1  | 0.1052 | 9.50   |
| INT8        | 1  | 0.1294 | 7.73   |
| INT8+Decomp | 1  | 0.1985 | 5.04   |
| NF4         | 8  | 0.2348 | 34.07  |
| NF4+DQ      | 8  | 0.2423 | 33.01  |
| INT8        | 8  | 0.1313 | 60.94  |
| INT8+Decomp | 8  | 0.2052 | 38.99  |
| NF4         | 32 | 0.2491 | 128.46 |
| NF4+DQ      | 32 | 0.2580 | 124.04 |
| INT8        | 32 | 0.1314 | 243.45 |
| INT8+Decomp | 32 | 0.2189 | 146.19 |
</details>

#### Software Configuration
We focus on the default PyTorch CUDA backend in 🤗 [`optimum-benchmark`](https://github.com/huggingface/optimum-benchmark). We used commit [`6e6b1036`](https://github.com/huggingface/optimum-benchmark/commit/6e6b10363f3ac65926881f2c6a6113b6cefc06cd).

For all hardware configurations, we used the following dependencies:
* `transformers==4.46.3`
* `accelerate==1.1.1`
* `tokenizers==0.20.3`
* `torch==2.5.1`
* `bitsandbytes==0.44.1`
* `bitsandbytes==0.45.0.dev`

In the RTX 4090 setting, the CUDA 12.4 build of PyTorch is used. In the other settings we used the CUDA 12.1 build.
