"""
Inference benchmarking tool.

Requirements:
    transformers
    accelerate
    bitsandbytes
    optimum-benchmark

Usage: python inference_benchmark.py model_id

options:
    -h, --help            show this help message and exit
    --configs {bf16,fp16,nf4,nf4-dq,int8,int8-decomp} [{bf16,fp16,nf4,nf4-dq,int8,int8-decomp} ...]
    --bf16
    --fp16
    --nf4
    --nf4-dq
    --int8
    --int8-decomp
    --batches BATCHES [BATCHES ...]
    --input-length INPUT_LENGTH
    --out-dir OUT_DIR
"""

import argparse
from pathlib import Path

from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging
import torch

BFLOAT16_SUPPORT = torch.cuda.get_device_capability()[0] >= 8

WEIGHTS_CONFIGS = {
    "fp16": {"torch_dtype": "float16", "quantization_scheme": None, "quantization_config": {}},
    "bf16": {"torch_dtype": "bfloat16", "quantization_scheme": None, "quantization_config": {}},
    "nf4": {
        "torch_dtype": "bfloat16" if BFLOAT16_SUPPORT else "float16",
        "quantization_scheme": "bnb",
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_compute_dtype": torch.bfloat16 if BFLOAT16_SUPPORT else "float16",
        },
    },
    "nf4-dq": {
        "torch_dtype": "bfloat16" if BFLOAT16_SUPPORT else "float16",
        "quantization_scheme": "bnb",
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.bfloat16 if BFLOAT16_SUPPORT else "float16",
        },
    },
    "int8-decomp": {
        "torch_dtype": "float16",
        "quantization_scheme": "bnb",
        "quantization_config": {
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
        },
    },
    "int8": {
        "torch_dtype": "float16",
        "quantization_scheme": "bnb",
        "quantization_config": {
            "load_in_8bit": True,
            "llm_int8_threshold": 0.0,
        },
    },
}

if __name__ == "__main__":
    setup_logging(level="INFO")

    parser = argparse.ArgumentParser(description="bitsandbytes inference benchmark tool")

    parser.add_argument("model_id", type=str, help="The model checkpoint to use.")

    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["bf16", "fp16", "nf4", "nf4-dq", "int8", "int8-decomp"],
        default=["nf4", "int8", "int8-decomp"],
    )
    parser.add_argument("--bf16", dest="configs", action="append_const", const="bf16")
    parser.add_argument("--fp16", dest="configs", action="append_const", const="fp16")
    parser.add_argument("--nf4", dest="configs", action="append_const", const="nf4")
    parser.add_argument("--nf4-dq", dest="configs", action="append_const", const="nf4-dq")
    parser.add_argument("--int8", dest="configs", action="append_const", const="int8")
    parser.add_argument("--int8-decomp", dest="configs", action="append_const", const="int8-decomp")

    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--input-length", type=int, default=64)

    parser.add_argument("--out-dir", type=str, default="reports")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for batch_size in args.batches:
        print(f"Benchmarking batch size: {batch_size}")
        for config in args.configs:
            launcher_config = ProcessConfig(device_isolation=True, start_method="spawn")
            scenario_config = InferenceConfig(
                latency=True,
                memory=True,
                input_shapes={"batch_size": batch_size, "sequence_length": args.input_length},
            )
            backend_config = PyTorchConfig(
                device="cuda",
                device_ids="0",
                device_map="auto",
                no_weights=False,
                model=args.model_id,
                **WEIGHTS_CONFIGS[config],
            )
            benchmark_config = BenchmarkConfig(
                name=f"benchmark-{config}-bsz{batch_size}",
                scenario=scenario_config,
                launcher=launcher_config,
                backend=backend_config,
            )

            out_path = out_dir / f"benchmark_{config}_bsz{batch_size}.json"

            benchmark_report = Benchmark.launch(benchmark_config)
            benchmark_report.log()
            benchmark_report.save_json(out_path)
