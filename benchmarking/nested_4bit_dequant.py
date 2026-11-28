#!/usr/bin/env python3
"""Benchmark fused nested 4-bit dequantization against the legacy CUDA launch chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import statistics

import torch

import bitsandbytes
from bitsandbytes import functional as F
from bitsandbytes.backends.cuda import ops as cuda_ops
from bitsandbytes.cextension import lib

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
DEFAULT_CASES = "square4096:4096:4096,wide11008:11008:4096,tall11008:4096:11008,square8192:8192:8192"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASES,
        help="Comma-separated name:rows:cols cases.",
    )
    parser.add_argument("--dtypes", default="fp16,bf16,fp32")
    parser.add_argument("--formats", default="nf4,fp4")
    parser.add_argument("--blocksize", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_cases(value: str) -> list[tuple[str, int, int]]:
    cases = []
    for item in value.split(","):
        name, rows, cols = item.split(":")
        cases.append((name, int(rows), int(cols)))
    return cases


def require_sm103() -> torch.cuda.DeviceProperties:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one CUDA device, got {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    if capability != (10, 3) or props.multi_processor_count != 148 or "B300" not in props.name.upper():
        raise RuntimeError(
            f"expected B300 SM103 with 148 SMs, got {props.name}, cc={capability}, sms={props.multi_processor_count}"
        )
    if not cuda_ops._dequantize_4bit_nested_supported(0):
        raise RuntimeError("the loaded package did not select nested dequantization for SM103")
    return props


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def timed_batch(function, repetitions: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repetitions


def benchmark_pair(baseline, candidate, warmup: int, rounds: int, repetitions: int) -> dict:
    for index in range(warmup):
        (baseline if index % 2 == 0 else candidate)()
    torch.cuda.synchronize()

    samples = {"legacy": [], "nested": []}
    for round_index in range(rounds):
        order = ("legacy", "nested") if round_index % 2 == 0 else ("nested", "legacy")
        for name in order:
            function = baseline if name == "legacy" else candidate
            samples[name].append(timed_batch(function, repetitions))

    result = {}
    for name, values in samples.items():
        result[name] = {
            "samples_us": values,
            "median_us": statistics.median(values),
            "p10_us": percentile(values, 0.1),
            "p90_us": percentile(values, 0.9),
        }
    result["ratio"] = result["legacy"]["median_us"] / result["nested"]["median_us"]
    return result


def repetitions_for(numel: int, requested: int) -> int:
    if numel >= 60_000_000:
        return min(requested, 25)
    if numel >= 40_000_000:
        return min(requested, 40)
    return requested


def raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8),
        right.contiguous().reshape(-1).view(torch.uint8),
    )


def emit(handle, record: dict) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.rounds < 1 or args.repetitions < 1:
        raise ValueError("warmup, rounds, and repetitions must be positive")

    props = require_sm103()
    cases = parse_cases(args.cases)
    dtype_names = args.dtypes.split(",")
    formats = args.formats.split(",")
    if any(name not in DTYPES for name in dtype_names):
        raise ValueError(f"unsupported dtype list: {args.dtypes}")
    if any(name not in ("nf4", "fp4") for name in formats):
        raise ValueError(f"unsupported format list: {args.formats}")

    package_library = Path(lib._lib._name).resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        emit(
            handle,
            {
                "kind": "metadata",
                "hostname": socket.gethostname(),
                "gpu": props.name,
                "capability": list(torch.cuda.get_device_capability(0)),
                "sms": props.multi_processor_count,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "bitsandbytes": bitsandbytes.__version__,
                "library": str(package_library),
                "cases": cases,
                "dtypes": dtype_names,
                "formats": formats,
                "blocksize": args.blocksize,
                "warmup": args.warmup,
                "rounds": args.rounds,
                "repetitions": args.repetitions,
                "seed": args.seed,
            },
        )

        ratios = []
        for case_index, (name, rows, cols) in enumerate(cases):
            numel = rows * cols
            for dtype_name in dtype_names:
                dtype = DTYPES[dtype_name]
                for quant_type in formats:
                    generator = torch.Generator(device="cuda").manual_seed(args.seed + case_index)
                    source = torch.randn((rows, cols), generator=generator, device="cuda", dtype=dtype)
                    packed, state = F.quantize_4bit(
                        source,
                        blocksize=args.blocksize,
                        compress_statistics=True,
                        quant_type=quant_type,
                    )
                    del source

                    legacy_scale = torch.empty_like(state.absmax, dtype=torch.float32)
                    legacy_out = torch.empty((rows, cols), device="cuda", dtype=dtype)
                    nested_out = torch.empty_like(legacy_out)

                    def legacy():
                        cuda_ops._dequantize_blockwise_impl(
                            state.absmax,
                            state.state2.absmax,
                            state.state2.code,
                            state.state2.blocksize,
                            torch.float32,
                            legacy_scale,
                        )
                        legacy_scale.add_(state.offset)
                        cuda_ops._dequantize_4bit_impl(
                            packed,
                            legacy_scale,
                            state.blocksize,
                            state.quant_type,
                            state.dtype,
                            legacy_out,
                        )

                    def nested():
                        cuda_ops._dequantize_4bit_nested_impl(
                            packed,
                            state.absmax,
                            state.state2.absmax,
                            state.state2.code,
                            state.offset,
                            state.blocksize,
                            state.quant_type,
                            state.dtype,
                            nested_out,
                        )

                    repetitions = repetitions_for(numel, args.repetitions)
                    result = benchmark_pair(legacy, nested, args.warmup, args.rounds, repetitions)
                    legacy()
                    nested()
                    torch.cuda.synchronize()
                    if not raw_equal(legacy_out, nested_out):
                        raise AssertionError(f"raw output mismatch for {name}/{dtype_name}/{quant_type}")

                    ratios.append(result["ratio"])
                    output_bytes = numel * dtype.itemsize
                    emit(
                        handle,
                        {
                            "kind": "direct",
                            "case": name,
                            "rows": rows,
                            "cols": cols,
                            "numel": numel,
                            "dtype": dtype_name,
                            "quant_type": quant_type,
                            "blocksize": args.blocksize,
                            "repetitions": repetitions,
                            "output_bytes": output_bytes,
                            "legacy_effective_gbps": output_bytes / result["legacy"]["median_us"] / 1e3,
                            "nested_effective_gbps": output_bytes / result["nested"]["median_us"] / 1e3,
                            "raw_equal": True,
                            **result,
                        },
                    )
        emit(
            handle,
            {
                "kind": "summary",
                "cells": len(ratios),
                "geomean_ratio": math.exp(statistics.mean(math.log(value) for value in ratios)),
                "minimum_ratio": min(ratios),
            },
        )


if __name__ == "__main__":
    main()
