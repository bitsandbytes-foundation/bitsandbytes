#!/usr/bin/env python3
"""Benchmark legacy stream-0 and current-stream blockwise quantization."""

import argparse
from contextlib import contextmanager
import ctypes as ct
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch

import bitsandbytes as bnb
from bitsandbytes.backends.cuda import ops as cuda_ops
import bitsandbytes.functional as F

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
QUANT_TYPES = ("general8", "fp4", "nf4")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ratio_ci(baseline: list[float], candidate: list[float], iterations: int = 2000) -> tuple[float, float]:
    rng = random.Random(0)
    ratios = []
    for _ in range(iterations):
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        baseline_median = statistics.median(baseline[index] for index in indices)
        candidate_median = statistics.median(candidate[index] for index in indices)
        ratios.append(baseline_median / candidate_median)
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def summarize(baseline: list[float], candidate: list[float], round_size: int) -> dict:
    baseline_median = statistics.median(baseline)
    candidate_median = statistics.median(candidate)
    ci_low, ci_high = bootstrap_ratio_ci(baseline, candidate)
    return {
        "baseline_samples": baseline,
        "candidate_samples": candidate,
        "baseline_median": baseline_median,
        "baseline_p10": percentile(baseline, 0.1),
        "baseline_p90": percentile(baseline, 0.9),
        "candidate_median": candidate_median,
        "candidate_p10": percentile(candidate, 0.1),
        "candidate_p90": percentile(candidate, 0.9),
        "baseline_round_medians": [
            statistics.median(baseline[start : start + round_size]) for start in range(0, len(baseline), round_size)
        ],
        "candidate_round_medians": [
            statistics.median(candidate[start : start + round_size]) for start in range(0, len(candidate), round_size)
        ],
        "ratio": baseline_median / candidate_median,
        "ratio_ci95_low": ci_low,
        "ratio_ci95_high": ci_high,
    }


def configure_legacy_library(path: Path):
    library = ct.CDLL(str(path))
    argtypes = [ct.c_void_p] * 4 + [ct.c_int32, ct.c_int32]
    for dtype_name in DTYPES:
        for suffix in ("", "_fp4", "_nf4"):
            function = getattr(library, f"cquantize_blockwise_{dtype_name}{suffix}")
            function.argtypes = argtypes
            function.restype = None
    return library


def native_function(library, dtype_name: str, quant_type: str, with_stream: bool):
    suffix = "" if quant_type == "general8" else f"_{quant_type}"
    stream_suffix = "_with_stream" if with_stream else ""
    return getattr(library, f"cquantize_blockwise_{dtype_name}{suffix}{stream_suffix}")


def allocate_direct(n: int, dtype_name: str, quant_type: str, blocksize: int):
    dtype = DTYPES[dtype_name]
    tensor = torch.linspace(-4.0, 4.0, n, dtype=torch.float32, device="cuda").to(dtype)
    blocks = -(n // -blocksize)
    output_size = n if quant_type == "general8" else (n + 1) // 2
    output = torch.empty(output_size, dtype=torch.uint8, device="cuda")
    absmax = torch.empty(blocks, dtype=torch.float32, device="cuda")
    code = F.create_dynamic_map().to("cuda") if quant_type == "general8" else None
    return tensor, output, absmax, code


def invoke_direct(function, tensors, blocksize: int, with_stream: bool) -> None:
    tensor, output, absmax, code = tensors
    arguments = [
        code.data_ptr() if code is not None else None,
        tensor.data_ptr(),
        absmax.data_ptr(),
        output.data_ptr(),
        blocksize,
        tensor.numel(),
    ]
    if with_stream:
        arguments.append(cuda_ops._get_raw_stream(tensor.device.index))
    function(*arguments)


def event_us(function, tensors, blocksize: int, with_stream: bool) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    invoke_direct(function, tensors, blocksize, with_stream)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def direct_repetitions(n: int, requested: int) -> int:
    if n <= 1 << 20:
        return requested
    if n <= 1 << 24:
        return min(requested, 50)
    if n <= 1 << 27:
        return min(requested, 20)
    return min(requested, 8)


def benchmark_direct_case(
    baseline_library,
    candidate_library,
    dtype_name: str,
    quant_type: str,
    n: int,
    blocksize: int,
    warmup: int,
    rounds: int,
    repetitions: int,
) -> dict:
    baseline_tensors = allocate_direct(n, dtype_name, quant_type, blocksize)
    candidate_tensors = allocate_direct(n, dtype_name, quant_type, blocksize)
    candidate_tensors[0].copy_(baseline_tensors[0])
    baseline_function = native_function(baseline_library, dtype_name, quant_type, False)
    candidate_function = native_function(candidate_library, dtype_name, quant_type, True)

    for index in range(warmup):
        if index % 2:
            invoke_direct(candidate_function, candidate_tensors, blocksize, True)
            invoke_direct(baseline_function, baseline_tensors, blocksize, False)
        else:
            invoke_direct(baseline_function, baseline_tensors, blocksize, False)
            invoke_direct(candidate_function, candidate_tensors, blocksize, True)
    torch.cuda.synchronize()

    baseline_samples = []
    candidate_samples = []
    for round_index in range(rounds):
        for sample_index in range(repetitions):
            if (round_index + sample_index) % 2:
                candidate_samples.append(event_us(candidate_function, candidate_tensors, blocksize, True))
                baseline_samples.append(event_us(baseline_function, baseline_tensors, blocksize, False))
            else:
                baseline_samples.append(event_us(baseline_function, baseline_tensors, blocksize, False))
                candidate_samples.append(event_us(candidate_function, candidate_tensors, blocksize, True))

    assert torch.equal(baseline_tensors[1], candidate_tensors[1])
    assert torch.equal(baseline_tensors[2], candidate_tensors[2])
    result = {
        "record_type": "direct",
        "dtype": dtype_name,
        "quant_type": quant_type,
        "n": n,
        "blocksize": blocksize,
        "warmup": warmup,
        "rounds": rounds,
        "repetitions": repetitions,
        "sample_unit": "us",
        "bytes_per_call": n * DTYPES[dtype_name].itemsize,
        "effective_bandwidth_definition": "input tensor bytes divided by CUDA-event kernel latency",
        **summarize(baseline_samples, candidate_samples, repetitions),
    }
    result["baseline_effective_gbps"] = result["bytes_per_call"] / result["baseline_median"] / 1000.0
    result["candidate_effective_gbps"] = result["bytes_per_call"] / result["candidate_median"] / 1000.0
    return result


@contextmanager
def baseline_public_quantizers(baseline_library):
    originals = {}
    for dtype_name in DTYPES:
        for quant_type in QUANT_TYPES:
            suffix = "" if quant_type == "general8" else f"_{quant_type}"
            stream_name = f"cquantize_blockwise_{dtype_name}{suffix}_with_stream"
            legacy_function = native_function(baseline_library, dtype_name, quant_type, False)
            originals[stream_name] = getattr(bnb.cextension.lib, stream_name)

            def legacy_adapter(*arguments, function=legacy_function):
                function(*arguments[:-1])

            setattr(bnb.cextension.lib, stream_name, legacy_adapter)
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(bnb.cextension.lib, name, function)


def quantize_public(tensor: torch.Tensor):
    return F.quantize_4bit(tensor, blocksize=64, quant_type="nf4", compress_statistics=True)


def raw_state_equal(actual, expected) -> bool:
    if not torch.equal(actual.absmax, expected.absmax):
        return False
    if not torch.equal(actual.offset, expected.offset):
        return False
    return torch.equal(actual.state2.absmax, expected.state2.absmax)


def run_pipeline(
    variant: str,
    baseline_library,
    host_inputs: list[torch.Tensor],
    device_inputs: list[torch.Tensor],
    streams: list[torch.cuda.Stream],
    keep_outputs: bool = False,
):
    outputs = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    if variant == "baseline":
        copy_events = []
        for host_input, device_input, stream in zip(host_inputs, device_inputs, streams):
            with torch.cuda.stream(stream):
                device_input.copy_(host_input, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                copy_events.append(event)
        for event in copy_events:
            event.synchronize()
        with baseline_public_quantizers(baseline_library):
            for device_input in device_inputs:
                outputs.append(quantize_public(device_input))
    else:
        for host_input, device_input, stream in zip(host_inputs, device_inputs, streams):
            with torch.cuda.stream(stream):
                device_input.copy_(host_input, non_blocking=True)
                outputs.append(quantize_public(device_input))
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return (elapsed_ms, outputs) if keep_outputs else elapsed_ms


def benchmark_pipeline_case(
    baseline_library,
    dtype_name: str,
    shape: tuple[int, int],
    warmup: int,
    rounds: int,
    repetitions: int,
) -> dict:
    dtype = DTYPES[dtype_name]
    host_inputs = [torch.full(shape, value, dtype=dtype, pin_memory=True) for value in (-0.75, 1.25)]
    device_inputs = [torch.empty(shape, dtype=dtype, device="cuda") for _ in host_inputs]
    streams = [torch.cuda.Stream() for _ in host_inputs]

    _, baseline_outputs = run_pipeline(
        "baseline", baseline_library, host_inputs, device_inputs, streams, keep_outputs=True
    )
    _, candidate_outputs = run_pipeline(
        "candidate", baseline_library, host_inputs, device_inputs, streams, keep_outputs=True
    )
    for (baseline_output, baseline_state), (candidate_output, candidate_state) in zip(
        baseline_outputs, candidate_outputs
    ):
        assert torch.equal(baseline_output, candidate_output)
        assert raw_state_equal(baseline_state, candidate_state)

    for index in range(warmup):
        order = ("candidate", "baseline") if index % 2 else ("baseline", "candidate")
        for variant in order:
            run_pipeline(variant, baseline_library, host_inputs, device_inputs, streams)

    baseline_samples = []
    candidate_samples = []
    for round_index in range(rounds):
        for sample_index in range(repetitions):
            order = ("candidate", "baseline") if (round_index + sample_index) % 2 else ("baseline", "candidate")
            for variant in order:
                elapsed = run_pipeline(variant, baseline_library, host_inputs, device_inputs, streams)
                if variant == "baseline":
                    baseline_samples.append(elapsed)
                else:
                    candidate_samples.append(elapsed)

    input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in host_inputs)
    result = {
        "record_type": "pipeline",
        "dtype": dtype_name,
        "shape": list(shape),
        "quant_type": "nf4",
        "compress_statistics": True,
        "streams": len(streams),
        "warmup": warmup,
        "rounds": rounds,
        "repetitions": repetitions,
        "sample_unit": "ms",
        "input_bytes_per_sample": input_bytes,
        "total_timed_input_bytes_per_variant": input_bytes * rounds * repetitions,
        "effective_bandwidth_definition": "sum of both H2D input byte counts divided by synchronized host wall time",
        "baseline_serialization": "host waits for both H2D copies, then quantizes on stream 0",
        "candidate_pipeline": "each nonblocking stream chains H2D copy and public quantize_4bit",
        **summarize(baseline_samples, candidate_samples, repetitions),
    }
    result["baseline_effective_gbps"] = input_bytes / result["baseline_median"] / 1_000_000.0
    result["candidate_effective_gbps"] = input_bytes / result["candidate_median"] / 1_000_000.0
    return result


def parse_shapes(value: str) -> list[tuple[int, int]]:
    return [tuple(int(dimension) for dimension in item.lower().split("x")) for item in value.split(",")]


def write_record(handle, record: dict) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="524288,8388608,67108864,536870912")
    parser.add_argument("--pipeline-shapes", default="4096x4096,11008x4096,4096x11008")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--direct-repetitions", type=int, default=100)
    parser.add_argument("--pipeline-repetitions", type=int, default=15)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    properties = torch.cuda.get_device_properties(0)
    assert torch.cuda.get_device_capability(0) == (10, 3)
    assert "B300" in properties.name.upper()
    assert properties.multi_processor_count == 148

    baseline_path = args.baseline_library.resolve()
    candidate_path = args.candidate_library.resolve()
    loaded_path = Path(bnb.cextension.lib._lib._name).resolve()
    assert loaded_path == candidate_path, (loaded_path, candidate_path)
    baseline_library = configure_legacy_library(baseline_path)
    for dtype_name in DTYPES:
        for quant_type in QUANT_TYPES:
            native_function(baseline_library, dtype_name, quant_type, False)
            native_function(bnb.cextension.lib._lib, dtype_name, quant_type, True)

    metadata = {
        "record_type": "metadata",
        "git_revision": os.environ.get("BENCHMARK_GIT_REVISION", "unknown"),
        "baseline_revision": os.environ.get("BENCHMARK_BASELINE_REVISION", "unknown"),
        "baseline_library": str(baseline_path),
        "candidate_library": str(candidate_path),
        "baseline_library_bytes": baseline_path.stat().st_size,
        "candidate_library_bytes": candidate_path.stat().st_size,
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "num_sms": properties.multi_processor_count,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_targets": os.environ.get("BENCHMARK_CUDA_TARGETS", "unknown"),
        "warmup": args.warmup,
        "rounds": args.rounds,
        "direct_repetitions": args.direct_repetitions,
        "pipeline_repetitions": args.pipeline_repetitions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        write_record(handle, metadata)
        if args.profile:
            shape = parse_shapes(args.pipeline_shapes)[0]
            dtype = torch.float16
            host_inputs = [torch.full(shape, value, dtype=dtype, pin_memory=True) for value in (-0.75, 1.25)]
            device_inputs = [torch.empty(shape, dtype=dtype, device="cuda") for _ in host_inputs]
            streams = [torch.cuda.Stream() for _ in host_inputs]
            torch.cuda.nvtx.range_push("serialized_baseline")
            baseline_ms = run_pipeline("baseline", baseline_library, host_inputs, device_inputs, streams)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("current_stream_candidate")
            candidate_ms = run_pipeline("candidate", baseline_library, host_inputs, device_inputs, streams)
            torch.cuda.nvtx.range_pop()
            write_record(
                handle,
                {
                    "record_type": "profile",
                    "shape": list(shape),
                    "baseline_ms": baseline_ms,
                    "candidate_ms": candidate_ms,
                },
            )
            return

        sizes = [int(value) for value in args.sizes.split(",")]
        for dtype_name in DTYPES:
            for quant_type in QUANT_TYPES:
                for n in sizes:
                    repetitions = direct_repetitions(n, args.direct_repetitions)
                    result = benchmark_direct_case(
                        baseline_library,
                        bnb.cextension.lib._lib,
                        dtype_name,
                        quant_type,
                        n,
                        64,
                        args.warmup,
                        args.rounds,
                        repetitions,
                    )
                    write_record(handle, result)

        for dtype_name in ("fp16", "bf16"):
            for shape in parse_shapes(args.pipeline_shapes):
                result = benchmark_pipeline_case(
                    baseline_library,
                    dtype_name,
                    shape,
                    args.warmup,
                    args.rounds,
                    args.pipeline_repetitions,
                )
                write_record(handle, result)


if __name__ == "__main__":
    main()
