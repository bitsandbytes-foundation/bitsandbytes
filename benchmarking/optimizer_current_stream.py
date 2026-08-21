#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

import bitsandbytes as bnb
from bitsandbytes.backends.cuda import ops as cuda_ops
from bitsandbytes.cextension import lib
from bitsandbytes.utils import sync_gpu

OPTIMIZER_DTYPES_32 = {
    "adam": ("fp32", "fp16", "bf16"),
    "momentum": ("32", "16"),
    "rmsprop": ("32", "16"),
    "lion": ("fp32", "fp16", "bf16"),
    "adagrad": ("32", "16"),
    "ademamix": ("fp32", "fp16", "bf16"),
}
OPTIMIZER_NAMES_8 = ("adam", "momentum", "rmsprop", "lion", "adagrad", "ademamix")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


class LegacyFunction:
    def __init__(self, function, stream_function):
        function.argtypes = stream_function.argtypes[:-1]
        function.restype = stream_function.restype
        self.function = function

    def __call__(self, *args):
        return self.function(*args[:-1])


def parse_csv(value, cast=str):
    return [cast(item) for item in value.split(",") if item]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare legacy and current-stream non-paged optimizer updates")
    parser.add_argument("--expected-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--inventories", default="32,256,1024")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--bits", default="8,32")
    parser.add_argument("--single-large-numel", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--peft-layers", type=int, default=32)
    parser.add_argument("--peft-hidden", type=int, default=512)
    parser.add_argument("--peft-rank", type=int, default=8)
    parser.add_argument("--peft-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values):
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
        "samples": values,
    }


def build_symbol_maps():
    stream32 = cuda_ops.str2optimizer32bit.copy()
    stream8 = cuda_ops.str2optimizer8bit_blockwise.copy()
    legacy32 = {}
    for name, dtypes in OPTIMIZER_DTYPES_32.items():
        functions = []
        for dtype in dtypes:
            legacy = getattr(lib, f"c{name}32bit_grad_{dtype}")
            current = getattr(lib, f"c{name}32bit_grad_{dtype}_with_stream")
            functions.append(LegacyFunction(legacy, current))
        legacy32[name] = tuple(functions)
    legacy32["lamb"] = legacy32["adam"]
    legacy32["lars"] = legacy32["momentum"]

    legacy8 = {}
    for name in OPTIMIZER_NAMES_8:
        functions = []
        for dtype in ("fp32", "fp16", "bf16"):
            legacy = getattr(lib, f"c{name}_8bit_blockwise_grad_{dtype}")
            current = getattr(lib, f"c{name}_8bit_blockwise_grad_{dtype}_with_stream")
            functions.append(LegacyFunction(legacy, current))
        legacy8[name] = tuple(functions)
    return stream32, stream8, legacy32, legacy8


@torch.no_grad()
def legacy_step(optimizer):
    if not optimizer.initialized:
        optimizer.check_overrides()
        optimizer.to_gpu()
        optimizer.initialized = True
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            if parameter.grad is None:
                continue
            state = optimizer.state[parameter]
            if not state:
                optimizer.init_state(group, parameter, group_index, parameter_index)
            optimizer.prefetch_state(parameter)
            optimizer.update_step(group, parameter, group_index, parameter_index)
            sync_gpu(parameter)


def activate_maps(maps, variant):
    stream32, stream8, legacy32, legacy8 = maps
    if variant == "legacy":
        cuda_ops.str2optimizer32bit = legacy32
        cuda_ops.str2optimizer8bit_blockwise = legacy8
    else:
        cuda_ops.str2optimizer32bit = stream32
        cuda_ops.str2optimizer8bit_blockwise = stream8


def completed_measure(label, function):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(label)
    started = time.perf_counter()
    start.record()
    function()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - started) * 1e3
    event_ms = start.elapsed_time(end)
    torch.cuda.nvtx.range_pop()
    return event_ms, wall_ms


def make_optimizer(parameters, state_bits):
    if state_bits == 8:
        return bnb.optim.AdamW8bit(parameters, lr=1e-3, min_8bit_size=4096)
    return bnb.optim.AdamW32bit(parameters, lr=1e-3)


def make_parameter_pair(sizes, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    baseline = []
    candidate = []
    for size in sizes:
        value = torch.randn(size, dtype=dtype, device="cuda", generator=generator) * 0.01
        gradient = torch.randn(size, dtype=dtype, device="cuda", generator=generator) * 0.001
        baseline_parameter = torch.nn.Parameter(value.clone())
        candidate_parameter = torch.nn.Parameter(value.clone())
        baseline_parameter.grad = gradient.clone()
        candidate_parameter.grad = gradient.clone()
        baseline.append(baseline_parameter)
        candidate.append(candidate_parameter)
    return baseline, candidate


def assert_optimizer_equal(baseline_optimizer, candidate_optimizer, baseline_params, candidate_params):
    for baseline_parameter, candidate_parameter in zip(baseline_params, candidate_params):
        if not torch.equal(baseline_parameter, candidate_parameter):
            raise AssertionError("parameter mismatch between legacy and current-stream variants")
        if not torch.equal(baseline_parameter.grad, candidate_parameter.grad):
            raise AssertionError("gradient changed or mismatched")
        baseline_state = baseline_optimizer.state[baseline_parameter]
        candidate_state = candidate_optimizer.state[candidate_parameter]
        if baseline_state.keys() != candidate_state.keys():
            raise AssertionError("optimizer state keys differ")
        for key in baseline_state:
            left = baseline_state[key]
            right = candidate_state[key]
            if isinstance(left, torch.Tensor):
                if not torch.equal(left, right):
                    raise AssertionError(f"optimizer state mismatch for {key}")
            elif left != right:
                raise AssertionError(f"optimizer metadata mismatch for {key}")


def optimizer_state_bytes(optimizer):
    seen = set()
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if not isinstance(value, torch.Tensor) or value.data_ptr() in seen:
                continue
            seen.add(value.data_ptr())
            total += value.numel() * value.element_size()
    return total


def run_inventory(maps, sizes, dtype_name, state_bits, warmups, rounds, seed, label):
    dtype = DTYPES[dtype_name]
    baseline_params, candidate_params = make_parameter_pair(sizes, dtype, seed)
    baseline_optimizer = make_optimizer(baseline_params, state_bits)
    candidate_optimizer = make_optimizer(candidate_params, state_bits)

    def run(variant):
        activate_maps(maps, variant)
        if variant == "legacy":
            legacy_step(baseline_optimizer)
        else:
            candidate_optimizer.step()

    for index in range(warmups):
        order = ("legacy", "current_stream") if index % 2 == 0 else ("current_stream", "legacy")
        for variant in order:
            completed_measure(f"warmup_{label}_{variant}", lambda variant=variant: run(variant))

    event_samples = {"legacy": [], "current_stream": []}
    wall_samples = {"legacy": [], "current_stream": []}
    for round_index in range(rounds):
        order = ("legacy", "current_stream") if round_index % 2 == 0 else ("current_stream", "legacy")
        for variant in order:
            event_ms, wall_ms = completed_measure(f"timed_{label}_{variant}", lambda variant=variant: run(variant))
            event_samples[variant].append(event_ms)
            wall_samples[variant].append(wall_ms)

    torch.cuda.synchronize()
    assert_optimizer_equal(baseline_optimizer, candidate_optimizer, baseline_params, candidate_params)
    legacy_event = summarize(event_samples["legacy"])
    current_event = summarize(event_samples["current_stream"])
    legacy_wall = summarize(wall_samples["legacy"])
    current_wall = summarize(wall_samples["current_stream"])
    return {
        "type": "inventory",
        "label": label,
        "optimizer": "AdamW",
        "state_bits": state_bits,
        "dtype": dtype_name,
        "parameter_count": len(sizes),
        "total_numel": sum(sizes),
        "min_numel": min(sizes),
        "max_numel": max(sizes),
        "warmups": warmups,
        "rounds": rounds,
        "bitwise_equal": True,
        "state_bytes": optimizer_state_bytes(candidate_optimizer),
        "legacy_event_ms": legacy_event,
        "current_stream_event_ms": current_event,
        "legacy_wall_ms": legacy_wall,
        "current_stream_wall_ms": current_wall,
        "event_ratio": legacy_event["median"] / current_event["median"],
        "wall_ratio": legacy_wall["median"] / current_wall["median"],
    }


class AdapterStack(torch.nn.Module):
    def __init__(self, layers, hidden, rank, dtype):
        super().__init__()
        self.left = torch.nn.ParameterList()
        self.right = torch.nn.ParameterList()
        for _ in range(layers):
            self.left.append(torch.nn.Parameter(torch.randn(hidden, rank, device="cuda", dtype=dtype) * 0.01))
            self.right.append(torch.nn.Parameter(torch.randn(rank, hidden, device="cuda", dtype=dtype) * 0.01))

    def forward(self, value):
        for left, right in zip(self.left, self.right):
            value = value + (value @ left) @ right
        return value


def run_peft(maps, layers, hidden, rank, batch, warmups, rounds, seed):
    torch.manual_seed(seed)
    baseline_model = AdapterStack(layers, hidden, rank, torch.float16)
    candidate_model = AdapterStack(layers, hidden, rank, torch.float16)
    candidate_model.load_state_dict(baseline_model.state_dict())
    baseline_optimizer = bnb.optim.AdamW8bit(baseline_model.parameters(), lr=1e-3, min_8bit_size=4096)
    candidate_optimizer = bnb.optim.AdamW8bit(candidate_model.parameters(), lr=1e-3, min_8bit_size=4096)
    inputs = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)

    def run(variant):
        if variant == "legacy":
            activate_maps(maps, variant)
            model = baseline_model
            optimizer = baseline_optimizer
        else:
            activate_maps(maps, variant)
            model = candidate_model
            optimizer = candidate_optimizer
        optimizer.zero_grad(set_to_none=True)
        model(inputs).float().square().mean().backward()
        if variant == "legacy":
            legacy_step(optimizer)
        else:
            optimizer.step()

    for index in range(warmups):
        order = ("legacy", "current_stream") if index % 2 == 0 else ("current_stream", "legacy")
        for variant in order:
            completed_measure(f"warmup_peft_{variant}", lambda variant=variant: run(variant))

    event_samples = {"legacy": [], "current_stream": []}
    wall_samples = {"legacy": [], "current_stream": []}
    for round_index in range(rounds):
        order = ("legacy", "current_stream") if round_index % 2 == 0 else ("current_stream", "legacy")
        for variant in order:
            event_ms, wall_ms = completed_measure(f"timed_peft_{variant}", lambda variant=variant: run(variant))
            event_samples[variant].append(event_ms)
            wall_samples[variant].append(wall_ms)

    torch.cuda.synchronize()
    assert_optimizer_equal(
        baseline_optimizer,
        candidate_optimizer,
        list(baseline_model.parameters()),
        list(candidate_model.parameters()),
    )
    legacy_event = summarize(event_samples["legacy"])
    current_event = summarize(event_samples["current_stream"])
    legacy_wall = summarize(wall_samples["legacy"])
    current_wall = summarize(wall_samples["current_stream"])
    return {
        "type": "peft",
        "optimizer": "AdamW8bit",
        "dtype": "fp16",
        "layers": layers,
        "hidden": hidden,
        "rank": rank,
        "batch": batch,
        "parameter_count": sum(1 for _ in baseline_model.parameters()),
        "warmups": warmups,
        "rounds": rounds,
        "bitwise_equal": True,
        "legacy_event_ms": legacy_event,
        "current_stream_event_ms": current_event,
        "legacy_wall_ms": legacy_wall,
        "current_stream_wall_ms": current_wall,
        "event_ratio": legacy_event["median"] / current_event["median"],
        "wall_ratio": legacy_wall["median"] / current_wall["median"],
    }


def main():
    args = parse_args()
    if args.warmups < 0 or args.rounds < 1:
        raise ValueError("warmups must be nonnegative and rounds must be positive")
    expected_library = args.expected_library.resolve()
    loaded_library = Path(lib._lib._name).resolve()
    if loaded_library != expected_library:
        raise AssertionError(f"loaded {loaded_library}, expected {expected_library}")
    props = torch.cuda.get_device_properties(0)
    if "B300" not in props.name or (props.major, props.minor) != (10, 3):
        raise AssertionError(f"expected B300/SM103, got {props.name} CC {props.major}.{props.minor}")

    maps = build_symbol_maps()
    metadata = {
        "type": "metadata",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "loaded_library": str(loaded_library),
        "warmups": args.warmups,
        "rounds": args.rounds,
    }
    records = [metadata]
    pattern = (4096, 8192, 16384, 32768)
    inventories = parse_csv(args.inventories, int)
    dtypes = parse_csv(args.dtypes)
    state_bits = parse_csv(args.bits, int)
    for count in inventories:
        sizes = [pattern[index % len(pattern)] for index in range(count)]
        for dtype_name in dtypes:
            for bits in state_bits:
                records.append(
                    run_inventory(
                        maps,
                        sizes,
                        dtype_name,
                        bits,
                        args.warmups,
                        args.rounds,
                        args.seed + count + bits,
                        f"inventory_{count}",
                    )
                )

    for dtype_name in dtypes:
        for bits in state_bits:
            records.append(
                run_inventory(
                    maps,
                    [args.single_large_numel],
                    dtype_name,
                    bits,
                    args.warmups,
                    args.rounds,
                    args.seed + bits,
                    "single_large",
                )
            )

    records.append(
        run_peft(
            maps,
            args.peft_layers,
            args.peft_hidden,
            args.peft_rank,
            args.peft_batch,
            args.warmups,
            args.rounds,
            args.seed,
        )
    )
    activate_maps(maps, "current_stream")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for record in records:
            line = json.dumps(record, sort_keys=True)
            output.write(line + "\n")
            print(line)


if __name__ == "__main__":
    main()
