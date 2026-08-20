#!/usr/bin/env python3
"""Compare blockwise-dequant implementations from two Git revisions on one GPU."""

import argparse
import ctypes as ct
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

import torch

BASELINE = "baseline"
CANDIDATE = "candidate"
VARIANTS = (BASELINE, CANDIDATE)
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
DTYPE_NAMES = {value: key for key, value in DTYPES.items()}
FORMATS = ("general8", "nf4", "fp4")
ARGTYPES = [ct.c_void_p] * 4 + [ct.c_int32, ct.c_int32, ct.c_void_p]
DEFAULT_SHAPES = (
    "4096x4096",
    "4096x11008",
    "11008x4096",
    "4096x14336",
    "14336x4096",
    "7168x7168",
    "8192x8192",
)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def comma_list(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_shape(value):
    parts = value.lower().split("x")
    if len(parts) != 2 or any(not part.isdigit() or int(part) <= 0 for part in parts):
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}; expected ROWSxCOLS")
    return int(parts[0]), int(parts[1])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--candidate-ref", default="HEAD")
    parser.add_argument("--compute-capability", default="75;80;86;89;90;100;120")
    parser.add_argument("--formats", type=comma_list, default=FORMATS)
    parser.add_argument("--dtypes", type=comma_list, default=tuple(DTYPES))
    parser.add_argument("--shapes", type=comma_list, default=DEFAULT_SHAPES)
    parser.add_argument("--tail-counts", type=comma_list, default=("511", "512", "513", "1023", "1024", "1025"))
    parser.add_argument("--general8-blocksizes", type=comma_list, default=("64", "256", "4096"))
    parser.add_argument("--fourbit-blocksizes", type=comma_list, default=("32", "64", "128", "256"))
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--repetitions", type=positive_int, default=100)
    parser.add_argument("--rounds", type=positive_int, default=7)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--include-user-controls", action="store_true")
    parser.add_argument("--require-sm103", action="store_true")
    return parser.parse_args()


def run(command, *, cwd=None, stdin=None, capture_output=False):
    print("command=" + " ".join(str(part) for part in command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        stdin=stdin,
        check=True,
        capture_output=capture_output,
        text=capture_output,
    )


def require_device(require_sm103):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the benchmark requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    if require_sm103 and (capability != (10, 3) or "B300" not in properties.name.upper()):
        raise RuntimeError(f"expected B300/SM103, found {properties.name} with capability {capability}")
    return properties


def extract_source(repo_root, destination, revision):
    destination.mkdir(parents=True)
    archive = subprocess.Popen(["git", "archive", revision], cwd=repo_root, stdout=subprocess.PIPE)
    try:
        run(["tar", "-x", "-C", destination], stdin=archive.stdout)
    finally:
        if archive.stdout is not None:
            archive.stdout.close()
    if archive.wait() != 0:
        raise subprocess.CalledProcessError(archive.returncode, ["git", "archive", revision])


def build_library(source_root, build_root, compute_capability):
    started = time.perf_counter()
    run(
        [
            "cmake",
            "-G",
            "Ninja",
            "-DCOMPUTE_BACKEND=cuda",
            f"-DCOMPUTE_CAPABILITY={compute_capability}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-S",
            source_root,
            "-B",
            build_root,
        ]
    )
    parallel = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    run(["cmake", "--build", build_root, "--parallel", str(parallel)])
    return time.perf_counter() - started


def prepare_libraries(repo_root, build_root, revisions, compute_capability):
    if build_root.exists():
        raise FileExistsError(f"build root must not exist: {build_root}")
    build_root.mkdir(parents=True)
    sources = {}
    build_seconds = {}
    for variant, revision in revisions.items():
        source = build_root / f"source-{variant}"
        extract_source(repo_root, source, revision)
        build_seconds[variant] = build_library(source, build_root / f"build-{variant}", compute_capability)
        sources[variant] = source
    cuda_version = "".join(torch.version.cuda.split(".")[:2])
    libraries = {
        variant: source / "bitsandbytes" / f"libbitsandbytes_cuda{cuda_version}.so"
        for variant, source in sources.items()
    }
    for path in libraries.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return sources, libraries, build_seconds


def symbol_name(dtype_name, quant_type):
    suffix = "" if quant_type == "general8" else f"_{quant_type}"
    return f"cdequantize_blockwise_{dtype_name}{suffix}"


def load_library(path):
    library = ct.CDLL(str(path.resolve()), mode=ct.RTLD_LOCAL)
    for dtype_name in DTYPES:
        for quant_type in FORMATS:
            function = getattr(library, symbol_name(dtype_name, quant_type))
            function.argtypes = ARGTYPES
            function.restype = None
    return library


def invoke(library, code, packed, absmax, output, blocksize, quant_type):
    function = getattr(library, symbol_name(DTYPE_NAMES[output.dtype], quant_type))
    function(
        code.data_ptr() if code is not None else None,
        packed.data_ptr(),
        absmax.data_ptr(),
        output.data_ptr(),
        blocksize,
        output.numel(),
        torch._C._cuda_getCurrentRawStream(output.device.index),
    )
    return output


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values):
    return {
        "median_ms": statistics.median(values),
        "p10_ms": percentile(values, 0.10),
        "p90_ms": percentile(values, 0.90),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def repetitions_for(numel, requested):
    if numel <= 65536:
        return requested
    if numel <= 1048576:
        return min(requested, 60)
    if numel <= 16777216:
        return min(requested, 20)
    if numel <= 67108864:
        return min(requested, 10)
    return min(requested, 4)


def measure(functions, warmup, repetitions, rounds):
    names = tuple(functions)
    samples = {name: [] for name in names}
    round_medians = {name: [] for name in names}
    for round_index in range(rounds):
        for index in range(warmup):
            for offset in range(len(names)):
                functions[names[(index + round_index + offset) % len(names)]]()
        torch.cuda.synchronize()
        events = {name: [] for name in names}
        for index in range(repetitions):
            for offset in range(len(names)):
                name = names[(index + round_index + offset) % len(names)]
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                functions[name]()
                end.record()
                events[name].append((start, end))
        torch.cuda.synchronize()
        for name in names:
            values = [start.elapsed_time(end) for start, end in events[name]]
            samples[name].extend(values)
            round_medians[name].append(statistics.median(values))
    return {name: summarize(values) for name, values in samples.items()}, round_medians, samples


def raw_equal(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.contiguous().view(torch.uint8),
            right.contiguous().view(torch.uint8),
        )
    )


def make_inputs(numel, dtype, quant_type, blocksize, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    packed_numel = numel if quant_type == "general8" else (numel + 1) // 2
    packed = torch.randint(0, 256, (packed_numel,), dtype=torch.uint8, device="cuda", generator=generator)
    absmax = torch.rand(
        (numel + blocksize - 1) // blocksize,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    code = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32, device="cuda") if quant_type == "general8" else None
    outputs = {name: torch.empty(numel, dtype=dtype, device="cuda") for name in VARIANTS}
    return code, packed, absmax, outputs


def direct_record(args, libraries, shape, dtype_name, quant_type, blocksize, case):
    dtype = DTYPES[dtype_name]
    numel = math.prod(shape)
    code, packed, absmax, outputs = make_inputs(
        numel,
        dtype,
        quant_type,
        blocksize,
        args.seed + numel % 1000003 + blocksize,
    )
    packed_before = packed.clone()
    absmax_before = absmax.clone()
    code_before = code.clone() if code is not None else None
    functions = {
        name: lambda name=name: invoke(libraries[name], code, packed, absmax, outputs[name], blocksize, quant_type)
        for name in VARIANTS
    }
    for _ in range(3):
        for function in functions.values():
            function()
    torch.cuda.synchronize()
    if not raw_equal(outputs[BASELINE], outputs[CANDIDATE]):
        raise AssertionError(f"baseline and candidate differ for {case}/{dtype_name}/{quant_type}")
    if not raw_equal(packed, packed_before) or not raw_equal(absmax, absmax_before):
        raise AssertionError("dequantization changed packed input or absmax")
    if code is not None and not raw_equal(code, code_before):
        raise AssertionError("dequantization changed code")
    repetitions = repetitions_for(numel, args.repetitions)
    stats, round_medians, samples = measure(functions, args.warmup, repetitions, args.rounds)
    ratio = stats[BASELINE]["median_ms"] / stats[CANDIDATE]["median_ms"]
    return {
        "type": "direct_dequant",
        "case": case,
        "shape": list(shape),
        "numel": numel,
        "dtype": dtype_name,
        "quant_type": quant_type,
        "blocksize": blocksize,
        "repetitions": repetitions,
        "stats": stats,
        "baseline_over_candidate": ratio,
        "round_ratios": [
            baseline / candidate for baseline, candidate in zip(round_medians[BASELINE], round_medians[CANDIDATE])
        ],
        "round_medians_ms": round_medians,
        "samples_ms": samples,
        "raw_equal": True,
        "inputs_unchanged": True,
    }


def user_record(args, libraries, bnb, functional, cuda_ops, control, shape, dtype_name, quant_type):
    dtype = DTYPES[dtype_name]
    source = torch.randn(shape, device="cuda", dtype=dtype)
    packed, state = functional.quantize_4bit(
        source,
        blocksize=64,
        quant_type=quant_type,
        compress_statistics=control == "nested",
    )
    original = cuda_ops.lib
    if control == "nested":
        outputs = {name: torch.empty(shape, device="cuda", dtype=dtype) for name in VARIANTS}

        def call(name):
            cuda_ops.lib = libraries[name]
            return functional.dequantize_4bit(packed, state, out=outputs[name])

    else:
        activation = torch.randn((6, shape[1]), device="cuda", dtype=dtype, requires_grad=True)
        forward = bnb.matmul_4bit(activation, packed, state)
        grad_output = torch.randn_like(forward)

        def call(name):
            cuda_ops.lib = libraries[name]
            return torch.autograd.grad(forward, activation, grad_output, retain_graph=True)[0]

    functions = {name: lambda name=name: call(name) for name in VARIANTS}
    try:
        outputs = {name: function() for name, function in functions.items()}
        torch.cuda.synchronize()
        if not raw_equal(outputs[BASELINE], outputs[CANDIDATE]):
            raise AssertionError(f"baseline and candidate differ for {control}")
        repetitions = repetitions_for(math.prod(shape), args.repetitions)
        stats, round_medians, samples = measure(functions, args.warmup, repetitions, args.rounds)
    finally:
        cuda_ops.lib = original
    return {
        "type": "nested_4bit_dequant" if control == "nested" else "matmul4bit_backward",
        "shape": list(shape),
        "dtype": dtype_name,
        "quant_type": quant_type,
        "repetitions": repetitions,
        "stats": stats,
        "baseline_over_candidate": stats[BASELINE]["median_ms"] / stats[CANDIDATE]["median_ms"],
        "round_ratios": [
            baseline / candidate for baseline, candidate in zip(round_medians[BASELINE], round_medians[CANDIDATE])
        ],
        "round_medians_ms": round_medians,
        "samples_ms": samples,
        "raw_equal": True,
    }


def write_record(output, record):
    output.write(json.dumps(record, sort_keys=True) + "\n")
    output.flush()


def main():
    args = parse_args()
    properties = require_device(args.require_sm103)
    unknown_formats = set(args.formats) - set(FORMATS)
    unknown_dtypes = set(args.dtypes) - set(DTYPES)
    if unknown_formats or unknown_dtypes:
        raise ValueError(f"unknown formats={sorted(unknown_formats)}, dtypes={sorted(unknown_dtypes)}")
    shapes = tuple(parse_shape(value) for value in args.shapes)
    tails = tuple(positive_int(value) for value in args.tail_counts)
    general8_blocks = tuple(positive_int(value) for value in args.general8_blocksizes)
    fourbit_blocks = tuple(positive_int(value) for value in args.fourbit_blocksizes)

    repo_root = Path(__file__).resolve().parents[1]
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        capture_output=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"tracked worktree must be clean:\n{status}")
    revisions = {BASELINE: args.baseline_ref, CANDIDATE: args.candidate_ref}
    resolved = {
        name: run(["git", "rev-parse", revision], cwd=repo_root, capture_output=True).stdout.strip()
        for name, revision in revisions.items()
    }
    sources, paths, build_seconds = prepare_libraries(
        repo_root,
        args.build_root.resolve(),
        revisions,
        args.compute_capability,
    )
    sys.path.insert(0, str(sources[CANDIDATE]))
    import bitsandbytes as bnb
    from bitsandbytes import functional
    from bitsandbytes.backends.cuda import ops as cuda_ops

    loaded = Path(bnb.cextension.lib._lib._name).resolve()
    if loaded != paths[CANDIDATE].resolve():
        raise RuntimeError(f"expected candidate library {paths[CANDIDATE]}, loaded {loaded}")
    libraries = {name: load_library(path) for name, path in paths.items()}
    entrypoints = {
        name: ct.cast(library.cdequantize_blockwise_fp16_nf4, ct.c_void_p).value for name, library in libraries.items()
    }
    if len(set(entrypoints.values())) != len(entrypoints):
        raise RuntimeError(f"isolated libraries share entrypoints: {entrypoints}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "metadata",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "num_sms": properties.multi_processor_count,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "revisions": resolved,
        "compute_capability_input": args.compute_capability,
        "warmup": args.warmup,
        "repetitions_max": args.repetitions,
        "rounds": args.rounds,
        "build_seconds": build_seconds,
        "library_bytes": {name: path.stat().st_size for name, path in paths.items()},
        "libraries": {name: str(path) for name, path in paths.items()},
        "loaded_candidate": str(loaded),
        "entrypoints": entrypoints,
    }
    with args.output.open("w", encoding="utf-8") as output:
        write_record(output, metadata)
        for quant_type in args.formats:
            primary_blocksize = 256 if quant_type == "general8" else 64
            controls = general8_blocks if quant_type == "general8" else fourbit_blocks
            for dtype_name in args.dtypes:
                for shape in shapes:
                    write_record(
                        output,
                        direct_record(args, libraries, shape, dtype_name, quant_type, primary_blocksize, "realistic"),
                    )
                for numel in tails:
                    write_record(
                        output,
                        direct_record(args, libraries, (numel,), dtype_name, quant_type, primary_blocksize, "tail"),
                    )
                for blocksize in controls:
                    write_record(
                        output,
                        direct_record(
                            args,
                            libraries,
                            (1048576,),
                            dtype_name,
                            quant_type,
                            blocksize,
                            "blocksize_control",
                        ),
                    )
        if args.include_user_controls:
            for shape in ((4096, 14336), (14336, 4096)):
                for dtype_name in ("fp16", "bf16"):
                    for quant_type in ("nf4", "fp4"):
                        for control in ("nested", "backward"):
                            write_record(
                                output,
                                user_record(
                                    args,
                                    libraries,
                                    bnb,
                                    functional,
                                    cuda_ops,
                                    control,
                                    shape,
                                    dtype_name,
                                    quant_type,
                                ),
                            )


if __name__ == "__main__":
    main()
