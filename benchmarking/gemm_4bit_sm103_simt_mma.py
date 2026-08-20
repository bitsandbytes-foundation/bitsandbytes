#!/usr/bin/env python3
"""Benchmark SM103 fused 4-bit GEMM SIMT, MMA, and automatic dispatch.

This CLI builds three SM103-compatible libraries from the current commit. The
SIMT and MMA forcing edits are applied only to temporary source copies.
Measurements rotate variant order for every sample and are written as JSONL.
"""

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

import torch

CALIBRATION_CASES = (
    ("wave1_below", 9408, 4096, (3, 4, 5, 6, 7, 8)),
    ("wave1_at", 9472, 4096, (3, 4, 5, 6, 7, 8)),
    ("wave1_above", 9536, 4096, (3, 4, 5, 6, 7, 8)),
    ("wide_11008x4096", 11008, 4096, (3, 4, 5, 6, 7, 8)),
    ("wide_14336x4096", 14336, 4096, (3, 4, 5, 6, 7, 8)),
    ("wave2_below", 18880, 4096, (3, 4, 5, 6, 7, 8)),
    ("wave2_at", 18944, 4096, (3, 4, 5, 6, 7, 8)),
    ("wave2_above", 19008, 4096, (3, 4, 5, 6, 7, 8)),
    ("wave3_below_tall", 28352, 32768, (3, 4, 5, 6, 7, 8)),
    ("wave3_at_tall", 28416, 32768, (3, 4, 5, 6, 7, 8)),
    ("wave3_above_tall", 28480, 32768, (3, 4, 5, 6, 7, 8)),
    ("wave4_tall_k49152", 37888, 49152, (3, 4, 5, 6, 7, 8)),
    ("wave8_tall_k81920", 75776, 81920, (3, 4, 5, 6, 7, 8)),
    ("vocab_128256x4096", 128256, 4096, (3, 4, 5, 6, 7, 8)),
)
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
VARIANTS = ("simt", "mma", "automatic")
GEMM_ARGTYPES = [ct.c_void_p] * 8 + [ct.c_int32] * 5 + [ct.c_void_p]

EARLY_SIMT = """    // fp32 and M<=3 are always SIMT regardless of GPU -- skip the props lookup.
    if (is_fp32 || M <= 3) {"""
FORCED_MMA_EARLY = """    // fp32 has no MMA implementation.
    if (is_fp32) {"""
AUTOMATIC_DISPATCH = """    const bool use_simt = (M == 4 && highbw_gddr) || undersubscribed || wide_n_simt || tall_k_simt ||
                          sm103_tall_k_simt || (M <= 16 && mma_blocks * 4 <= num_sms) ||
                          (M <= 32 && mma_blocks * 8 <= num_sms) || (K % 64 != 0); // MMA requirement"""


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_case(value):
    try:
        name, n_text, k_text, m_text = value.split(":")
        n = positive_int(n_text)
        k = positive_int(k_text)
        m_values = tuple(positive_int(item) for item in m_text.split(","))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("case must be NAME:N:K:M1,M2,...") from error
    if not name or not m_values:
        raise argparse.ArgumentTypeError("case name and M values must not be empty")
    return name, n, k, m_values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument(
        "--compute-capability",
        default="103",
        help="CMake COMPUTE_CAPABILITY target list; defaults to a native SM103 build",
    )
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--repetitions", type=positive_int, default=100)
    parser.add_argument("--rounds", type=positive_int, default=7)
    parser.add_argument("--case", action="append", type=parse_case, help="NAME:N:K:M1,M2,...; overrides the grid")
    parser.add_argument("--dtype", action="append", choices=tuple(DTYPES))
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def run(command, *, cwd=None, stdin=None):
    print("command=" + " ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, cwd=cwd, stdin=stdin, check=True)


def require_sm103():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")
    capability = torch.cuda.get_device_capability(0)
    properties = torch.cuda.get_device_properties(0)
    if capability != (10, 3) or "B300" not in properties.name.upper():
        raise RuntimeError(f"expected B300/SM103, found {properties.name} with capability {capability}")
    return properties


def replace_once(path, old, new):
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"expected one dispatch block in {path}, found {source.count(old)}")
    path.write_text(source.replace(old, new), encoding="utf-8")


def extract_source(repo_root, destination):
    destination.mkdir(parents=True)
    archive = subprocess.Popen(["git", "archive", "HEAD"], cwd=repo_root, stdout=subprocess.PIPE)
    try:
        run(["tar", "-x", "-C", destination], stdin=archive.stdout)
    finally:
        if archive.stdout is not None:
            archive.stdout.close()
    if archive.wait() != 0:
        raise subprocess.CalledProcessError(archive.returncode, ["git", "archive", "HEAD"])


def force_variant(source_root, variant):
    dispatch_path = source_root / "csrc" / "gemm_4bit.cu"
    if variant == "simt":
        replace_once(dispatch_path, AUTOMATIC_DISPATCH, "    const bool use_simt = true;")
    elif variant == "mma":
        replace_once(dispatch_path, EARLY_SIMT, FORCED_MMA_EARLY)
        replace_once(
            dispatch_path,
            AUTOMATIC_DISPATCH,
            "    const bool use_simt = K % 64 != 0; // Preserve the MMA alignment requirement.",
        )
    else:
        raise ValueError(variant)


def build_library(source_root, build_root, compute_capability):
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


def prepare_libraries(repo_root, build_root, compute_capability):
    if build_root.exists():
        raise FileExistsError(f"build root must not exist: {build_root}")
    build_root.mkdir(parents=True)
    sources = {}
    for variant in VARIANTS:
        source_root = build_root / f"source-{variant}"
        extract_source(repo_root, source_root)
        if variant != "automatic":
            force_variant(source_root, variant)
        build_library(source_root, build_root / f"build-{variant}", compute_capability)
        sources[variant] = source_root

    cuda_version = "".join(torch.version.cuda.split(".")[:2])
    libraries = {
        variant: source_root / "bitsandbytes" / f"libbitsandbytes_cuda{cuda_version}.so"
        for variant, source_root in sources.items()
    }
    for path in libraries.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return sources, libraries


def load_library(path):
    library = ct.CDLL(str(path.resolve()), mode=ct.RTLD_LOCAL)
    for suffix in ("fp16", "bf16"):
        function = getattr(library, f"cgemm_4bit_{suffix}")
        function.argtypes = GEMM_ARGTYPES
        function.restype = None
    return library


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": percentile(samples, 0.10),
        "p90_ms": percentile(samples, 0.90),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def quantized_weight(functional, n, k, dtype):
    weight = torch.empty((n, k), device="cuda", dtype=dtype)
    weight.normal_(mean=0.0, std=k**-0.5)
    packed, state = functional.quantize_4bit(
        weight,
        blocksize=64,
        quant_type="nf4",
        compress_statistics=True,
    )
    if state.offset.dtype != torch.float32:
        raise RuntimeError(f"expected float32 nested offset, found {state.offset.dtype}")
    reference_weight = functional.dequantize_4bit(packed, state).to(dtype)
    del weight
    return packed, state, reference_weight


def invoke_native(library, activation, packed, state, dtype):
    output = torch.empty((activation.shape[0], state.shape[0]), device=activation.device, dtype=dtype)
    function = getattr(library, "cgemm_4bit_fp16" if dtype == torch.float16 else "cgemm_4bit_bf16")
    function(
        activation.data_ptr(),
        packed.data_ptr(),
        state.state2.absmax.data_ptr(),
        state.absmax.data_ptr(),
        state.state2.code.data_ptr(),
        state.offset.data_ptr(),
        output.data_ptr(),
        None,
        activation.shape[0],
        state.shape[0],
        activation.shape[1],
        state.blocksize,
        2,
        torch._C._cuda_getCurrentRawStream(activation.device.index),
    )
    return output


def invoke_automatic(activation, packed, state):
    return torch.ops.bitsandbytes.gemm_4bit.default(
        activation,
        packed,
        list(state.shape),
        state.state2.absmax,
        state.blocksize,
        state.quant_type,
        absmax_8bit=state.absmax,
        absmax_code=state.state2.code,
        absmax_offset=state.offset,
    )


def collect_interleaved(functions, warmup, repetitions, round_index):
    for index in range(warmup):
        for offset in range(len(VARIANTS)):
            functions[VARIANTS[(index + round_index + offset) % len(VARIANTS)]]()
    torch.cuda.synchronize()

    events = {variant: [] for variant in VARIANTS}
    for index in range(repetitions):
        for offset in range(len(VARIANTS)):
            variant = VARIANTS[(index + round_index + offset) % len(VARIANTS)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[variant]()
            end.record()
            events[variant].append((start, end))
    torch.cuda.synchronize()
    return {variant: [start.elapsed_time(end) for start, end in events[variant]] for variant in VARIANTS}


def automatic_internal_path(m, n, k, num_sms):
    if m <= 3:
        return "simt"
    blocks = ((m + 31) // 32) * ((n + 63) // 64)
    undersubscribed = (m <= 8 and blocks * 3 <= num_sms * 2) or (m == 4 and blocks <= num_sms)
    sm103_tall_k_simt = k > n and m <= 5 and blocks >= num_sms * 3
    use_simt = (
        undersubscribed
        or sm103_tall_k_simt
        or (m <= 16 and blocks * 4 <= num_sms)
        or (m <= 32 and blocks * 8 <= num_sms)
        or k % 64 != 0
    )
    return "simt" if use_simt else "mma"


def benchmark_cell(args, libraries, use_custom, num_sms, case_name, m, n, k, dtype, packed, state, reference_weight):
    import torch.nn.functional as torch_functional

    activation = torch.randn((m, k), device="cuda", dtype=dtype)
    functions = {
        "simt": lambda: invoke_native(libraries["simt"], activation, packed, state, dtype),
        "mma": lambda: invoke_native(libraries["mma"], activation, packed, state, dtype),
        "automatic": lambda: invoke_automatic(activation, packed, state),
    }
    reference = torch_functional.linear(activation, reference_weight)
    correctness = {}
    for variant, function in functions.items():
        result = function()
        torch.cuda.synchronize()
        tolerance = 0.02 if dtype == torch.float16 else 0.08
        torch.testing.assert_close(result, reference, rtol=0.02, atol=tolerance)
        difference = result.float() - reference.float()
        reference_rms = reference.float().square().mean().sqrt().item()
        correctness[variant] = {
            "max_abs_error": difference.abs().max().item(),
            "relative_rms_error": difference.square().mean().sqrt().item() / max(reference_rms, 1e-12),
            "finite": bool(torch.isfinite(result).all().item()),
        }

    samples = {variant: [] for variant in VARIANTS}
    round_medians = {variant: [] for variant in VARIANTS}
    for round_index in range(args.rounds):
        round_samples = collect_interleaved(functions, args.warmup, args.repetitions, round_index)
        for variant in VARIANTS:
            samples[variant].extend(round_samples[variant])
            round_medians[variant].append(statistics.median(round_samples[variant]))

    stats = {variant: summarize(samples[variant]) for variant in VARIANTS}
    simt_over_mma = stats["simt"]["median_ms"] / stats["mma"]["median_ms"]
    return {
        "type": "measurement",
        "case": case_name,
        "m": m,
        "n": n,
        "k": k,
        "n_blocks_64": (n + 63) // 64,
        "dtype": str(dtype).removeprefix("torch."),
        "public_fused": bool(use_custom(0, dtype, m, n, k)),
        "automatic_internal_path": automatic_internal_path(m, n, k, num_sms),
        "stats": stats,
        "simt_over_mma": simt_over_mma,
        "winner_at_5_percent": ("simt" if simt_over_mma < 0.95 else "mma" if simt_over_mma > 1.05 else "noise"),
        "correctness": correctness,
        "round_medians_ms": round_medians,
        "samples_ms": samples,
    }


def main():
    args = parse_args()
    properties = require_sm103()
    repo_root = Path(__file__).resolve().parents[1]
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status:
        raise RuntimeError(f"tracked worktree must be clean:\n{tracked_status}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    source_roots, library_paths = prepare_libraries(
        repo_root,
        args.build_root.resolve(),
        args.compute_capability,
    )

    sys.path.insert(0, str(source_roots["automatic"]))
    import bitsandbytes
    from bitsandbytes import functional
    from bitsandbytes.backends.cuda.ops import _gemm_4bit_use_custom_cuda

    automatic_library_name = getattr(getattr(bitsandbytes.cextension.lib, "_lib", None), "_name", None)
    if not automatic_library_name:
        raise RuntimeError("automatic bitsandbytes native library path is unavailable")
    automatic_library = Path(automatic_library_name).resolve()
    if automatic_library != library_paths["automatic"].resolve():
        raise RuntimeError(f"expected automatic library {library_paths['automatic']}, loaded {automatic_library}")

    libraries = {variant: load_library(library_paths[variant]) for variant in ("simt", "mma")}
    entrypoints = {
        variant: ct.cast(library.cgemm_4bit_fp16, ct.c_void_p).value for variant, library in libraries.items()
    }
    if len(set(entrypoints.values())) != len(entrypoints):
        raise RuntimeError(f"forced libraries resolved to the same entrypoint: {entrypoints}")

    cases = tuple(args.case) if args.case else CALIBRATION_CASES
    dtype_names = tuple(dict.fromkeys(args.dtype or tuple(DTYPES)))
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "metadata",
        "git_commit": commit,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "num_sms": properties.multi_processor_count,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bitsandbytes": bitsandbytes.__version__,
        "automatic_library": str(automatic_library),
        "forced_libraries": {variant: str(library_paths[variant]) for variant in ("simt", "mma")},
        "forced_entrypoints_distinct": True,
        "compute_capability_targets": args.compute_capability,
        "cases": [{"name": name, "n": n, "k": k, "m_values": list(m_values)} for name, n, k, m_values in cases],
        "dtypes": list(dtype_names),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "rounds": args.rounds,
        "blocksize": 64,
        "quant_type": "nf4",
        "compress_statistics": True,
        "seed": args.seed,
    }

    with args.output.open("w", encoding="utf-8") as output:
        line = json.dumps(metadata, sort_keys=True)
        print(line, flush=True)
        output.write(line + "\n")
        for case_index, (case_name, n, k, m_values) in enumerate(cases):
            for dtype_name in dtype_names:
                dtype = DTYPES[dtype_name]
                packed, state, reference_weight = quantized_weight(functional, n, k, dtype)
                for m in m_values:
                    record = benchmark_cell(
                        args,
                        libraries,
                        _gemm_4bit_use_custom_cuda,
                        properties.multi_processor_count,
                        case_name,
                        m,
                        n,
                        k,
                        dtype,
                        packed,
                        state,
                        reference_weight,
                    )
                    line = json.dumps(record, sort_keys=True)
                    print(line, flush=True)
                    output.write(line + "\n")
                    output.flush()
                del packed, state, reference_weight
                torch.cuda.empty_cache()
            print(f"completed_case={case_index + 1}/{len(cases)} name={case_name}", flush=True)


if __name__ == "__main__":
    main()
