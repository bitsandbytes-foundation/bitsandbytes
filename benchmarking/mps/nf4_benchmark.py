#!/usr/bin/env python3
"""Measure NF4 quantization and Linear4bit latency on Apple MPS."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import platform
import statistics
import time
from typing import TypeVar

import torch

import bitsandbytes as bnb

T = TypeVar("T")


def timed_ms(fn: Callable[[], T], repeats: int) -> tuple[float, T]:
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1_000)
    assert result is not None
    return statistics.median(samples), result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=4_194_304, help="Number of fp32 values to quantize")
    parser.add_argument("--hidden", type=int, default=2048, help="Input and output size of the linear layers")
    parser.add_argument("--repeats", type=int, default=11, help="Number of timed samples")
    parser.add_argument("--warmups", type=int, default=5, help="Number of untimed warm-up iterations")
    args = parser.parse_args()
    if min(args.elements, args.hidden, args.repeats) < 1 or args.warmups < 0:
        parser.error("elements, hidden, and repeats must be positive; warmups must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("Apple MPS is not available")

    device = torch.device("mps")
    torch.manual_seed(0)
    source = torch.randn(args.elements, device=device, dtype=torch.float32)

    for _ in range(args.warmups):
        packed, state = bnb.functional.quantize_4bit(source, blocksize=64, quant_type="nf4")
        bnb.functional.dequantize_4bit(packed, state)
    torch.mps.synchronize()

    quant_ms, (packed, state) = timed_ms(
        lambda: bnb.functional.quantize_4bit(source, blocksize=64, quant_type="nf4"),
        args.repeats,
    )
    dequant_ms, restored = timed_ms(
        lambda: bnb.functional.dequantize_4bit(packed, state),
        args.repeats,
    )
    mae = (source - restored).abs().mean().item()
    max_error = (source - restored).abs().max().item()

    weight = torch.randn(args.hidden, args.hidden, dtype=torch.float16)
    dense = torch.nn.Linear(args.hidden, args.hidden, bias=False, dtype=torch.float16)
    dense.weight.data.copy_(weight)
    dense = dense.to(device)

    quantized = bnb.nn.Linear4bit(
        args.hidden,
        args.hidden,
        bias=False,
        compute_dtype=torch.float16,
        quant_type="nf4",
    )
    quantized.weight.data.copy_(weight)
    quantized = quantized.to(device)

    activation = torch.randn(1, args.hidden, device=device, dtype=torch.float16)
    for _ in range(args.warmups):
        dense(activation)
        quantized(activation)
    torch.mps.synchronize()
    dense_ms, dense_out = timed_ms(lambda: dense(activation), args.repeats)
    nf4_ms, nf4_out = timed_ms(lambda: quantized(activation), args.repeats)
    output_mae = (dense_out - nf4_out).abs().mean().item()

    source_bytes = source.numel() * source.element_size()
    packed_bytes = packed.numel() * packed.element_size()
    absmax_bytes = state.absmax.numel() * state.absmax.element_size()

    print(f"device: Apple MPS ({platform.machine()})")
    print(f"torch: {torch.__version__}; bitsandbytes: {bnb.__version__}")
    print(
        f"NF4 packed+absmax storage: {(packed_bytes + absmax_bytes) / source_bytes:.3f}x "
        f"of fp32 ({source_bytes / 2**20:.1f} MiB -> {(packed_bytes + absmax_bytes) / 2**20:.1f} MiB)"
    )
    print(f"quantize median: {quant_ms:.3f} ms")
    print(f"dequantize median: {dequant_ms:.3f} ms")
    print(f"round-trip MAE: {mae:.6f}; max error: {max_error:.6f}")
    print(f"dense fp16 GEMV median: {dense_ms:.3f} ms")
    print(f"NF4 Linear4bit median: {nf4_ms:.3f} ms")
    print(f"dense-vs-NF4 output MAE: {output_mae:.6f}")


if __name__ == "__main__":
    main()
