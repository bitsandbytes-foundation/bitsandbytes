"""Reference AdamW8bit state layout for two-rank FSDP1 FULL_SHARD.

Run: torchrun --standalone --nproc-per-node=2 examples/fsdp1_original_parameter_state.py --device cpu
Use --device cuda for two CUDA devices. Requires PyTorch and bitsandbytes.

This example retains replicated state for each original parameter and accesses
private FSDP1 metadata. It is a reference for discussing the state-layout
contract, with no checkpoint, AMP, offload, or hybrid-sharding support.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as functional

WORLD_SIZE = 2
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.0
MIN_8BIT_SIZE = 4096


@dataclass(frozen=True)
class FlatBinding:
    flat: Any
    fqns: tuple[str, ...]
    numels: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]
    numels_with_padding: tuple[int, ...]
    padding_mask: tuple[bool, ...]

    @property
    def local(self) -> torch.Tensor:
        value = getattr(self.flat, "_local_shard", None)
        return value if value is not None else self.flat


def _join_fqn(prefix: str, local: str) -> str:
    return f"{prefix}.{local}" if prefix and local else prefix or local


def _flat_bindings(
    model: torch.nn.Module,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> list[FlatBinding]:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    result: list[FlatBinding] = []
    seen: set[str] = set()
    for module in model.modules():
        if not isinstance(module, FSDP):
            continue
        handle = getattr(module, "_handle", None)
        if handle is None:
            continue
        flat = handle.flat_param
        prefix = getattr(module, "_example_original_prefix", None)
        if type(prefix) is not str:
            raise RuntimeError("every parameter-owning FSDP unit must bind an original prefix")
        fqns = tuple(_join_fqn(prefix, str(name)) for name in flat._fqns)
        if any(name not in expected_shapes or name in seen for name in fqns):
            raise RuntimeError(f"FSDP original-FQN binding differs: {fqns}")
        seen.update(fqns)
        numels = tuple(int(value) for value in flat._numels)
        shapes = tuple(tuple(int(value) for value in shape) for shape in flat._shapes)
        padded = tuple(int(value) for value in flat._numels_with_padding)
        mask = tuple(bool(value) for value in flat._is_padding_mask)
        if len(fqns) != len(numels) or shapes != tuple(expected_shapes[name] for name in fqns):
            raise RuntimeError("FSDP FlatParameter metadata differs from original parameters")
        result.append(FlatBinding(flat, fqns, numels, shapes, padded, mask))
    if seen != set(expected_shapes):
        missing = sorted(set(expected_shapes).difference(seen))[:10]
        raise RuntimeError(f"FSDP did not bind the original parameter roster: {missing}")
    return result


def _all_gather_local(local: torch.Tensor, unpadded_numel: int) -> torch.Tensor:
    source = local.detach().contiguous().view(-1)
    gathered = torch.empty(
        source.numel() * WORLD_SIZE,
        dtype=source.dtype,
        device=source.device,
    )
    dist.all_gather_into_tensor(gathered, source)
    if unpadded_numel < 1 or unpadded_numel > gathered.numel():
        raise RuntimeError("unsharded FlatParameter size is invalid")
    return gathered[:unpadded_numel]


def _unflatten(binding: FlatBinding, full: torch.Tensor) -> dict[str, torch.Tensor]:
    pieces = full.split(binding.numels_with_padding)
    result: dict[str, torch.Tensor] = {}
    parameter_index = 0
    for piece, is_padding in zip(pieces, binding.padding_mask, strict=True):
        if is_padding:
            continue
        name = binding.fqns[parameter_index]
        result[name] = piece.view(binding.shapes[parameter_index])
        parameter_index += 1
    if parameter_index != len(binding.fqns):
        raise RuntimeError("FlatParameter unflatten roster differs")
    return result


def _gather_binding(binding: FlatBinding, *, gradient: bool = False) -> dict[str, torch.Tensor]:
    if gradient:
        local = binding.flat.grad
        if local is None:
            raise RuntimeError("FlatParameter gradient is absent")
    else:
        local = binding.local
    unpadded = int(binding.flat._unpadded_unsharded_size.numel())
    return _unflatten(binding, _all_gather_local(local, unpadded))


def _gather_named(bindings: Sequence[FlatBinding], *, gradient: bool = False) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for binding in bindings:
        values = _gather_binding(binding, gradient=gradient)
        if set(result).intersection(values):
            raise RuntimeError("duplicate original FQN during full-tensor gather")
        result.update(values)
    return result


def _flatten_for_binding(binding: FlatBinding, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    parameter_index = 0
    device = binding.local.device
    dtype = binding.local.dtype
    for numel, is_padding in zip(binding.numels_with_padding, binding.padding_mask, strict=True):
        if is_padding:
            pieces.append(torch.zeros(numel, device=device, dtype=dtype))
            continue
        name = binding.fqns[parameter_index]
        value = values[name].detach().to(device=device, dtype=dtype).reshape(-1)
        if value.numel() != binding.numels[parameter_index]:
            raise RuntimeError("shadow parameter numel differs")
        pieces.append(value)
        parameter_index += 1
    return torch.cat(pieces)


@torch.no_grad()
def _scatter_named(bindings: Sequence[FlatBinding], values: Mapping[str, torch.Tensor]) -> None:
    rank = dist.get_rank()
    for binding in bindings:
        full = _flatten_for_binding(binding, values)
        local = binding.local
        padded_numel = local.numel() * WORLD_SIZE
        if full.numel() > padded_numel:
            raise RuntimeError("shadow FlatParameter exceeds the padded shard geometry")
        if full.numel() < padded_numel:
            full = functional.pad(full, (0, padded_numel - full.numel()))
        shard = full.narrow(0, rank * local.numel(), local.numel())
        local.copy_(shard)
        if binding.flat.numel() == local.numel() and binding.flat.data_ptr() != local.data_ptr():
            binding.flat.copy_(shard)


class OriginalParameterOptimizer:
    """Replicated original-FQN shadow state for the reference implementation."""

    def __init__(
        self,
        bindings: Sequence[FlatBinding],
        *,
        method: str,
        lr: float,
    ) -> None:
        gathered = _gather_named(bindings)
        self.parameters = {
            name: torch.nn.Parameter(value.detach().clone(), requires_grad=True) for name, value in gathered.items()
        }
        ordered = [self.parameters[name] for name in sorted(self.parameters)]
        if method == "adamw8bit":
            from bitsandbytes.optim import AdamW8bit

            self.optimizer: Any = AdamW8bit(
                ordered,
                lr=lr,
                betas=BETAS,
                eps=EPS,
                weight_decay=WEIGHT_DECAY,
                min_8bit_size=MIN_8BIT_SIZE,
            )
        elif method == "adamw32":
            self.optimizer = torch.optim.AdamW(
                ordered,
                lr=lr,
                betas=BETAS,
                eps=EPS,
                weight_decay=WEIGHT_DECAY,
            )
        else:
            raise RuntimeError(f"unsupported shadow method: {method}")
        self.method = method

    @torch.no_grad()
    def step(self, bindings: Sequence[FlatBinding]) -> None:
        gradients = _gather_named(bindings, gradient=True)
        if set(gradients) != set(self.parameters):
            raise RuntimeError("repair gradient roster differs")
        for name, parameter in self.parameters.items():
            parameter.grad = gradients[name].detach().to(parameter).clone()
        self.optimizer.step()
        _scatter_named(bindings, self.parameters)
        self.optimizer.zero_grad(set_to_none=True)

    def fqn_state_bits(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for name, parameter in self.parameters.items():
            state = self.optimizer.state.get(parameter, {})
            if self.method == "adamw8bit":
                state1 = state.get("state1")
                result[name] = 8 if state1 is not None and state1.dtype is torch.uint8 else 32
            else:
                result[name] = 32
        return result


class VectorParameter(torch.nn.Module):
    def __init__(self, numel: int, frequency: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(numel, dtype=torch.float32))
        self.frequency = frequency

    def forward(self, token: torch.Tensor, update: int) -> torch.Tensor:
        index = torch.arange(self.weight.numel(), device=self.weight.device, dtype=torch.float32)
        coefficient = torch.sin((index + 1) * self.frequency + update * 0.03125)
        return token * (self.weight * coefficient).mean()


class SmallParameters(torch.nn.Module):
    def __init__(self, numels: Sequence[int]) -> None:
        super().__init__()
        self.leaves = torch.nn.ModuleList(
            VectorParameter(numel, 0.0078125 * (index + 1)) for index, numel in enumerate(numels)
        )

    def forward(self, token: torch.Tensor, update: int) -> torch.Tensor:
        return torch.stack([leaf(token, update) for leaf in self.leaves]).sum()


class ExampleModel(torch.nn.Module):
    def __init__(self, small_numels: Sequence[int]) -> None:
        super().__init__()
        self.small = SmallParameters(small_numels)
        self.large = VectorParameter(8192, 0.00390625)

    def forward(self, token: torch.Tensor, update: int) -> torch.Tensor:
        return self.small(token, update) + self.large(token, update)


def _initialize_model(module: torch.nn.Module, seed: int) -> None:
    with torch.no_grad():
        for name, parameter in sorted(module.named_parameters()):
            key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
            index = torch.arange(parameter.numel(), dtype=torch.float32).view(parameter.shape)
            parameter.copy_(0.025 * torch.sin(index * 0.017 + seed * 0.001 + key % 997))


def _fsdp_options(device: torch.device) -> dict[str, object]:
    from torch.distributed.fsdp import ShardingStrategy

    return {
        "device_id": device,
        "forward_prefetch": True,
        "limit_all_gathers": True,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "sync_module_states": False,
        "use_orig_params": False,
    }


def _wrap_model(
    arm: str,
    small_numels: Sequence[int],
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[FlatBinding], dict[str, tuple[int, ...]]]:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    probe = ExampleModel(small_numels)
    _initialize_model(probe, seed)
    shapes = {name: tuple(parameter.shape) for name, parameter in probe.named_parameters()}
    options = _fsdp_options(device)
    if arm == "fine":
        for index, leaf in enumerate(probe.small.leaves):
            wrapped = FSDP(leaf, **options)
            wrapped._example_original_prefix = f"small.leaves.{index}"
            probe.small.leaves[index] = wrapped
    elif arm == "coarse":
        wrapped_small = FSDP(probe.small, **options)
        wrapped_small._example_original_prefix = "small"
        probe.small = wrapped_small
    else:
        raise RuntimeError("Wrapping must be fine or coarse")
    wrapped_large = FSDP(probe.large, **options)
    wrapped_large._example_original_prefix = "large"
    probe.large = wrapped_large
    root = FSDP(probe, **options)
    root._example_original_prefix = ""
    return root, _flat_bindings(root, shapes), shapes


def run_layout(wrapping, reference, device):
    from bitsandbytes.optim import AdamW8bit

    model, bindings, _ = _wrap_model(wrapping, (2048,) * 4, seed=19, device=device)
    initial = {name: value.clone() for name, value in _gather_named(bindings).items()}
    if reference:
        optimizer = OriginalParameterOptimizer(bindings, method="adamw8bit", lr=1e-3)
    else:
        optimizer = AdamW8bit(
            model.parameters(),
            lr=1e-3,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            min_8bit_size=MIN_8BIT_SIZE,
        )
    for step in range(2):
        model.zero_grad(set_to_none=True)
        prediction = model(torch.ones((), device=device), step)
        (0.5 * (prediction - 0.1).square()).backward()
        if reference:
            optimizer.step(bindings)
        else:
            optimizer.step()
    values = _gather_named(bindings)
    updates = {name: value - initial[name] for name, value in values.items()}
    if reference:
        state_bits = optimizer.fqn_state_bits()
    else:
        state_bits = {}
        for binding in bindings:
            bits = 8 if optimizer.state[binding.flat]["state1"].dtype == torch.uint8 else 32
            state_bits.update({name: bits for name in binding.fqns})
    return updates, state_bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise RuntimeError("Run this fixture with exactly two torchrun processes")
    device = torch.device(args.device, int(os.environ["LOCAL_RANK"])) if args.device == "cuda" else torch.device("cpu")
    if args.device == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    try:
        results = {}
        for reference in (False, True):
            fine, fine_bits = run_layout("fine", reference, device)
            coarse, coarse_bits = run_layout("coarse", reference, device)
            maximum = max(float((fine[name] - coarse[name]).abs().max()) for name in fine)
            if reference:
                for name in fine:
                    torch.testing.assert_close(fine[name], coarse[name], rtol=0, atol=0)
                assert fine_bits == coarse_bits
                assert all(bits == 32 for name, bits in fine_bits.items() if name != "large.weight")
                assert fine_bits["large.weight"] == 8
            results["original_parameter_reference" if reference else "native"] = {
                "fine_state_bits": fine_bits,
                "coarse_state_bits": coarse_bits,
                "max_update_difference": maximum,
            }
        if dist.get_rank() == 0:
            print(json.dumps(results, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
