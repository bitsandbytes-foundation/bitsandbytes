"""FSDP state_dict save integration test for 4-bit quantized models (#1405).

This script must be launched via torchrun (not directly):
    torchrun --nproc_per_node=1 tests/fsdp_state_dict_save.py

It wraps a QLoRA-style model (frozen 4-bit base + trainable adapter) in FSDP
and calls get_model_state_dict with cpu_offload=True, which exercises the
_get_fqns() getattr traversal that previously crashed with:
    AttributeError: 'Params4bit' object has no attribute 'absmax'
"""

import sys

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn as nn

import bitsandbytes as bnb


class SimpleQLoRAModel(nn.Module):
    """Minimal model with a frozen 4-bit base layer and a trainable adapter."""

    def __init__(self, quant_type="nf4"):
        super().__init__()
        self.base = bnb.nn.Linear4bit(64, 64, bias=False, quant_type=quant_type)
        self.adapter = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.base(x) + self.adapter(x)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    errors = []

    for quant_type in ("nf4", "fp4"):
        model = SimpleQLoRAModel(quant_type=quant_type)
        model = model.to("cuda")

        # Freeze quantized base weights (as in real QLoRA)
        for p in model.base.parameters():
            p.requires_grad = False

        # Tell FSDP to ignore the frozen quantized params (can't flatten int dtypes)
        ignored = list(model.base.parameters())
        fsdp_model = FSDP(model, device_id=rank, ignored_states=ignored, use_orig_params=True)

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        try:
            state_dict = get_model_state_dict(fsdp_model, options=options)

            # Verify expected keys are present
            expected_substrings = ["base.weight", "absmax", "quant_map", "adapter.weight"]
            for substr in expected_substrings:
                if not any(substr in k for k in state_dict.keys()):
                    errors.append(f"{quant_type}: missing key containing '{substr}' in {list(state_dict.keys())}")

            print(f"{quant_type}: SUCCESS ({len(state_dict)} keys)", flush=True)
        except Exception as e:
            errors.append(f"{quant_type}: {type(e).__name__}: {e}")
            print(f"{quant_type}: FAILED: {e}", flush=True)

    dist.destroy_process_group()

    if errors:
        print("\nFAILURES:\n" + "\n".join(errors), file=sys.stderr, flush=True)
        sys.exit(1)
    else:
        print("\nAll FSDP state_dict tests passed.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
