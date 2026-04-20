#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end verification of bitsandbytes telemetry.

Exercises every wired-up feature once, tagging every event with a unique
run_id so the run's events can be correlated in Elasticsearch afterwards.

Usage:
    python scripts/verify_telemetry.py

Then wait ~30 seconds for ES indexing and query:

    es-cli -H esql 'FROM ds-hub-telemetry
      | WHERE @timestamp >= NOW() - 1 hour
        AND metadata.bitsandbytes.tag == "<PRINTED_RUN_ID>"
      | KEEP @timestamp, path_filename,
             metadata.bitsandbytes.feature,
             metadata.bitsandbytes.quant_type,
             metadata.bitsandbytes.variant,
             metadata.bitsandbytes.accel
      | SORT @timestamp ASC'

Expected features per run (on a CUDA host):
    params_4bit, linear_4bit, int8_params, linear_8bit, embedding (each
    variant hit), optimizer, optim_override_config,
    optim_register_module_override, outlier_aware_linear, int8_double_quant.
"""

from __future__ import annotations

import os
import time
import uuid


def main() -> int:
    run_id = f"verify-{uuid.uuid4().hex[:8]}"
    # Must be set BEFORE importing bitsandbytes so the tag is live when the
    # first telemetry event fires.
    os.environ["BNB_TELEMETRY_TAG"] = run_id
    print(f"run_id = {run_id}")

    import torch

    if not torch.cuda.is_available():
        print("CUDA not available — this script exercises GPU-only code paths.")
        print("Feature events requiring CUDA will not fire.")

    import bitsandbytes as bnb
    from bitsandbytes.functional import int8_double_quant
    from bitsandbytes.nn import (
        Embedding,
        Embedding4bit,
        Embedding8bit,
        Linear8bitLt,
        LinearNF4,
        OutlierAwareLinear,
        StableEmbedding,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Embeddings — fire at __init__, no device requirement
    StableEmbedding(16, 8)
    Embedding(16, 8)
    if device == "cuda":
        Embedding8bit(16, 8).to(device)
        Embedding4bit(16, 8, quant_type="nf4").to(device)

    # Linear4bit + Params4bit (NF4)
    if device == "cuda":
        layer4 = LinearNF4(64, 64, compute_dtype=torch.bfloat16).to(device)
        layer4(torch.randn(1, 64, device=device, dtype=dtype))

        # Linear8bitLt + Int8Params
        layer8 = Linear8bitLt(64, 64, has_fp16_weights=False, threshold=6.0).to(device)
        layer8(torch.randn(1, 64, device=device, dtype=dtype))

        # Optimizer: step() triggers the event
        param = torch.nn.Parameter(torch.randn(64, device=device))
        param.grad = torch.zeros_like(param)
        opt = bnb.optim.AdamW8bit([param], lr=1e-3)
        opt.step()

        # GlobalOptimManager: override_config + register_module_override
        mng = bnb.optim.GlobalOptimManager.get_instance()
        mng.override_config([param], "optim_bits", 32)
        mng.register_module_override(layer4, "weight", {"optim_bits": 32})

        # OutlierAwareLinear (deprecation candidate)
        OutlierAwareLinear(64, 64).to(device)

        # int8_double_quant (deprecation candidate)
        A = torch.randn(4, 64, device=device, dtype=torch.float16)
        int8_double_quant(A, threshold=6.0)

    # Drain the hf_hub telemetry daemon thread — events are queued
    # asynchronously, exiting too fast would kill them before they flush.
    print("exercising done, draining telemetry queue ...")
    time.sleep(8)

    print("\nnext steps:")
    print(f"  run_id: {run_id}")
    print(
        "  query: es-cli -H esql 'FROM ds-hub-telemetry | WHERE metadata.bitsandbytes.tag == \""
        + run_id
        + "\" | STATS count = COUNT(*) BY metadata.bitsandbytes.feature | SORT count DESC'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
