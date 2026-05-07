"""Regression test for #1936: sm_87 multi-shape Linear4bit reboot.

Bug history
-----------
On bnb 0.46.1 against an NVIDIA Jetson Orin (sm_87) at nvpmodel
MAXN_SUPER, the host kernel reboots when three Linear4bit layers are
constructed and forwarded in sequence with the following recipe:

  - shape order: monotonically-increasing by output-feature product (A → B → C below)
  - quant_type: NF4
  - quant_storage: torch.bfloat16 (FSDP-compatible)
  - compute_dtype: torch.bfloat16
  - compress_statistics (double_quant): True
  - no inter-layer memory hygiene (no `del layer` or `empty_cache`)
  - batch size: 1

The fault is overwhelmingly cold-start-specific (~78% reboot rate
at cold-start, 0% at warm state across N=29+ warm-state samples).
bnb 0.49.2 also reboots at cold-start (N=1) — the original "fixed
in 0.49.2" framing was a warm-state artifact and was retracted in
a 2026-05-05 comment on the issue. A 256x256 NF4 forward executed
as the first GPU op after boot closes the cold-start race window
(N=3 verified). This test runs the original failing recipe at
sm_87 so any future regression surfaces in CI; CI runners targeting
sm_87 are expected to provide cold-state (e.g., reboot before the
test run) for it to fire reliably.

References
----------
- Issue: https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1936
- Reproduced N=4 across two physically separate Jetson Orin Nano
  Super 8GB units; fault travels with silicon + bnb build, not one
  defective board.
- A 13-test orthogonal-axis bisection found six axes (shape order,
  quant_type, quant_storage, compute_dtype, double_quant, hygiene)
  each independently sufficient to prevent the fault. The recipe in
  this test is the unique intersection that triggers it on 0.46.1.

Caveats
-------
- On a *broken* bnb the test crashes the host (system reboot), not
  just the test runner — failure mode is OS-level. pytest cannot
  capture this; absence of test output IS the regression signal.
- The bug is timing-sensitive (race condition); lower-power
  nvpmodel modes (15W / 25W) prevent it. The test does not enforce
  a power mode — sm_87 CI is expected to run at MAXN.
- Skipped on all non-sm_87 hardware. sm_87 is exclusive to the
  Jetson Orin family (Nano / NX / AGX).
"""

import pytest
import torch

import bitsandbytes as bnb

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 7),
    reason="Regression test for sm_87 (NVIDIA Jetson Orin family) only",
)


# Real-world shapes from Llama-3 / Qwen-2.5 / Mistral lm_head dimensions,
# selected to match the historical #1936 reproducer. Monotonically
# increasing by output-feature product is load-bearing — bisection found
# all five non-ABC permutations (ACB, BAC, BCA, CAB, CBA) pass cleanly.
# Do not substitute toy shapes here without re-validating against the
# historical repro.
SHAPES_ABC = [
    (4096, 32768),  # A
    (4096, 128256),  # B
    (3584, 152064),  # C
]


def test_linear4bit_multishape_bf16_storage_no_fault():
    """Run the #1936 recipe; assert each forward completes without fault.

    Cold-start race: at sm_87 cold-state on a buggy bnb the host reboots
    before the third forward returns. At warm-state (any prior bnb-NF4 op
    in the same session) every bnb version tested passes — see module
    docstring for the cold/warm characterization.
    """
    for in_features, out_features in SHAPES_ABC:
        layer = bnb.nn.Linear4bit(
            in_features,
            out_features,
            bias=False,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,  # double_quant
            quant_type="nf4",
            quant_storage=torch.bfloat16,  # FSDP-compatible storage path
        ).to("cuda")

        x = torch.randn(1, in_features, dtype=torch.bfloat16, device="cuda")
        y = layer(x)

        assert y.shape == (1, out_features)
        assert y.dtype == torch.bfloat16

        # Deliberately no `del layer`, `torch.cuda.empty_cache()`, or
        # `torch.cuda.synchronize()` between iterations. Any of those
        # individually prevents the historical fault and would mask a
        # regression of the 0.49.2 fix.
