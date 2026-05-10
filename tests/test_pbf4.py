"""Tests for the PBF4 (peace-quant PBF-MX) implementation.

PBF4 here is a fixed 4-bit LUT derived dynamically from the PBF8 spine
(``_pbf8``) at every-other level — same construction as peace-quant's
``mx_pbf_lut``. No per-tensor calibration. The same LUT lives in
``QuantState.code`` for every tensor; CUDA's ``kgemm_4bit_inference_naive``
reads it as the ``datatype`` arg into ``__shared__ T quant_map[16]``.
"""

import math

import pytest
import torch

from bitsandbytes import _pbf8
from bitsandbytes._pbf4 import (
    NUM_LUT_MAGS,
    PBF_MX_LUT,
)
from bitsandbytes.backends.utils import CODE
import bitsandbytes.functional as F
from bitsandbytes.nn import Linear4bit


def test_no_default_lut_in_code_dict():
    assert "pbf4" not in CODE


def test_pbf_mx_lut_layout():
    listed = PBF_MX_LUT.tolist()
    assert len(listed) == 16
    assert listed == sorted(listed)
    assert listed[0] == pytest.approx(-1.0)
    assert listed[-1] == pytest.approx(1.0)
    assert sum(1 for v in listed if v == 0.0) == 1
    # NF4-style asymmetric: 7 negative + 0 + 8 positive.
    assert sum(1 for v in listed if v < 0) == 7
    assert sum(1 for v in listed if v > 0) == 8


def test_pbf_mx_lut_log_step():
    # Step ratio between adjacent positive entries should be exp(2·LEVEL_LOG_STEP)
    # = 2^(3/8) ≈ 1.297 — the every-other-level sampling of PBF8.
    pos = [v for v in PBF_MX_LUT.tolist() if v > 0]
    expected_ratio = math.exp(2.0 * _pbf8.LEVEL_LOG_STEP)
    for k in range(len(pos) - 1):
        ratio = pos[k + 1] / pos[k]
        assert ratio == pytest.approx(expected_ratio, rel=1e-5), (
            f"pos[{k + 1}]/pos[{k}] = {ratio}, expected {expected_ratio}"
        )


def test_pbf_mx_lut_derived_from_pbf8():
    raw = _pbf8.sample_every_other_level(NUM_LUT_MAGS, start_level=2)
    top = raw[-1]
    expected_normalised = [m / top for m in raw]
    # Nonzero positives in the LUT match the 8 normalised mags from PBF8.
    pos_nonzero = [v for v in PBF_MX_LUT.tolist() if v > 0]
    assert len(pos_nonzero) == NUM_LUT_MAGS
    for actual, expected in zip(pos_nonzero, expected_normalised):
        assert actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("blocksize", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pbf4_roundtrip_cpu(blocksize, dtype):
    torch.manual_seed(0)
    A = torch.randn(256, 256, device="cpu", dtype=dtype)

    qa, state = F.quantize_4bit(A, blocksize=blocksize, quant_type="pbf4")
    A_dq = F.dequantize_4bit(qa, state, blocksize=blocksize, quant_type="pbf4")

    assert state.quant_type == "pbf4"
    assert state.code.numel() == 16
    assert qa.dtype == torch.uint8
    assert A_dq.shape == A.shape
    assert A_dq.dtype == dtype
    assert torch.isfinite(A_dq).all()


def test_pbf4_state_code_is_the_fixed_lut():
    torch.manual_seed(0)
    A = torch.randn(128, 128)
    _, state = F.quantize_4bit(A, blocksize=64, quant_type="pbf4")
    torch.testing.assert_close(state.code, PBF_MX_LUT.to(state.code.device), rtol=0, atol=0)

    # All tensors get the same LUT — this is a fixed format, not calibrated.
    B = torch.randn(64, 64) * 100
    _, state2 = F.quantize_4bit(B, blocksize=64, quant_type="pbf4")
    torch.testing.assert_close(state2.code, state.code, rtol=0, atol=0)


def test_pbf4_quant_state_serialization_cpu():
    torch.manual_seed(0)
    A = torch.randn(128, 128, device="cpu", dtype=torch.float32)

    qa, state = F.quantize_4bit(A, blocksize=64, quant_type="pbf4")
    packed = state.as_dict(packed=True)
    assert any("bitsandbytes__pbf4" in k for k in packed.keys())

    restored = F.QuantState.from_dict(state.as_dict(), device=torch.device("cpu"))
    assert restored.quant_type == "pbf4"
    torch.testing.assert_close(restored.code, state.code, rtol=0, atol=0)
    torch.testing.assert_close(restored.absmax, state.absmax, rtol=0, atol=0)

    A_dq_orig = F.dequantize_4bit(qa, state, blocksize=64, quant_type="pbf4")
    A_dq_restored = F.dequantize_4bit(qa, restored, blocksize=64, quant_type="pbf4")
    torch.testing.assert_close(A_dq_orig, A_dq_restored)


def test_pbf4_compress_statistics_compatible():
    torch.manual_seed(0)
    A = torch.randn(512, 512)
    qa, state = F.quantize_4bit(A, blocksize=64, quant_type="pbf4", compress_statistics=True)
    assert state.nested
    A_dq = F.dequantize_4bit(qa, state, blocksize=64, quant_type="pbf4")
    assert torch.isfinite(A_dq).all()


def test_pbf4_op_direct_roundtrip():
    # The op layer handles pbf4 directly via the fixed PBF_MX_LUT —
    # no special functional-only path required.
    torch.manual_seed(0)
    A = torch.randn(64, 64)
    qa, absmax = torch.ops.bitsandbytes.quantize_4bit.default(A, 64, "pbf4", torch.uint8)
    out = torch.ops.bitsandbytes.dequantize_4bit.default(qa, absmax, 64, "pbf4", tuple(A.shape), A.dtype)
    assert torch.isfinite(out).all()
    err = (A - out).abs().mean().item()
    assert err < 0.15


def test_pbf4_finite_on_diverse_distributions():
    torch.manual_seed(0)
    distros = {
        "normal": torch.randn(4096),
        "uniform": torch.rand(4096) * 2 - 1,
        "log-uniform": torch.exp(torch.empty(4096).uniform_(-6, 0)) * torch.sign(torch.randn(4096)),
        "cauchy": torch.distributions.Cauchy(0.0, 0.3).sample((4096,)).clamp_(-50, 50),
        "student-t": torch.distributions.StudentT(3.0).sample((4096,)),
        "mix-outliers": torch.cat([torch.randn(3900), torch.randn(196) * 50.0]),
    }
    for name, A in distros.items():
        if A.dim() == 1:
            A = A.unsqueeze(0)
        qa, st = F.quantize_4bit(A, blocksize=64, quant_type="pbf4")
        dq = F.dequantize_4bit(qa, st, blocksize=64, quant_type="pbf4")
        assert torch.isfinite(dq).all(), f"{name}: dequantised output has NaN/Inf"
        assert (dq.abs() <= A.abs().max() * 1.05).all(), f"{name}: dequantised exceeds 1.05x tensor max"


def test_linear4bit_pbf4_forward_cpu():
    torch.manual_seed(0)
    layer = Linear4bit(64, 32, bias=True, quant_type="pbf4", compress_statistics=False)
    layer = layer.to("cpu")  # triggers quantization

    state = layer.weight.quant_state
    assert state.quant_type == "pbf4"
    assert state.code.numel() == 16

    x = torch.randn(8, 64, dtype=torch.float32)
    y = layer(x)
    assert y.shape == (8, 32)
    assert torch.isfinite(y).all()
