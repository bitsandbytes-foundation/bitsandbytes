"""PBF8 — peace-quant's standard 8-bit log-polar spine.

Mirrors the constants from
[crates/pbf8/src/format.rs](crates/pbf8/src/format.rs):

- ``BASE = φ + π`` (≈ 4.7596) — the irrational base anchoring all rings.
- ``RING_POW`` — 8-element ring spine, each ring 8x the previous (R=8 spacing).
  Spans roughly ``5.8e-4 .. 1100``.
- ``LEVEL_LOG_STEP = ln(8) / 16`` — log-step per level inside a ring.
- 8 rings * 16 levels = 128 magnitudes per sign side. With the byte-0/byte-255
  sentinels, total 256 codes per byte.

For a magnitude index ``mag ∈ [0, 127]``:
``decode_mag(mag) = (BASE/8192) · exp(mag · LEVEL_LOG_STEP) = RING_POW[mag>>4] · exp((mag & 0xF) · LEVEL_LOG_STEP)``.

This module exposes the spine to higher-level formats. PBF4 (``_pbf4``) builds
its 4-bit LUT by sampling 8 magnitudes from this spine at every-other level.
"""

import math

PHI: float = 1.618_034
BASE: float = PHI + math.pi  # ≈ 4.7595918

# Ring spine — 8 rings at R=8 spacing, span ≈ 5.8e-4 .. 1100.
RING_POW: tuple[float, ...] = (
    BASE / 8192.0,
    BASE / 1024.0,
    BASE / 128.0,
    BASE / 16.0,
    BASE / 2.0,
    4.0 * BASE,
    32.0 * BASE,
    256.0 * BASE,
)

LEVEL_LOG_STEP: float = math.log(8.0) / 16.0  # = 3·ln(2)/16

# 128 magnitudes (mag indices 0..127). Byte 0 is the zero sentinel, byte 255
# is the saturation sentinel; nonzero magnitudes occupy bytes 1..254.
N_MAGS: int = 128


def decode_mag(mag: int) -> float:
    """Decode a magnitude index ``mag ∈ [0, N_MAGS)`` to its positive fp32 value."""
    if mag <= 0:
        return RING_POW[0]
    if mag >= N_MAGS:
        mag = N_MAGS - 1
    return (BASE / 8192.0) * math.exp(mag * LEVEL_LOG_STEP)


def sample_every_other_level(n: int = 8, start_level: int = 0) -> list[float]:
    """Return ``n`` magnitudes by sampling the PBF8 spine at every-other level.

    Used to derive lower-bit-depth LUTs (PBF4 takes ``n=8``). Sampling stride
    is 2 levels = ``2 · LEVEL_LOG_STEP`` (~30% step ratio).
    """
    return [decode_mag(start_level + 2 * k) for k in range(n)]
