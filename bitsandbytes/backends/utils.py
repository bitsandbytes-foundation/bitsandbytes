import functools
from importlib.metadata import metadata
from warnings import warn

from packaging import version
import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    triton_available = True
except ImportError:
    triton_available = False


# Cache 4-bit dequantization code tensors per (quant_type, device).
_code_4bit_cache: dict[tuple[str, torch.device], torch.Tensor] = {}


def _get_4bit_code(quant_type: str, device: torch.device) -> torch.Tensor:
    key = (quant_type, device)
    if key not in _code_4bit_cache:
        from bitsandbytes.functional import get_4bit_type

        _code_4bit_cache[key] = get_4bit_type(quant_type, device=device)
    return _code_4bit_cache[key]


@functools.cache
def _warn_gemm_4bit_unaligned(K: int, blocksize: int) -> None:
    """Warn at most once per (K, blocksize): the misalignment is a fixed property of the
    model, so repeating it on every gemm_4bit call is pure noise."""
    warn(
        f"inner dimension ({K}) is not aligned for fast kernel "
        f"with blocksize={blocksize}, falling back to slower implementation.",
        UserWarning,
    )


def get_gaudi_sw_version():
    """
    Returns the installed version of Gaudi SW.
    """
    try:
        # if we find the spec, examine the installed version
        plugin_metadata = metadata("habana-torch-plugin")
        plugin_version = plugin_metadata.get("Version")
        if plugin_version:
            gaudi_version = version.parse(plugin_version)
    except Exception:
        gaudi_version = None

    return gaudi_version


GAUDI_SW_VER = get_gaudi_sw_version()
