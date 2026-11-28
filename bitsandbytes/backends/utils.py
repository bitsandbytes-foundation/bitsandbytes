from importlib.metadata import metadata

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


def get_gaudi_sw_version():
    """
    Returns the installed version of Gaudi SW.
    """
    gaudi_version = None
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
