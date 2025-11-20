import subprocess

from packaging import version
import torch

try:
    import triton.language as tl  # noqa: F401

    import triton  # noqa: F401

    triton_available = True
except ImportError:
    triton_available = False


_NF4_QUANT_TABLE = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
    device="xpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cpu",  # Only cpu/xpu use this table for now.
)
_FP4_QUANT_TABLE = torch.tensor(
    [
        0.0000,
        0.0052,
        0.6667,
        1.0000,
        0.3333,
        0.5000,
        0.1667,
        0.2500,
        0.0000,
        -0.0052,
        -0.6667,
        -1.0000,
        -0.3333,
        -0.5000,
        -0.1667,
        -0.2500,
    ],
    dtype=torch.float32,
    device="xpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cpu",  # Only cpu/xpu use this table for now.
)
CODE = {"nf4": _NF4_QUANT_TABLE, "fp4": _FP4_QUANT_TABLE}


def get_gaudi_sw_version():
    """
    Returns the installed version of Gaudi SW.
    """
    output = subprocess.run(
        "pip list | grep habana-torch-plugin",
        shell=True,
        text=True,
        capture_output=True,
    )
    # If grep return nothing
    if not output.stdout.strip():
        return None

    return version.parse(output.stdout.split("\n")[0].split()[-1])


GAUDI_SW_VER = get_gaudi_sw_version()
