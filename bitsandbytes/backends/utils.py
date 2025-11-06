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


def convert_weight_packed_for_cpu(qweight: torch.Tensor,
                                  scales: torch.Tensor,
                                  block_n: int = 32):
    """
    qweight: (K * N / 2)  uint8
    return: packed_weight
    """
    assert qweight.dtype == torch.uint8, "qweight must be uint8"
    qweight = qweight.reshape(-1)
    unpacked_w = torch.empty(qweight.shape[0] * 2, dtype=torch.int32, device=A.device)
    unpacked_w[1::2] = qweight & 0xF
    unpacked_w[::2] = qweight >> 4
    qweight_final = unpacked_w.reshape(shape).transpose(-1, -2).to(torch.uint8)  # (*, N, K)
    # pack weight: [*, N, K] -> [*, N, K/2] combine low and high bit
    assert len(qweight_final.shape) == 2
    N, K = qweight_final.shape[0], qweight_final.shape[1]
    assert N % block_n == 0, "N must be divisible by block_n"
    assert K % 2 == 0, "K must be even"
    BLOCK_N = block_n
    BIT_COUNT = 32  # (=32 low +32 high)
    prefix = sizes[:-2]
    new_shape = [N // BLOCK_N, BLOCK_N, K // 2, 2]
    out_shape = [N, K // 2]
    qw = qweight_final.reshape(new_shape)                # (..., N/B, B, K/2, 2)
    qw = qw.transpose(-3, -2).contiguous()               # (..., N/B, K/2, B, 2)
    qw = qw.reshape(-1, BIT_COUNT * 2)                   # [-1, 64]
    high = qw[:, BIT_COUNT:]                             # high 32
    low  = qw[:, :BIT_COUNT]                             # low 32
    packed = ((high << 4) | low).to(torch.uint8)         # combine
    final_qweight = packed.reshape(out_shape)
    return final_qweight


GAUDI_SW_VER = get_gaudi_sw_version()
