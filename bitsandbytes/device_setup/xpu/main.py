from .cpu.main import is_ipex_cpu_available
from warnings import warn

def is_ipex_xpu_available():
    if is_ipex_cpu_available():
        import intel_extension_for_pytorch
    else:
        return False

    if torch.xpu.is_available():
        return True
    else:
        warn("The installed version of intel_extension_for_pytorch is not supporting XPU device, "
            " or the XPU device is unavailable.")
        return False
