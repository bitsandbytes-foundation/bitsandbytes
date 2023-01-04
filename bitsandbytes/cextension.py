import ctypes as ct
import os
import torch

from pathlib import Path
from warnings import warn

from bitsandbytes.cuda_setup.main import CUDASetup


setup = CUDASetup.get_instance()
if setup.initialized != True:
    setup.run_cuda_setup()
    if 'BITSANDBYTES_NOWELCOME' not in os.environ or str(os.environ['BITSANDBYTES_NOWELCOME']) == '0':
        setup.print_log_stack()

lib = setup.lib
try:
    if lib is None and torch.cuda.is_available():
        CUDASetup.get_instance().generate_instructions()
        CUDASetup.get_instance().print_log_stack()
        raise RuntimeError('''
        CUDA Setup failed despite GPU being available. Inspect the CUDA SETUP outputs above to fix your environment!
        If you cannot find any issues and suspect a bug, please open an issue with detals about your environment:
        https://github.com/TimDettmers/bitsandbytes/issues''')
    lib.cadam32bit_g32
    lib.get_context.restype = ct.c_void_p
    lib.get_cusparse.restype = ct.c_void_p
    COMPILED_WITH_CUDA = True
except AttributeError:
    warn("The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers and GPU quantization are unavailable.")
    COMPILED_WITH_CUDA = False
