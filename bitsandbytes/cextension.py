import ctypes as ct
import os
from warnings import warn

lib = ct.cdll.LoadLibrary(os.path.dirname(__file__) + '/libbitsandbytes.so')

try:
    lib.cadam32bit_g32
    COMPILED_WITH_CUDA = True
except AttributeError:
    warn("The installed version of bitsandbytes was compiled without GPU support. "
         "8-bit optimizers and GPU quantization are unavailable.")
    COMPILED_WITH_CUDA = False
