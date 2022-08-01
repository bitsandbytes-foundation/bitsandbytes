import ctypes as ct
import os
from warnings import warn

from bitsandbytes.cuda_setup import evaluate_cuda_setup


class CUDALibrary_Singleton(object):
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.context = {}
        binary_name = evaluate_cuda_setup()
        if not os.path.exists(os.path.dirname(__file__) + f"/{binary_name}"):
            print(f"TODO: compile library for specific version: {binary_name}")
            print("defaulting to libbitsandbytes.so")
            self.lib = ct.cdll.LoadLibrary(
                os.path.dirname(__file__) + "/libbitsandbytes.so"
            )
        else:
            self.lib = ct.cdll.LoadLibrary(
                os.path.dirname(__file__) + f"/{binary_name}"
            )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


lib = CUDALibrary_Singleton.get_instance().lib
try:
    lib.cadam32bit_g32
    lib.get_context.restype = ct.c_void_p
    lib.get_cusparse.restype = ct.c_void_p
    COMPILED_WITH_CUDA = True
except AttributeError:
    warn(
        "The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers and GPU quantization are unavailable."
    )
    COMPILED_WITH_CUDA = False
