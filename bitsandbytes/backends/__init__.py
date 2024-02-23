from ..cextension import lib
from ._base import COOSparseTensor
from .nvidia import CudaBackend

_backend = CudaBackend(lib) if lib else None
# TODO: this should actually be done in `cextension.py` and potentially with .get_instance()
#       for now this is just a simplifying assumption
#
# Notes from Tim:
#       backend = CUDABackend.get_instance()
#    	 -> CUDASetup -> lib -> backend.clib = lib
