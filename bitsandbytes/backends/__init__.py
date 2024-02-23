from ..cextension import lib
from ._base import COOSparseTensor
from .nvidia import CudaBackend

_backend = CudaBackend(lib)
