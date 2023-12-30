import ctypes as ct
import torch

from pathlib import Path
from warnings import warn

if torch.backends.mps.is_built():
    package_dir = Path(__file__).parent
    binary_path = package_dir / "libbitsandbytes_mps.dylib"
    lib = ct.cdll.LoadLibrary(binary_path)
    COMPILED_WITH_CUDA = False
elif torch.cuda.is_available():
    from bitsandbytes.cuda_setup.main import CUDASetup

    setup = CUDASetup.get_instance()
    if setup.initialized != True:
        setup.run_cuda_setup()

# print the setup details after checking for errors so we do not print twice
#if 'BITSANDBYTES_NOWELCOME' not in os.environ or str(os.environ['BITSANDBYTES_NOWELCOME']) == '0':
    #setup.print_log_stack()
