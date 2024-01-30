import os
import pytest
import torch
from pathlib import Path

from bitsandbytes.cextension import HIP_ENVIRONMENT

# hardcoded test. Not good, but a sanity check for now
# TODO: improve this
@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
def test_manual_override(requires_cuda):
    manual_cuda_path = str(Path('/mmfs1/home/dettmers/data/local/cuda-12.2'))

    pytorch_version = torch.version.cuda.replace('.', '')

    assert pytorch_version != 122  # TODO: this will never be true...

    os.environ['CUDA_HOME']='{manual_cuda_path}'
    os.environ['BNB_CUDA_VERSION']='122'
    #assert str(manual_cuda_path) in os.environ['LD_LIBRARY_PATH']
    import bitsandbytes as bnb
    loaded_lib = bnb.cuda_setup.main.CUDASetup.get_instance().binary_name
    #assert loaded_lib == 'libbitsandbytes_cuda122.so'





