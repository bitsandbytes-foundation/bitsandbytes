import os
from pathlib import Path

import torch


# hardcoded test. Not good, but a sanity check for now
# TODO: improve this
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
