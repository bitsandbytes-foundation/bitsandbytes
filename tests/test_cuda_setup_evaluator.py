import os
import pytest
import torch
from pathlib import Path

# hardcoded test. Not good, but a sanity check for now
def test_manual_override():
    manual_cuda_path = str(Path('/mmfs1/home/dettmers/data/local/cuda-12.2'))

    pytorch_version = torch.version.cuda.replace('.', '')

    assert pytorch_version != 122

    os.environ['CUDA_HOME']='{manual_cuda_path}'
    os.environ['CUDA_VERSION']='122'
    assert str(manual_cuda_path) in os.environ['LD_LIBRARY_PATH']
    import bitsandbytes as bnb
    loaded_lib = bnb.cuda_setup.main.CUDASetup.get_instance().binary_name
    assert loaded_lib == 'libbitsandbytes_cuda122.so'








