import shlex
import subprocess
from typing import Tuple
import torch

def execute_and_return(command_string: str) -> Tuple[str, str]:
    def _decode(subprocess_err_out_tuple):
        return tuple(
            to_decode.decode("UTF-8").strip()
            for to_decode in subprocess_err_out_tuple
        )

    def execute_and_return_decoded_std_streams(command_string):
        return _decode(
            subprocess.Popen(
                shlex.split(command_string),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).communicate()
        )

    std_out, std_err = execute_and_return_decoded_std_streams(command_string)
    return std_out, std_err

__cuda_devices = None
def get_cuda_devices():
    global __cuda_devices
    if __cuda_devices is None:
        cuda_devices = []
        if torch.cuda.is_available():
            devices = [d for d in range(torch.cuda.device_count())]
            cuda_devices = [torch.cuda.get_device_name(d) for d in devices]
        __cuda_devices = cuda_devices
    return __cuda_devices

def is_cuda_device(device):
    return device in get_cuda_devices()
