import ctypes as ct
import torch

from pathlib import Path
from warnings import warn


class CUDASetup(object):
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def generate_instructions(self):
        if self.cuda is None:
            self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA library was not detected.')
            self.add_log_entry('CUDA SETUP: Solution 1): Your paths are probably not up-to-date. You can update them via: sudo ldconfig.')
            self.add_log_entry('CUDA SETUP: Solution 2): If you do not have sudo rights, you can do the following:')
            self.add_log_entry('CUDA SETUP: Solution 2a): Find the cuda library via: find / -name libcuda.so 2>/dev/null')
            self.add_log_entry('CUDA SETUP: Solution 2b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_2a')
            self.add_log_entry('CUDA SETUP: Solution 2c): For a permanent solution add the export from 2b into your .bashrc file, located at ~/.bashrc')
            return

        if self.cudart_path is None:
            self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA runtime library was not detected.')
            self.add_log_entry('CUDA SETUP: Solution 1: To solve the issue the libcudart.so location needs to be added to the LD_LIBRARY_PATH variable')
            self.add_log_entry('CUDA SETUP: Solution 1a): Find the cuda runtime library via: find / -name libcudart.so 2>/dev/null')
            self.add_log_entry('CUDA SETUP: Solution 1b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_1a')
            self.add_log_entry('CUDA SETUP: Solution 1c): For a permanent solution add the export from 1b into your .bashrc file, located at ~/.bashrc')
            self.add_log_entry('CUDA SETUP: Solution 2: If no library was found in step 1a) you need to install CUDA.')
            self.add_log_entry('CUDA SETUP: Solution 2a): Download CUDA install script: wget https://github.com/TimDettmers/bitsandbytes/blob/main/cuda_install.sh')
            self.add_log_entry('CUDA SETUP: Solution 2b): Install desired CUDA version to desired location. The syntax is bash cuda_install.sh CUDA_VERSION PATH_TO_INSTALL_INTO.')
            self.add_log_entry('CUDA SETUP: Solution 2b): For example, "bash cuda_install.sh 113 ~/local/" will download CUDA 11.3 and install into the folder ~/local')
            return

        make_cmd = f'CUDA_VERSION={self.cuda_version_string}'
        if len(self.cuda_version_string) < 3:
            make_cmd += ' make cuda92'
        elif self.cuda_version_string == '110':
            make_cmd += ' make cuda110'
        elif self.cuda_version_string[:2] == '11' and int(self.cuda_version_string[2]) > 0:
            make_cmd += ' make cuda11x'

        has_cublaslt = self.cc in ["7.5", "8.0", "8.6"]
        if not has_cublaslt:
            make_cmd += '_nomatmul'

        self.add_log_entry('CUDA SETUP: Something unexpected happened. Please compile from source:')
        self.add_log_entry('git clone git@github.com:TimDettmers/bitsandbytes.git')
        self.add_log_entry('cd bitsandbytes')
        self.add_log_entry(make_cmd)
        self.add_log_entry('python setup.py install')

    def initialize(self):
        self.cuda_setup_log = []
        self.lib = None

        from .cuda_setup.main import evaluate_cuda_setup
        binary_name, cudart_path, cuda, cc, cuda_version_string = evaluate_cuda_setup()
        self.cudart_path = cudart_path
        self.cuda = cuda
        self.cc = cc
        self.cuda_version_string = cuda_version_string

        package_dir = Path(__file__).parent
        binary_path = package_dir / binary_name

        try:
            if not binary_path.exists():
                self.add_log_entry(f"CUDA SETUP: Required library version not found: {binary_name}. Maybe you need to compile it from source?")
                legacy_binary_name = "libbitsandbytes.so"
                self.add_log_entry(f"CUDA SETUP: Defaulting to {legacy_binary_name}...")
                binary_path = package_dir / legacy_binary_name
                if not binary_path.exists():
                    self.add_log_entry('')
                    self.add_log_entry('='*48 + 'ERROR' + '='*37)
                    self.add_log_entry('CUDA SETUP: CUDA detection failed! Possible reasons:')
                    self.add_log_entry('1. CUDA driver not installed')
                    self.add_log_entry('2. CUDA not installed')
                    self.add_log_entry('3. You have multiple conflicting CUDA libraries')
                    self.add_log_entry('4. Required library not pre-compiled for this bitsandbytes release!')
                    self.add_log_entry('CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=113`.')
                    self.add_log_entry('='*80)
                    self.add_log_entry('')
                    self.generate_instructions()
                    self.print_log_stack()
                    raise Exception('CUDA SETUP: Setup Failed!')
                self.lib = ct.cdll.LoadLibrary(binary_path)
            else:
                self.add_log_entry(f"CUDA SETUP: Loading binary {binary_path}...")
                self.lib = ct.cdll.LoadLibrary(binary_path)
        except:
            self.print_log_stack()

    def add_log_entry(self, msg, is_warning=False):
        self.cuda_setup_log.append((msg, is_warning))

    def print_log_stack(self):
        for msg, is_warning in self.cuda_setup_log:
            if is_warning:
                warn(msg)
            else:
                print(msg)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


lib = CUDASetup.get_instance().lib
try:
    if lib is None and torch.cuda.is_available():
        CUDASetup.get_instance().generate_instructions()
        CUDASetup.get_instance().print_log_stack()
        raise RuntimeError('''
        CUDA Setup failed despite GPU being available. Inspect the CUDA SETUP outputs aboveto fix your environment!
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
