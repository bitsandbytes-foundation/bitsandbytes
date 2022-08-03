import ctypes
from dataclasses import dataclass, field


@dataclass
class CudaLibVals:
    # code bits taken from
    # https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549

    nGpus: ctypes.c_int = field(default=ctypes.c_int())
    cc_major: ctypes.c_int = field(default=ctypes.c_int())
    cc_minor: ctypes.c_int = field(default=ctypes.c_int())
    device: ctypes.c_int = field(default=ctypes.c_int())
    error_str: ctypes.c_char_p = field(default=ctypes.c_char_p())
    cuda: ctypes.CDLL = field(init=False, repr=False)
    ccs: List[str, ...] = field(init=False)

    def _initialize_driver_API(self):
        self.check_cuda_result(self.cuda.cuInit(0))

    def _load_cuda_lib(self):
        """
        1. find libcuda.so library (GPU driver) (/usr/lib)
           init_device -> init variables -> call function by reference
        """
        libnames = "libcuda.so"
        for libname in libnames:
            try:
                self.cuda = ctypes.CDLL(libname)
            except OSError:
                continue
            else:
                break
        else:
            raise OSError("could not load any of: " + " ".join(libnames))

    def call_cuda_func(self, function_obj, **kwargs):
        CUDA_SUCCESS = 0  # constant taken from cuda.h
        pass
        # if (CUDA_SUCCESS := function_obj(

    def _error_handle(cuda_lib_call_return_value):
        """
        2. call extern C function to determine CC
           (see https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html)
        """
        CUDA_SUCCESS = 0  # constant taken from cuda.h

        if cuda_lib_call_return_value != CUDA_SUCCESS:
            self.cuda.cuGetErrorString(
                cuda_lib_call_return_value,
                ctypes.byref(self.error_str),
            )
            print("Count not initialize CUDA - failure!")
            raise Exception("CUDA exception!")
        return cuda_lib_call_return_value

    def __post_init__(self):
        self._load_cuda_lib()
        self._initialize_driver_API()
        self.check_cuda_result(
            self.cuda, self.cuda.cuDeviceGetCount(ctypes.byref(self.nGpus))
        )
        tmp_ccs = []
        for gpu_index in range(self.nGpus.value):
            check_cuda_result(
                self.cuda,
                self.cuda.cuDeviceGet(ctypes.byref(self.device), gpu_index),
            )
            check_cuda_result(
                self.cuda,
                self.cuda.cuDeviceComputeCapability(
                    ctypes.byref(self.cc_major),
                    ctypes.byref(self.cc_minor),
                    self.device,
                ),
            )
            tmp_ccs.append(f"{self.cc_major.value}.{self.cc_minor.value}")
        self.ccs = sorted(tmp_ccs, reverse=True)
