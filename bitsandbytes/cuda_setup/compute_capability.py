import ctypes
from dataclasses import dataclass, field


CUDA_SUCCESS = 0

@dataclass
class CudaLibVals:
    # code bits taken from 
    # https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549

    nGpus = ctypes.c_int()
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    device = ctypes.c_int()
    error_str = ctypes.c_char_p()
    cuda: ctypes.CDLL = field(init=False, repr=False)
    ccs: List[str, ...] = field(init=False)

    def load_cuda_lib(self):
        """
        1. find libcuda.so library (GPU driver) (/usr/lib)
           init_device -> init variables -> call function by reference
        """
        libnames = ("libcuda.so")
        for libname in libnames:
            try:
                self.cuda = ctypes.CDLL(libname)
            except OSError:
                continue
            else:
                break
        else:
            raise OSError("could not load any of: " + " ".join(libnames))

    def check_cuda_result(self, result_val):
        """
        2. call extern C function to determine CC 
           (see https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html)
        """
        cls_fields: Tuple[Field, ...] = fields(self.__class__)

        if result_val != 0:
            self.cuda.cuGetErrorString(result_val, ctypes.byref(self.error_str))
            print("Count not initialize CUDA - failure!")
            raise Exception("CUDA exception!")
        return result_val

    def __post_init__(self):
        self.load_cuda_lib()
        self.check_cuda_result(self.cuda.cuInit(0))
        self.check_cuda_result(self.cuda, self.cuda.cuDeviceGetCount(ctypes.byref(self.nGpus)))
        tmp_ccs = []
        for gpu_index in range(self.nGpus.value):
            check_cuda_result(
                self.cuda, self.cuda.cuDeviceGet(ctypes.byref(self.device), gpu_index)
            )
            check_cuda_result(
                self.cuda,
                self.cuda.cuDeviceComputeCapability(
                    ctypes.byref(self.cc_major), ctypes.byref(self.cc_minor), self.device
                ),
            )
            tmp_ccs.append(f"{self.cc_major.value}.{self.cc_minor.value}")
        self.ccs = sorted(tmp_ccs, reverse=True)
