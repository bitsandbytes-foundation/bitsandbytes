from bitsandbytes.cextension import COMPILED_WITH_CUDA


class Backends:
    """
    An dict class for device backends that registered with 8bits and 4bits functions.

    The values of this device backends are lowercase strings, e.g., ``"cuda"``. They can
    be accessed as attributes with key-value, e.g., ``Backends.device["cuda"]``.

    """

    devices = {}

    @classmethod
    def register_backend(cls, backend_name: str, backend_instance):
        assert backend_name.lower() in {
            "cpu",
            "cuda",
            "xpu",
        }, "register device backend choices in [cpu, cuda, xpu]"

        cls.devices[backend_name.lower()] = backend_instance

if COMPILED_WITH_CUDA:
    from .cuda import CUDABackend
    cuda_backend = CUDABackend(torch.device("cuda").type)
    Backends.register_backend(cuda_backend.get_name(), cuda_backend)
# TODO: register more backends support
