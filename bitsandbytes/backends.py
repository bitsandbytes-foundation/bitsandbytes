class Backends:
    """
    An dict class for device backends that registered with 8bits and 4bits functions.

    The values of this device backends are lowercase strings, e.g., ``"cuda"``. They can
    be accessed as attributes with key-value, e.g., ``Backends.device["cuda"]``.

    """

    def __init__(self):
        self.devices = {}

    @classmethod
    def register_backend(backend_name: str, backend_class):
        assert backend_name.lower() in {
            "cpu",
            "cuda",
            "xpu",
        }, "register device backend choices in [cpu, cuda, xpu]"

        # check 8bits or 4bits functionality, at least one is compelete
        if (
            hasattr(backend_class, "double_quant")
            and hasattr(backend_class, "transform")
            and hasattr(backend_class, "igemmlt")
            and hasattr(backend_class, "mm_dequant")
            and hasattr(backend_class, "extract_outliers")
        ):
            self.devices[backend_name.lower()] = backend_class

        elif hasattr(backend_class, "quantize_4bit") and hasattr(
            backend_class, "dequantize_4bit"
        ):
            self.devices[backend_name.lower()] = backend_classq
        else:
            assert (
                False
            ), f"register device backend {backend_name.lower()} but its functionality is not compelete"
