import importlib

def is_triton_available():
    return importlib.util.find_spec("triton") is not None
