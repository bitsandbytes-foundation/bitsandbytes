import functools


@functools.lru_cache(None)
def is_triton_available():
    try:
        # torch>=2.2.0
        from torch.utils._triton import has_triton, has_triton_package

        return has_triton_package() and has_triton()
    except ImportError:
        from torch._inductor.utils import has_triton

        return has_triton()

def assert_same_device(a, b):
    assert a.is_cuda and b.is_cuda or a.is_xpu and b.is_xpu
