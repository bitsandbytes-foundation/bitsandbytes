import functools


@functools.lru_cache(None)
def is_triton_available():
    try:
        from torch.utils._triton import has_triton, has_triton_package

        return has_triton_package() and has_triton()
    except Exception:
        return False
