import torch
import os
import ctypes as ct

lib = ct.cdll.LoadLibrary(os.path.dirname(__file__) + '/libClusterNet.so')

def get_ptr(A:torch.tensor):
    return ct.c_void_p(A.data.storage().data_ptr())

def estimate_quantiles(A:torch.Tensor, out:torch.Tensor=None, offset:float=1/512):
    '''
    Estimates 256 equidistant quantiles of the input tensor.

    Uses SRAM-quantiles algorithm to quickly estimate 256 equidistant quantiles
    via the eCDF of the input tensor `A`.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor. Any shape.
    out : torch.Tensor
        Tensor with the 256 estimated quantiles.
    offset : float
        The offset for the first and last quantile from 0 and 1.
    '''
    if out is None: out = torch.zeros((256,), dtype=torch.float32, device=A.device)
    if A.dtype == torch.float32:
        lib.estimate_quantiles_fp32(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    elif A.dtype == torch.float16:
        lib.estimate_quantiles_fp16(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    else:
        raise NotImplementError(f'Not supported data type {A.dtype}')
    return out
