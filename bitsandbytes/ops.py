import torch
import os
import ctypes as ct

lib = ct.cdll.LoadLibrary(os.path.dirname(__file__) + '/libClusterNet.so')

def get_ptr(A:torch.tensor):
    return ct.c_void_p(A.data.storage().data_ptr())

def testmul(A : int, scalar=1.0, out=None):
    '''
    Doc test function.

    This function tests the documentation system.

    Parameters
    ----------
    A : int
        its actually a tensor
    scalar : float
        scales
    out : tensor, opional
        can be used.
    '''
    if out is None: out = torch.empty((A.shape[0],A.shape[1]))
    lib.ffscalar_mul(ct.c_void_p(A.data.storage().data_ptr()), ct.c_void_p(out.data.storage().data_ptr()), ct.c_int32(A.numel()), ct.c_float(scalar))
    return out


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
    lib.estimate_quantiles(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    return out
