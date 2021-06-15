import torch
import os
import ctypes as ct

torch.optim.Adam
lib = ct.cdll.LoadLibrary(os.path.dirname(__file__) + '/libClusterNet.so')

def create_dynamic_map(signed=True):
    '''
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    '''

    n = 7

    data = []
    for i in range(n):
        fraction_items = 2**i+1 if signed else 2**(i+1)+1
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1]+boundaries[1:])/2.0
        data += ((10**(-6+i))*means).tolist()
        if signed:
            data += (-(10**(-6+i))*means).tolist()

    data.append(0)
    data.append(1.0)
    data.sort()
    return torch.Tensor(data)

def get_ptr(A: torch.Tensor) -> ct.c_void_p:
    '''
    Get the ctypes pointer from a PyTorch Tensor.

    Parameters
    ----------
    A : torch.tensor
        The PyTorch tensor.
    '''
    return ct.c_void_p(A.data.storage().data_ptr())

def estimate_quantiles(A: torch.Tensor, out: torch.Tensor=None, offset: float=1/512) -> torch.Tensor:
    '''
    Estimates 256 equidistant quantiles on the input tensor eCDF.

    Uses SRAM-Quantiles algorithm to quickly estimate 256 equidistant quantiles
    via the eCDF of the input tensor `A`. This is a fast but approximate algorithm
    and the extreme quantiles close to 0 and 1 have high variance / large estimation
    errors. These large errors can be circumnavigated by using the offset variable.
    Default offset value of 1/512 ensures minimum entropy encoding. An offset value
    of 0.01 to 0.02 usually has a much lower error. Given an offset of 0.02 equidistance
    points in the range [0.02, 0.98] are used for the quantiles.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor. Any shape.
    out : torch.Tensor
        Tensor with the 256 estimated quantiles.
    offset : float
        The offset for the first and last quantile from 0 and 1. Default: 1/512

    Returns
    -------
    torch.Tensor:
        The 256 quantiles in float32 datatype.
    '''
    if out is None: out = torch.zeros((256,), dtype=torch.float32, device=A.device)
    if A.dtype == torch.float32:
        lib.cestimate_quantiles_fp32(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    elif A.dtype == torch.float16:
        lib.cestimate_quantiles_fp16(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    else:
        raise NotImplementError(f'Not supported data type {A.dtype}')
    return out

def quantize(code: torch.Tensor, A: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    '''
    Quantizes input tensor to 8-bit.

    Quantizes the 32-bit input tensor `A` to the 8-bit output tensor
    `out` using the quantization map `code`.

    Parameters
    ----------
    code : torch.Tensor
        The quantization map.
    A : torch.Tensor
        The input tensor.
    out : torch.Tensor, optional
        The output tensor. Needs to be of type byte.

    Returns
    -------
    torch.Tensor:
        Quantized 8-bit tensor.
    '''
    if out is None: out = torch.zeros_like(A, dtype=torch.uint8)
    lib.cquantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    return out

def dequantize_with_absmax(code: torch.Tensor, absmax:torch.Tensor, A: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    out = dequantize(code, A, out)
    return out*absmax

def dequantize(code: torch.Tensor, A: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    '''
    Dequantizes the 8-bit tensor to 32-bit.

    Dequantizes the 8-bit tensor `A` to the 32-bit tensor `out` via
    the quantization map `code`.

    Parameters
    ----------
    code : torch.Tensor
        The quantization map.
    A : torch.Tensor
        The 8-bit input tensor.
    out : torch.Tensor
        The 32-bit output tensor.

    Returns
    -------
    torch.Tensor:
        32-bit output tensor.
    '''
    if out is None: out = torch.zeros_like(A, dtype=torch.float32)
    lib.cdequantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    return out


def adam_update_32bit(g: torch.Tensor, p: torch.Tensor, state1: torch.Tensor, state2: torch.Tensor,
                beta1: float, beta2: float, eps: float,
                step: int, lr: float, weight_decay: float=0.0, is_sparse: bool = False) -> None:
    '''
    Performs an inplace Adam update.

    Universal Adam update for 32/8-bit state and 32/16-bit gradients/weights.
    Uses AdamW formulation if weight decay > 0.0.

    Parameters
    ----------
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Adam state 1.
    state2 : torch.Tensor
        Adam state 2.
    beta1 : float
        Adam beta1.
    beta2 : float
        Adam beta2.
    eps : float
        Adam epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    is_sparse : bool
        If the gradient can be sparse or not.
    '''

    if g.dtype == torch.float32 and state1.dtype == torch.float32:
        lib.cadam32bit_g32(get_ptr(g), get_ptr(p), get_ptr(state1), get_ptr(state2),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_float(weight_decay),
                    ct.c_int32(step), ct.c_float(lr), ct.c_bool(is_sparse), ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.float32:
        lib.cadam32bit_g16(get_ptr(g), get_ptr(p), get_ptr(state1), get_ptr(state2),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_float(weight_decay),
                    ct.c_int32(step), ct.c_float(lr), ct.c_bool(is_sparse), ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')


def adam_update_8bit(g: torch.Tensor, p: torch.Tensor, state1: torch.Tensor, state2: torch.Tensor,
                beta1: float, beta2: float, eps: float,
                step: int, lr: float, qmap1: torch.Tensor, qmap2: torch.Tensor,
                max1: torch.Tensor, max2: torch.Tensor, new_max1: torch.Tensor, new_max2: torch.Tensor,
                weight_decay: float=0.0, is_sparse: bool=False) -> None:
    '''
    Performs an inplace Adam update.

    Universal Adam update for 32/8-bit state and 32/16-bit gradients/weights.
    Uses AdamW formulation if weight decay > 0.0.

    Parameters
    ----------
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Adam state 1.
    state2 : torch.Tensor
        Adam state 2.
    beta1 : float
        Adam beta1.
    beta2 : float
        Adam beta2.
    eps : float
        Adam epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    is_sparse : bool
        If the gradient can be sparse or not.
    qmap1 : torch.Tensor
        Quantization map for first Adam state.
    qmap2 : torch.Tensor
        Quantization map for second Adam state.
    max1 : torch.Tensor
        Max value for first Adam state update.
    max2 : torch.Tensor
        Max value for second Adam state update.
    new_max1 : torch.Tensor
        Max value for the next Adam update of the first state.
    new_max2 : torch.Tensor
        Max value for the next Adam update of the second state.
    '''

    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        lib.coptimizer_static_8bit_2state_g32(get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr),
                    get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(max1), get_ptr(max2), get_ptr(new_max1), get_ptr(new_max2),
                    ct.c_float(weight_decay),ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        lib.coptimizer_static_8bit_2state_g16(get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr),
                    get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(max1), get_ptr(max2), get_ptr(new_max1), get_ptr(new_max2),
                    ct.c_float(weight_decay),ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')
