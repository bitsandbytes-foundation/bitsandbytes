# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import os
import random
import math
import ctypes as ct
import torch
import numpy as np

from torch import Tensor
from typing import Tuple

lib = ct.cdll.LoadLibrary(os.path.dirname(__file__) + '/libbitsandbytes.so')
lib.cget_managed_ptr_fp32.restype = ct.c_void_p
name2qmap = {}

''' C FUNCTIONS FOR OPTIMIZERS '''

str2optimizer32bit = {}
str2optimizer32bit['adam'] = (lib.cadam32bit_g32, lib.cadam32bit_g16)
str2optimizer32bit['momentum'] = (lib.cmomentum32bit_g32, lib.cmomentum32bit_g16)
str2optimizer32bit['rmsprop'] = (lib.crmsprop32bit_g32, lib.crmsprop32bit_g16)
str2optimizer32bit['adagrad'] = (lib.cadagrad32bit_g32, lib.cadagrad32bit_g16)
str2optimizer32bit['lars'] = (lib.cmomentum32bit_g32, lib.cmomentum32bit_g16)
str2optimizer32bit['lamb'] = (lib.cadam32bit_g32, lib.cadam32bit_g16)

str2optimizer8bit_blockwise = {}
str2optimizer8bit_blockwise['adam'] = (lib.cadam_8bit_blockwise_fp32, lib.cadam_8bit_blockwise_fp16)
str2optimizer8bit_blockwise['momentum'] = (lib.cmomentum_8bit_blockwise_fp32, lib.cmomentum_8bit_blockwise_fp16)
str2optimizer8bit_blockwise['rmsprop'] = (lib.crmsprop_8bit_blockwise_fp32, lib.crmsprop_8bit_blockwise_fp16)
str2optimizer8bit_blockwise['adagrad'] = (lib.cadagrad_8bit_blockwise_fp32, lib.cadagrad_8bit_blockwise_fp16)

def get_managed(rows, cols, dtype=torch.float32):
    size = 0
    nptype = np.float32
    if dtype == torch.float32:
        size = 4
        nptype = np.float32
    elif dtype == torch.uint8:
        size = 1
        nptype = np.uint8
    assert size > 0

    ptr = lib.cget_managed_ptr_fp32(ct.c_int64(rows), ct.c_int64(cols), ct.c_int64(size))
    cptr = ct.cast(ptr, ct.POINTER(ct.c_int))
    new_array = np.ctypeslib.as_array(cptr, shape=(rows,cols))
    new_a = np.frombuffer(new_array, dtype=nptype, count=rows*cols)
    torch_arr = torch.from_numpy(new_a).view(rows, cols)
    torch_arr.is_managed = True
    return torch_arr


def create_linear_map(signed=True):
    if signed:
        return torch.linspace(-1.0, 1.0, 256)
    else:
        return torch.linspace(0.0, 1.0, 256)

def create_dynamic_map(signed=True, n=7):
    '''
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    '''

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    additional_items = 2**(7-n)-1
    if not signed: additional_items = 2*additional_items
    for i in range(n):
        fraction_items = 2**(i+7-n)+1 if signed else 2**(i+7-n+1)+1
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1]+boundaries[1:])/2.0
        data += ((10**(-(n-1)+i))*means).tolist()
        if signed:
            data += (-(10**(-(n-1)+i))*means).tolist()

    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items+1)
        means = (boundaries[:-1]+boundaries[1:])/2.0
        data += ((10**(-(n-1)+i))*means).tolist()
        if signed:
            data += (-(10**(-(n-1)+i))*means).tolist()

    data.append(0)
    data.append(1.0)
    data.sort()
    return Tensor(data)

def get_ptr(A: Tensor) -> ct.c_void_p:
    '''
    Get the ctypes pointer from a PyTorch Tensor.

    Parameters
    ----------
    A : torch.tensor
        The PyTorch tensor.

    Returns
    -------
    ctypes.c_void_p
    '''
    if A is None: return None
    else: return ct.c_void_p(A.data.storage().data_ptr())

def estimate_quantiles(A: Tensor, out: Tensor=None, offset: float=1/512) -> Tensor:
    '''
    Estimates 256 equidistant quantiles on the input tensor eCDF.

    Uses SRAM-Quantiles algorithm to quickly estimate 256 equidistant quantiles
    via the eCDF of the input tensor `A`. This is a fast but approximate algorithm
    and the extreme quantiles close to 0 and 1 have high variance / large estimation
    errors. These large errors can be avoided by using the offset variable which trims
    the distribution. The default offset value of 1/512 ensures minimum entropy encoding -- it
    trims 1/512 = 0.2% from each side of the distrivution. An offset value of 0.01 to 0.02
    usually has a much lower error but is not a minimum entropy encoding. Given an offset
    of 0.02 equidistance points in the range [0.02, 0.98] are used for the quantiles.

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

def quantize_blockwise(A: Tensor, code: Tensor=None, absmax: Tensor=None, rand=None, out: Tensor=None, is_managed=False) -> Tensor:
    '''
    Quantize tensor A in blocks of size 4096 values.

    Quantizes tensor A by dividing it into blocks of 4096 values.
    Then the absolute maximum value within these blocks is calculated
    for the non-linear quantization.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization map.
    absmax : torch.Tensor
        The absmax values.
    rand : torch.Tensor
        The tensor for stochastic rounding.
    out : torch.Tensor
        The output tensor (8-bit).

    Returns
    -------
    torch.Tensor:
        The 8-bit tensor.
    tuple(torch.Tensor, torch.Tensor):
        The quantization state to undo the quantization.
    '''

    device = (A.device if not is_managed else torch.device('cuda'))

    if code is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(device)
        code = name2qmap['dynamic']
        code = code.to(device)

    if absmax is None:
        n = A.numel()
        num_blocks = 4096
        blocks = n//num_blocks
        blocks += 1 if n % num_blocks > 0 else 0
        absmax = torch.zeros((blocks,), device=device)

    if out is None: out = torch.zeros_like(A, dtype=torch.uint8, device=device)


    if device.type != 'cpu':
        if rand is not None:
            assert rand.numel() >= 1024
            rand_offset = random.randint(0, 1023)
            if A.dtype == torch.float32:
                lib.cquantize_blockwise_stochastic_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), get_ptr(rand), ct.c_int32(rand_offset), ct.c_int(A.numel()))
            elif A.dtype == torch.float16:
                lib.cquantize_blockwise_stochastic_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), get_ptr(rand), ct.c_int32(rand_offset), ct.c_int(A.numel()))
            else:
                raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')
        else:
            if A.dtype == torch.float32:
                lib.cquantize_blockwise_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))
            elif A.dtype == torch.float16:
                lib.cquantize_blockwise_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))
            else:
                raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')
    else:
        # cpu
        assert rand is None
        lib.cquantize_blockwise_cpu_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))

    return out, (absmax, code)

def dequantize_blockwise(A: Tensor, quant_state: Tuple[Tensor, Tensor]=None,
                         absmax: Tensor=None, code: Tensor=None, out: Tensor=None,
                         blocksize: int=4096) -> Tensor:
    '''
    Dequantizes blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in
    blocks of size 4096.

    Parameters
    ----------
    A : torch.Tensor
        The input 8-bit tensor.
    quant_state : tuple(torch.Tensor, torch.Tensor)
        Tuple of code and absmax values. 
    absmax : torch.Tensor
        The absmax values.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        Dequantized output tensor (default: float32)


    Returns
    -------
    torch.Tensor:
        Dequantized tensor (default: float32)
    '''
    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    if out is None: out = torch.zeros_like(A, dtype=torch.float32)
    if quant_state is None: quant_state = (absmax, code)

    if blocksize not in [2048, 4096]:
        raise ValueError(f'The blockwise of {blocksize} is not supported. Supported values: [2048 4096]')

    if A.device.type != 'cpu':
        if out.dtype == torch.float32:
            lib.cdequantize_blockwise_fp32(get_ptr(quant_state[1]), get_ptr(A), get_ptr(quant_state[0]), get_ptr(out), ct.c_int(blocksize), ct.c_int(A.numel()))
        elif out.dtype == torch.float16:
            lib.cdequantize_blockwise_fp16(get_ptr(quant_state[1]), get_ptr(A), get_ptr(quant_state[0]), get_ptr(out), ct.c_int(blocksize), ct.c_int(A.numel()))
        else:
            raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')
    else:
        lib.cdequantize_blockwise_cpu_fp32(get_ptr(quant_state[1]), get_ptr(A), get_ptr(quant_state[0]), get_ptr(out), ct.c_int(A.numel()))


    return out


def quantize(A: Tensor, code: Tensor=None, out: Tensor=None) -> Tensor:
    if code is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    absmax = torch.abs(A).max()
    inp = A/absmax
    out = quantize_no_absmax(inp, code, out)
    return out, (absmax, code)

def dequantize(A: Tensor, quant_state: Tuple[Tensor, Tensor]=None, absmax: Tensor=None, code: Tensor=None, out: Tensor=None) -> Tensor:
    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    if quant_state is None: quant_state = (absmax, code)
    out = dequantize_no_absmax(A, quant_state[1], out)
    return out*quant_state[0]

def quantize_no_absmax(A: Tensor, code: Tensor, out: Tensor=None) -> Tensor:
    '''
    Quantizes input tensor to 8-bit.

    Quantizes the 32-bit input tensor `A` to the 8-bit output tensor
    `out` using the quantization map `code`.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization map.
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

def dequantize_no_absmax(A: Tensor, code: Tensor, out: Tensor=None) -> Tensor:
    '''
    Dequantizes the 8-bit tensor to 32-bit.

    Dequantizes the 8-bit tensor `A` to the 32-bit tensor `out` via
    the quantization map `code`.

    Parameters
    ----------
    A : torch.Tensor
        The 8-bit input tensor.
    code : torch.Tensor
        The quantization map.
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

def optimizer_update_32bit(optimizer_name:str, g: Tensor, p: Tensor, state1: Tensor,
                beta1: float, eps: float, step: int, lr: float,
                state2: Tensor=None, beta2: float=0.0,
                weight_decay: float=0.0, gnorm_scale: float=1.0,
                unorm_vec: Tensor=None, max_unorm: float=0.0, skip_zeros=False) -> None:
    '''
    Performs an inplace optimizer update with one or two optimizer states.

    Universal optimizer update for 32-bit state and 32/16-bit gradients/weights.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer: {adam}.
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Optimizer state 1.
    beta1 : float
        Optimizer beta1.
    eps : float
        Optimizer epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    state2 : torch.Tensor
        Optimizer state 2.
    beta2 : float
        Optimizer beta2.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    skip_zeros : bool
        Whether to skip zero-valued gradients or not (default: False).
    '''

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    if optimizer_name not in str2optimizer32bit:
        raise NotImplementError(f'Optimizer not implemented: {optimizer_name}. Choices: {",".join(str2optimizer32bit.keys())}')

    if g.dtype == torch.float32 and state1.dtype == torch.float32:
        str2optimizer32bit[optimizer_name][0](get_ptr(g), get_ptr(p), get_ptr(state1), get_ptr(state2), get_ptr(unorm_vec), ct.c_float(max_unorm),
                    ct.c_float(param_norm), ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_float(weight_decay),
                    ct.c_int32(step), ct.c_float(lr), ct.c_float(gnorm_scale), ct.c_bool(skip_zeros), ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.float32:
        str2optimizer32bit[optimizer_name][1](get_ptr(g), get_ptr(p), get_ptr(state1), get_ptr(state2), get_ptr(unorm_vec), ct.c_float(max_unorm),
                    ct.c_float(param_norm), ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_float(weight_decay),
                    ct.c_int32(step), ct.c_float(lr), ct.c_float(gnorm_scale), ct.c_bool(skip_zeros), ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')

def optimizer_update_8bit(optimizer_name: str, g: Tensor, p: Tensor, state1: Tensor, state2: Tensor,
                beta1: float, beta2: float, eps: float,
                step: int, lr: float, qmap1: Tensor, qmap2: Tensor,
                max1: Tensor, max2: Tensor, new_max1: Tensor, new_max2: Tensor,
                weight_decay: float=0.0, gnorm_scale: float=1.0,
                unorm_vec: Tensor=None, max_unorm: float=0.0) -> None:
    '''
    Performs an inplace Adam update.

    Universal Adam update for 32/8-bit state and 32/16-bit gradients/weights.
    Uses AdamW formulation if weight decay > 0.0.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer. Choices {adam, momentum}
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
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    '''

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][0](get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    get_ptr(unorm_vec), ct.c_float(max_unorm), ct.c_float(param_norm),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr),
                    get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(max1), get_ptr(max2), get_ptr(new_max1), get_ptr(new_max2),
                    ct.c_float(weight_decay),ct.c_float(gnorm_scale), ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][1](get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    get_ptr(unorm_vec), ct.c_float(max_unorm), ct.c_float(param_norm),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr),
                    get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(max1), get_ptr(max2), get_ptr(new_max1), get_ptr(new_max2),
                    ct.c_float(weight_decay),ct.c_float(gnorm_scale), ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')


def optimizer_update_8bit_blockwise(optimizer_name: str, g: Tensor, p: Tensor, state1: Tensor, state2: Tensor,
                beta1: float, beta2: float, eps: float,
                step: int, lr: float, qmap1: Tensor, qmap2: Tensor,
                absmax1: Tensor, absmax2: Tensor, weight_decay: float=0.0, gnorm_scale: float=1.0,
                skip_zeros=False) -> None:


    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        str2optimizer8bit_blockwise[optimizer_name][0](get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr), get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(absmax1), get_ptr(absmax2), ct.c_float(weight_decay), ct.c_float(gnorm_scale),
                    ct.c_bool(skip_zeros), ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        str2optimizer8bit_blockwise[optimizer_name][1](get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr), get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(absmax1), get_ptr(absmax2), ct.c_float(weight_decay), ct.c_float(gnorm_scale),
                    ct.c_bool(skip_zeros), ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')


def percentile_clipping(grad: Tensor, gnorm_vec: Tensor, step: int, percentile: int=5):
    """Applies percentile clipping

    grad: torch.Tensor
        The gradient tensor.
    gnorm_vec: torch.Tensor
        Vector of gradient norms. 100 elements expected.
    step: int
        The current optimiation steps (number of past gradient norms).

    """
    if grad.dtype == torch.float32:
        lib.cpercentile_clipping_g32(get_ptr(grad), get_ptr(gnorm_vec), ct.c_int32(step), ct.c_int32(grad.numel()))
    elif grad.dtype == torch.float16:
        lib.cpercentile_clipping_g16(get_ptr(grad), get_ptr(gnorm_vec), ct.c_int32(step), ct.c_int32(grad.numel()))
    else:
        raise ValueError(f'Gradient type {grad.dtype} not supported!')

    current_gnorm = torch.sqrt(gnorm_vec[step % 100])
    vals, idx = torch.sort(gnorm_vec)
    clip_value = torch.sqrt(vals[percentile])
    gnorm_scale = 1.0

    if current_gnorm > clip_value:
        gnorm_scale = clip_value/current_gnorm

    return current_gnorm, clip_value, gnorm_scale


def histogram_scatter_add_2d(histogram: Tensor, index1: Tensor, index2: Tensor, source: Tensor):
    assert len(histogram.shape) == 2
    assert histogram.dtype == torch.float32
    assert source.dtype == torch.float32
    assert index1.dtype == torch.int32
    assert index2.dtype == torch.int32

    assert histogram.device.type == 'cuda'
    assert index1.device.type == 'cuda'
    assert index2.device.type == 'cuda'
    assert source.device.type == 'cuda'

    maxdim1 = ct.c_int32(histogram.shape[0])
    n = ct.c_int32(index1.numel())
    lib.chistogram_scatter_add_2d(get_ptr(histogram), get_ptr(index1), get_ptr(index2), get_ptr(source), maxdim1, n)


type2size = {}
type2size[torch.float32] = 4
type2size[torch.uint8] = 1
def prefetch(A, deviceid=0):
    assert A.is_managed
    size = type2size[A.dtype]
    lib.cprefetch(get_ptr(A), ct.c_int64(A.numel()), ct.c_int64(size), ct.c_int32(deviceid))

def prefetch_cpu(A):
    prefetch(A, -1)


def elementwise_func(func_name, A, value, device=None):
    if A.dtype == torch.float32:
        func = getattr(lib, f'c{func_name}_fp32', None)
        cvalue = ct.c_float(value)
    elif A.dtype == torch.uint8:
        func = getattr(lib, f'c{func_name}_uint8', None)
        cvalue = ct.c_uint8(value)

    if func is None: raise NotImplementError(f'Function not implemented: {func_name}')

    if device is None: device_idx = torch.cuda.current_device()
    else: device_idx = device.index

    is_managed = getattr(A, 'is_managed', False)
    if is_managed: prefetch(A, device_idx)

    func(get_ptr(A), cvalue, ct.c_int64(A.numel()))

def fill(A, value, device=None): elementwise_func('fill', A, value, device)
def arange(A, device=None): elementwise_func('arange', A, 0, device)


def quantize_blockwise_dynamic(A: Tensor, absmax: Tensor=None, out: Tensor=None) -> Tensor:
    '''
    Quantize tensor A in blocks of size 4096 values.

    Quantizes tensor A by dividing it into blocks of 4096 values.
    Then the absolute maximum value within these blocks is calculated
    for the non-linear quantization.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        The output tensor (8-bit).

    Returns
    -------
    torch.Tensor:
        The 8-bit tensor.
    tuple(torch.Tensor, torch.Tensor):
        The quantization state to undo the quantization.
    '''

    assert A.device.type == 'cuda'
    is_managed = getattr(A, 'is_managed', False)
    device = (A.device if not is_managed else torch.device('cuda'))

    if absmax is None:
        n = A.numel()
        block_size = 2048
        #block_size = 4096
        blocks = n//block_size
        blocks += 1 if n % block_size > 0 else 0
        absmax = torch.zeros((blocks,), device=device)

    if out is None: out = torch.zeros_like(A, dtype=torch.uint8, device=device)

    if A.dtype == torch.float32:
        lib.cquantize_blockwise_dynamic_fp32_2048b(get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))
        #lib.cquantize_blockwise_dynamic_fp32_4096b(get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))
    else:
        raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')

    return out, absmax


def dequantize_blockwise_dynamic(A: Tensor, absmax: Tensor=None, out: Tensor=None,
                         blocksize: int=2048) -> Tensor:
    '''
    Dequantizes blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in
    blocks of size 4096.

    Parameters
    ----------
    A : torch.Tensor
        The input 8-bit tensor.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        Dequantized output tensor (default: float32)


    Returns
    -------
    torch.Tensor:
        Dequantized tensor (default: float32)
    '''
    if out is None: out = torch.zeros_like(A, dtype=torch.float32)

    if blocksize not in [2048, 4096]:
        raise ValueError(f'The blockwise of {blocksize} is not supported. Supported values: [2048 4096]')

    if out.dtype == torch.float32:
        lib.cdequantize_blockwise_dynamic_fp32_2048b(get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))


    return out
