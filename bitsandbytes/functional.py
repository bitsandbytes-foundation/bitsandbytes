import torch
import os
import ctypes as ct

torch.nn.utils.clip_grad_norm_
lib = ct.cdll.LoadLibrary(os.path.dirname(__file__) + '/libBitsNBytes.so')
name2qmap = {}

str2optimizer32bit = {}
str2optimizer32bit['adam'] = (lib.cadam32bit_g32, lib.cadam32bit_g16)
str2optimizer32bit['momentum'] = (lib.cmomentum32bit_g32, lib.cmomentum32bit_g16)
str2optimizer32bit['rmsprop'] = (lib.crmsprop32bit_g32, lib.crmsprop32bit_g16)
str2optimizer32bit['lars'] = (lib.cmomentum32bit_g32, lib.cmomentum32bit_g16)
str2optimizer32bit['lamb'] = (lib.cadam32bit_g32, lib.cadam32bit_g16)

str2optimizer8bit = {}
str2optimizer8bit['adam'] = (lib.cadam_static_8bit_g32, lib.cadam_static_8bit_g16)
str2optimizer8bit['momentum'] = (lib.cmomentum_static_8bit_g32, lib.cmomentum_static_8bit_g16)
str2optimizer8bit['rmsprop'] = (lib.crmsprop_static_8bit_g32, lib.crmsprop_static_8bit_g16)
str2optimizer8bit['lamb'] = (lib.cadam_static_8bit_g32, lib.cadam_static_8bit_g16)
str2optimizer8bit['lars'] = (lib.cmomentum_static_8bit_g32, lib.cmomentum_static_8bit_g16)

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
    if A is None: return None
    else: return ct.c_void_p(A.data.storage().data_ptr())

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

def quantize_blockwise(A: torch.Tensor, code: torch.Tensor=None, absmax: torch.Tensor=None, out: torch.Tensor=None) -> torch.Tensor:
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
    out : torch.Tensor
        The output tensor (8-bit).

    Returns
    -------
    torch.Tensor:
        The 8-bit tensor.
    '''

    if code is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    if absmax is None:
        n = A.numel()
        blocks = n//4096
        blocks += 1 if n % 4096 > 0 else 0
        absmax = torch.zeros((blocks,), device=A.device)

    if out is None: out = torch.zeros_like(A, dtype=torch.uint8)

    if A.dtype == torch.float32:
        lib.cquantize_blockwise_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))
    elif A.dtype == torch.float16:
        lib.cquantize_blockwise_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))
    else:
        raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')

    return absmax, out

def dequantize_blockwise(absmax: torch.Tensor, A: torch.Tensor, code: torch.Tensor=None, out: torch.Tensor=None) -> torch.Tensor:
    '''
    Dequantizes blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in
    blocks of size 4096.

    Parameters
    ----------
    absmax : torch.Tensor
        The absmax values.
    A : torch.Tensor
        The input 8-bit tensor.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        Dequantized output tensor (default: float32)


    Returns
    -------
    torch.Tensor:
        Dequantized tensor (default: float32)
    '''
    if code is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    if out is None: out = torch.zeros_like(A, dtype=torch.float32)

    if out.dtype == torch.float32:
        lib.cdequantize_blockwise_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))
    elif out.dtype == torch.float16:
        lib.cdequantize_blockwise_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))
    else:
        raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')

    return out


def quantize(A: torch.Tensor, code: torch.Tensor=None, out: torch.Tensor=None) -> torch.Tensor:
    if code is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    absmax = torch.abs(A).max()
    inp = A/absmax
    out = quantize_no_absmax(code, inp, out)
    return absmax, out

def dequantize(absmax:torch.Tensor, A: torch.Tensor, code: torch.Tensor=None, out: torch.Tensor=None) -> torch.Tensor:
    if code is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    out = dequantize_no_absmax(code, A, out)
    return out*absmax

def quantize_no_absmax(code: torch.Tensor, A: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
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

def dequantize_no_absmax(code: torch.Tensor, A: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
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



def momentum_update_32bit(g: torch.Tensor, p: torch.Tensor, state1: torch.Tensor, weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool,
                          step: int, is_sparse: bool, gnorm_scale: float):
    '''
    Performance inplace SGD Momentum update.

    Parameters
    ----------
    g : torch.Tensor
        The gradient
    p : torch.Tensor
        The paramter/weight tensor.
    state1 : torch.Tensor
        The momentum buffer/optimizer state.
    weight_decay : float
        Weight decay / L2 penalty value.
    momentum : float
        Momentum value.
    lr : float
        The learning rate
    dampening : float
        Dampening constant.
    nesterov : bool
        Whether to use nesterov momentum.
    step : int
        Current optimizer step.
    is_sparse : bool
        If the gradient can be sparse or not.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    '''
    optimizer_update_32bit('momentum', g, p, state1, momentum, 0.0, step, lr, None, dampening, weight_decay, is_sparse, gnorm_scale)

def adam_update_32bit(g: torch.Tensor, p: torch.Tensor, state1: torch.Tensor, state2: torch.Tensor,
                beta1: float, beta2: float, eps: float,
                step: int, lr: float, weight_decay: float=0.0, is_sparse: bool = False, gnorm_scale: float=1.0) -> None:
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
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    '''
    optimizer_update_32bit('adam', g, p, state1, beta1, eps, step, lr, state2, beta2, weight_decay, is_sparse, gnorm_scale)


def optimizer_update_32bit(optimizer_name:str, g: torch.Tensor, p: torch.Tensor, state1: torch.Tensor,
                beta1: float, eps: float, step: int, lr: float,
                state2: torch.Tensor=None, beta2: float=0.0,
                weight_decay: float=0.0, is_sparse: bool = False, gnorm_scale: float=1.0,
                unorm_vec: torch.Tensor=None, max_unorm: float=0.0) -> None:
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
    is_sparse : bool
        If the gradient can be sparse or not.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    '''

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    if optimizer_name not in str2optimizer32bit:
        raise NotImplementError(f'Optimizer not implemented: {optimizer_name}. Choices: {",".join(str2optimizer32bit.keys())}')

    if g.dtype == torch.float32 and state1.dtype == torch.float32:
        str2optimizer32bit[optimizer_name][0](get_ptr(g), get_ptr(p), get_ptr(state1), get_ptr(state2), get_ptr(unorm_vec), ct.c_float(max_unorm),
                    ct.c_float(param_norm), ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_float(weight_decay),
                    ct.c_int32(step), ct.c_float(lr), ct.c_bool(is_sparse), ct.c_float(gnorm_scale), ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.float32:
        str2optimizer32bit[optimizer_name][1](get_ptr(g), get_ptr(p), get_ptr(state1), get_ptr(state2), get_ptr(unorm_vec), ct.c_float(max_unorm),
                    ct.c_float(param_norm), ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_float(weight_decay),
                    ct.c_int32(step), ct.c_float(lr), ct.c_bool(is_sparse), ct.c_float(gnorm_scale), ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')

def adam_update_8bit(g: torch.Tensor, p: torch.Tensor, state1: torch.Tensor, state2: torch.Tensor,
                beta1: float, beta2: float, eps: float,
                step: int, lr: float, qmap1: torch.Tensor, qmap2: torch.Tensor,
                max1: torch.Tensor, max2: torch.Tensor, new_max1: torch.Tensor, new_max2: torch.Tensor,
                weight_decay: float=0.0, is_sparse: bool=False, gnorm_scale: float=1.0,
                unorm_vec: torch.Tensor=None, max_unorm: float=0.0) -> None:
    optimizer_update_8bit('adam', g, p, state1, state2, beta1, beta2, eps, step, lr, qmap1, qmap2, max1, max2, new_max1, new_max2, weight_decay, is_sparse, gnorm_scale, unorm_vec, max_unorm)


def optimizer_update_8bit(optimizer_name: str, g: torch.Tensor, p: torch.Tensor, state1: torch.Tensor, state2: torch.Tensor,
                beta1: float, beta2: float, eps: float,
                step: int, lr: float, qmap1: torch.Tensor, qmap2: torch.Tensor,
                max1: torch.Tensor, max2: torch.Tensor, new_max1: torch.Tensor, new_max2: torch.Tensor,
                weight_decay: float=0.0, is_sparse: bool=False, gnorm_scale: float=1.0,
                unorm_vec: torch.Tensor=None, max_unorm: float=0.0) -> None:
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
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
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


def optimizer_update_8bit_blockwise(optimizer_name: str, g: torch.Tensor, p: torch.Tensor, state1: torch.Tensor, state2: torch.Tensor,
                beta1: float, beta2: float, eps: float,
                step: int, lr: float, qmap1: torch.Tensor, qmap2: torch.Tensor,
                absmax1: torch.Tensor, absmax2: torch.Tensor, weight_decay: float=0.0, is_sparse: bool=False, gnorm_scale: float=1.0) -> None:

    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        lib.coptimizer_static_8bit_blockwise_fp32(get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr), get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(absmax1), get_ptr(absmax2), ct.c_float(weight_decay), ct.c_float(gnorm_scale), ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        lib.coptimizer_static_8bit_blockwise_fp16(get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr), get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(absmax1), get_ptr(absmax2), ct.c_float(weight_decay), ct.c_float(gnorm_scale), ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')


def percentile_clipping(grad: torch.Tensor, gnorm_vec: torch.Tensor, step: int, percentile: int=5):
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
