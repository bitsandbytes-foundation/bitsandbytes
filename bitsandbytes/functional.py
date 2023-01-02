# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import ctypes as ct
import itertools
import operator
import random
import torch
import itertools
import math

from functools import reduce  # Required in Python 3
from typing import Tuple
from torch import Tensor

from .cextension import COMPILED_WITH_CUDA, lib


# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

name2qmap = {}

if COMPILED_WITH_CUDA:
    """C FUNCTIONS FOR OPTIMIZERS"""
    str2optimizer32bit = {}
    str2optimizer32bit["adam"] = (lib.cadam32bit_g32, lib.cadam32bit_g16)
    str2optimizer32bit["momentum"] = (
        lib.cmomentum32bit_g32,
        lib.cmomentum32bit_g16,
    )
    str2optimizer32bit["rmsprop"] = (
        lib.crmsprop32bit_g32,
        lib.crmsprop32bit_g16,
    )
    str2optimizer32bit["adagrad"] = (
        lib.cadagrad32bit_g32,
        lib.cadagrad32bit_g16,
    )
    str2optimizer32bit["lars"] = (
        lib.cmomentum32bit_g32,
        lib.cmomentum32bit_g16,
    )
    str2optimizer32bit["lamb"] = (lib.cadam32bit_g32, lib.cadam32bit_g16)

    str2optimizer8bit = {}
    str2optimizer8bit["adam"] = (
        lib.cadam_static_8bit_g32,
        lib.cadam_static_8bit_g16,
    )
    str2optimizer8bit["momentum"] = (
        lib.cmomentum_static_8bit_g32,
        lib.cmomentum_static_8bit_g16,
    )
    str2optimizer8bit["rmsprop"] = (
        lib.crmsprop_static_8bit_g32,
        lib.crmsprop_static_8bit_g16,
    )
    str2optimizer8bit["lamb"] = (
        lib.cadam_static_8bit_g32,
        lib.cadam_static_8bit_g16,
    )
    str2optimizer8bit["lars"] = (
        lib.cmomentum_static_8bit_g32,
        lib.cmomentum_static_8bit_g16,
    )

    str2optimizer8bit_blockwise = {}
    str2optimizer8bit_blockwise["adam"] = (
        lib.cadam_8bit_blockwise_fp32,
        lib.cadam_8bit_blockwise_fp16,
    )
    str2optimizer8bit_blockwise["momentum"] = (
        lib.cmomentum_8bit_blockwise_fp32,
        lib.cmomentum_8bit_blockwise_fp16,
    )
    str2optimizer8bit_blockwise["rmsprop"] = (
        lib.crmsprop_8bit_blockwise_fp32,
        lib.crmsprop_8bit_blockwise_fp16,
    )
    str2optimizer8bit_blockwise["adagrad"] = (
        lib.cadagrad_8bit_blockwise_fp32,
        lib.cadagrad_8bit_blockwise_fp16,
    )


class CUBLAS_Context:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.context = {}
        # prev_device = torch.cuda.current_device()
        # for i in range(torch.cuda.device_count()):
        #    torch.cuda.set_device(torch.device('cuda', i))
        #    self.context.append(ct.c_void_p(lib.get_context()))
        # torch.cuda.set_device(prev_device)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def get_context(self, device):
        if device.index not in self.context:
            prev_device = torch.cuda.current_device()
            torch.cuda.set_device(device)
            self.context[device.index] = ct.c_void_p(lib.get_context())
            torch.cuda.set_device(prev_device)
        return self.context[device.index]


class Cusparse_Context:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.context = ct.c_void_p(lib.get_cusparse())

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


def create_linear_map(signed=True, total_bits=8, add_zero=True):
    sign = (-1.0 if signed else 0.0)
    total_values = 2**total_bits
    if add_zero or total_bits < 8:
        # add a zero
        # since we simulate less bits by having zeros in the data type, we
        # we need to center the quantization around zero and as such lose
        # a single value
        total_values = (2**total_bits if not signed else 2**total_bits-1)

    values = torch.linspace(sign, 1.0, total_values)
    gap = 256 - values.numel()
    if gap == 0:
        return values
    else:
        l = values.numel()//2
        #return torch.Tensor(values[:l].tolist() + [-1e-6]*((gap//2)-1) + [0]*2 + [1e-6]*((gap//2)-1) + values[l:].tolist())
        return torch.Tensor(values[:l].tolist() + [0]*gap + values[l:].tolist())


def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e+p == total_bits-has_sign
    # the exponent is biased to 2^(e-1) -1 == 0
    evalues = []
    pvalues = []
    for i, val in enumerate(range(-((2**(exponent_bits-has_sign))), 2**(exponent_bits-has_sign), 1)):
        evalues.append(2**val)


    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    #for ev in evalues:
    bias = 2**(exponent_bits-1)-1
    for evalue in range(2**(exponent_bits)):
        for bit_pattern in lst:
            value = (1 if evalue != 0 else 0)
            for i, pval in enumerate(list(bit_pattern)):
                value += pval*(2**-(i+1))
            if evalue == 0:
                # subnormals
                value = value*2**-(bias-1)
            else:
                # normals
                value = value*2**-(evalue-bias-2)
            values.append(value)
            if signed:
                values.append(-value)


    assert len(values) == 2**total_bits
    values.sort()
    if total_bits < 8:
        gap = 256 - len(values)
        for i in range(gap):
            values.append(0)
    values.sort()
    code = torch.Tensor(values)
    code /= code.max()

    return code



def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
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
    """

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - (1 if signed else 0)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    if not signed:
        additional_items = 2 * additional_items
    for i in range(max_exponent_bits):
        fraction_items = int((2 ** (i + non_sign_bits - max_exponent_bits) + 1 if signed else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1))
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

        if additional_items > 0:
            boundaries = torch.linspace(0.1, 1, additional_items + 1)
            means = (boundaries[:-1] + boundaries[1:]) / 2.0
            data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
            if signed:
                data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    data.sort()
    return Tensor(data)

def create_quantile_map(A, total_bits=8):
    q = estimate_quantiles(A, num_quantiles=2**total_bits-1)
    q = q.tolist()
    q.append(0)

    gap = 256 - len(q)
    for i in range(gap):
        q.append(0)

    q.sort()

    q = Tensor(q)
    q = q/q.abs().max()
    return q

def get_special_format_str():
    if not torch.cuda.is_available(): return 'col_turing'
    major, _minor = torch.cuda.get_device_capability()
    if major <= 7:
        return "col_turing"
    if major == 8:
        return "col_ampere"
    return "col_turing"



def is_on_gpu(tensors):
    on_gpu = True
    for t in tensors:
        if t is None: continue # NULL pointers are fine
        on_gpu &= t.device.type == 'cuda'
    return on_gpu

def get_ptr(A: Tensor) -> ct.c_void_p:
    """
    Get the ctypes pointer from a PyTorch Tensor.

    Parameters
    ----------
    A : torch.tensor
        The PyTorch tensor.

    Returns
    -------
    ctypes.c_void_p
    """
    if A is None:
        return None
    else:
        return ct.c_void_p(A.data.data_ptr())


def pre_call(device):
    prev_device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    return prev_device


def post_call(prev_device):
    torch.cuda.set_device(prev_device)


def get_transform_func(dtype, orderA, orderOut, transpose=False):
    name = f'ctransform_{(8 if dtype == torch.int8 else 32)}_{orderA}_to_{orderOut}_{"t" if transpose else "n"}'
    if not hasattr(lib, name):
        print(name)
        raise ValueError(
            f"Transform function not supported: {orderA} to {orderOut} for data type {dtype} and transpose={transpose}"
        )
    else:
        return getattr(lib, name)


def get_transform_buffer(
    shape, dtype, device, to_order, from_order="row", transpose=False
):
    # init_func = torch.empty
    init_func = torch.zeros
    dims = len(shape)

    if dims == 2:
        rows = shape[0]
    elif dims == 3:
        rows = shape[0] * shape[1]
    cols = shape[-1]

    state = (shape, to_order)
    if transpose:
        # swap dims
        tmp = rows
        rows = cols
        cols = tmp
        state = (shape[::-1], to_order)

    if to_order == "row" or to_order == "col":
        return init_func(shape, dtype=dtype, device=device), state
    elif to_order == "col32":
        # blocks of 32 columns (padded)
        cols = 32 * ((cols + 31) // 32)
        return init_func((rows, cols), dtype=dtype, device=device), state
    elif to_order == "col_turing":
        # blocks of 32 columns and 8 rows
        cols = 32 * ((cols + 31) // 32)
        rows = 8 * ((rows + 7) // 8)
        return init_func((rows, cols), dtype=dtype, device=device), state
    elif to_order == "col_ampere":
        # blocks of 32 columns and 32 rows
        cols = 32 * ((cols + 31) // 32)
        rows = 32 * ((rows + 31) // 32)
        return init_func((rows, cols), dtype=dtype, device=device), state
    else:
        raise NotImplementedError(f"To_order not supported: {to_order}")


def nvidia_transform(
    A,
    to_order,
    from_order="row",
    out=None,
    transpose=False,
    state=None,
    ld=None,
):
    if state is None:
        state = (A.shape, from_order)
    else:
        from_order = state[1]
    if out is None:
        out, new_state = get_transform_buffer(
            state[0], A.dtype, A.device, to_order, state[1]
        )
    else:
        new_state = (state[1], to_order)
    func = get_transform_func(A.dtype, from_order, to_order, transpose)

    shape = state[0]
    if len(shape) == 2:
        dim1 = ct.c_int32(shape[0])
        dim2 = ct.c_int32(shape[1])
    elif ld is not None:
        n = prod(shape)
        dim1 = prod([shape[i] for i in ld])
        dim2 = ct.c_int32(n // dim1)
        dim1 = ct.c_int32(dim1)
    else:
        dim1 = ct.c_int32(shape[0] * shape[1])
        dim2 = ct.c_int32(shape[2])

    ptr = CUBLAS_Context.get_instance().get_context(A.device)
    func(ptr, get_ptr(A), get_ptr(out), dim1, dim2)

    return out, new_state


def estimate_quantiles(A: Tensor, out: Tensor = None, offset: float = 1 / 512, num_quantiles=256) -> Tensor:
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
        The offset for the first and last quantile from 0 and 1. Default: 1/(2*num_quantiles)
    num_quantiles : int
        The number of equally spaced quantiles.

    Returns
    -------
    torch.Tensor:
        The 256 quantiles in float32 datatype.
    '''
    if A.numel() < 256: raise NotImplementedError(f'Quantile estimation needs at least 256 values in the Tensor, but Tensor had only {A.numel()} values.')
    if num_quantiles > 256: raise NotImplementedError(f"Currently only a maximum of 256 equally spaced quantiles are supported, but the argument num_quantiles={num_quantiles}")
    if num_quantiles < 256 and offset == 1/(512):
        # override default arguments
        offset = 1/(2*num_quantiles)

    if out is None: out = torch.zeros((256,), dtype=torch.float32, device=A.device)
    is_on_gpu([A, out])
    device = pre_call(A.device)
    if A.dtype == torch.float32:
        lib.cestimate_quantiles_fp32(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    elif A.dtype == torch.float16:
        lib.cestimate_quantiles_fp16(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    else:
        raise NotImplementedError(f"Not supported data type {A.dtype}")
    post_call(device)

    if num_quantiles < 256:
        step = round(256/num_quantiles)
        idx = torch.linspace(0, 255, num_quantiles).long().to(A.device)
        out = out[idx]

    return out


def quantize_blockwise(A: Tensor, code: Tensor = None, absmax: Tensor = None, rand=None, out: Tensor = None, blocksize=4096) -> Tensor:
    """
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
    """


    if code is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]

    if absmax is None:
        n = A.numel()
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0
        absmax = torch.zeros((blocks,), device=A.device)

    if out is None:
        out = torch.zeros_like(A, dtype=torch.uint8)

    if A.device.type != 'cpu':
        assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
        cblocksize = ct.c_int32(blocksize)
        prev_device = pre_call(A.device)
        code = code.to(A.device)
        if rand is not None:
            is_on_gpu([code, A, out, absmax, rand])
            assert blocksize==4096
            assert rand.numel() >= 1024
            rand_offset = random.randint(0, 1023)
            if A.dtype == torch.float32:
                lib.cquantize_blockwise_stochastic_fp32(get_ptr(code), get_ptr(A),get_ptr(absmax), get_ptr(out), get_ptr(rand), ct.c_int32(rand_offset), ct.c_int(A.numel()))
            elif A.dtype == torch.float16:
                lib.cquantize_blockwise_stochastic_fp16(get_ptr(code), get_ptr(A),get_ptr(absmax), get_ptr(out), get_ptr(rand), ct.c_int32(rand_offset), ct.c_int(A.numel()))
            else:
                raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
        else:
            is_on_gpu([code, A, out, absmax])
            if A.dtype == torch.float32:
                lib.cquantize_blockwise_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), cblocksize, ct.c_int(A.numel()))
            elif A.dtype == torch.float16:
                lib.cquantize_blockwise_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), cblocksize, ct.c_int(A.numel()))
            else:
                raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
        post_call(A.device)
    else:
        # cpu
        code = code.cpu()
        assert rand is None
        lib.cquantize_blockwise_cpu_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_longlong(blocksize), ct.c_longlong(A.numel()))

    return out, (absmax, code)


def dequantize_blockwise(
    A: Tensor,
    quant_state: Tuple[Tensor, Tensor] = None,
    absmax: Tensor = None,
    code: Tensor = None,
    out: Tensor = None,
    blocksize: int = 4096,
) -> Tensor:
    """
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
    """
    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]

    if out is None:
        out = torch.zeros_like(A, dtype=torch.float32)
    if quant_state is None:
        quant_state = (absmax, code)
    else:
        absmax, code = quant_state


    if A.device.type != 'cpu':
        device = pre_call(A.device)
        code = code.to(A.device)
        if blocksize not in [2048, 4096, 1024, 512, 256, 128, 64]:
            raise ValueError(f"The blockwise of {blocksize} is not supported. Supported values: [2048, 4096, 1024, 512, 256, 128, 64]")
        is_on_gpu([A, out])
        if out.dtype == torch.float32:
            lib.cdequantize_blockwise_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(blocksize), ct.c_int(A.numel()))
        elif out.dtype == torch.float16:
            lib.cdequantize_blockwise_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(blocksize), ct.c_int(A.numel()))
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
        post_call(A.device)
    else:
        code = code.cpu()
        lib.cdequantize_blockwise_cpu_fp32(get_ptr(quant_state[1]), get_ptr(A), get_ptr(quant_state[0]), get_ptr(out), ct.c_longlong(blocksize), ct.c_longlong(A.numel()))

    return out


def quantize(A: Tensor, code: Tensor = None, out: Tensor = None) -> Tensor:
    if code is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
        code = code.to(A.device)

    absmax = torch.abs(A).max()
    inp = A / absmax
    out = quantize_no_absmax(inp, code, out)
    return out, (absmax, code)


def dequantize(
    A: Tensor,
    quant_state: Tuple[Tensor, Tensor] = None,
    absmax: Tensor = None,
    code: Tensor = None,
    out: Tensor = None,
) -> Tensor:
    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
        code = code.to(A.device)

    if quant_state is None:
        quant_state = (absmax, code)
    out = dequantize_no_absmax(A, quant_state[1], out)
    return out * quant_state[0]


def quantize_no_absmax(A: Tensor, code: Tensor, out: Tensor = None) -> Tensor:
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
    is_on_gpu([A, out])
    lib.cquantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    return out


def dequantize_no_absmax(A: Tensor, code: Tensor, out: Tensor = None) -> Tensor:
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
    is_on_gpu([code, A, out])
    lib.cdequantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    return out


def optimizer_update_32bit(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    beta1: float,
    eps: float,
    step: int,
    lr: float,
    state2: Tensor = None,
    beta2: float = 0.0,
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Tensor = None,
    max_unorm: float = 0.0,
    skip_zeros=False,
) -> None:
    """
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
    """

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    if optimizer_name not in str2optimizer32bit:
        raise NotImplementedError(
            f'Optimizer not implemented: {optimizer_name}. Choices: {",".join(str2optimizer32bit.keys())}'
        )

    if g.dtype == torch.float32 and state1.dtype == torch.float32:
        str2optimizer32bit[optimizer_name][0](
            get_ptr(g),
            get_ptr(p),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_float(weight_decay),
            ct.c_int32(step),
            ct.c_float(lr),
            ct.c_float(gnorm_scale),
            ct.c_bool(skip_zeros),
            ct.c_int32(g.numel()),
        )
    elif g.dtype == torch.float16 and state1.dtype == torch.float32:
        str2optimizer32bit[optimizer_name][1](
            get_ptr(g),
            get_ptr(p),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_float(weight_decay),
            ct.c_int32(step),
            ct.c_float(lr),
            ct.c_float(gnorm_scale),
            ct.c_bool(skip_zeros),
            ct.c_int32(g.numel()),
        )
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}"
        )


def optimizer_update_8bit(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    state2: Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: Tensor,
    qmap2: Tensor,
    max1: Tensor,
    max2: Tensor,
    new_max1: Tensor,
    new_max2: Tensor,
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Tensor = None,
    max_unorm: float = 0.0,
) -> None:
    """
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
    """

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][0](
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(max1),
            get_ptr(max2),
            get_ptr(new_max1),
            get_ptr(new_max2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_int32(g.numel()),
        )
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][1](
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(max1),
            get_ptr(max2),
            get_ptr(new_max1),
            get_ptr(new_max2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_int32(g.numel()),
        )
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}"
        )


def optimizer_update_8bit_blockwise(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    state2: Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: Tensor,
    qmap2: Tensor,
    absmax1: Tensor,
    absmax2: Tensor,
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    skip_zeros=False,
) -> None:

    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        str2optimizer8bit_blockwise[optimizer_name][0](
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(absmax1),
            get_ptr(absmax2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_bool(skip_zeros),
            ct.c_int32(g.numel()),
        )
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        str2optimizer8bit_blockwise[optimizer_name][1](
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(absmax1),
            get_ptr(absmax2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_bool(skip_zeros),
            ct.c_int32(g.numel()),
        )
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}"
        )


def percentile_clipping(
    grad: Tensor, gnorm_vec: Tensor, step: int, percentile: int = 5
):
    """Applies percentile clipping

    grad: torch.Tensor
        The gradient tensor.
    gnorm_vec: torch.Tensor
        Vector of gradient norms. 100 elements expected.
    step: int
        The current optimiation steps (number of past gradient norms).

    """
    is_on_gpu([grad, gnorm_vec])
    if grad.dtype == torch.float32:
        lib.cpercentile_clipping_g32(
            get_ptr(grad),
            get_ptr(gnorm_vec),
            ct.c_int32(step),
            ct.c_int32(grad.numel()),
        )
    elif grad.dtype == torch.float16:
        lib.cpercentile_clipping_g16(
            get_ptr(grad),
            get_ptr(gnorm_vec),
            ct.c_int32(step),
            ct.c_int32(grad.numel()),
        )
    else:
        raise ValueError(f"Gradient type {grad.dtype} not supported!")

    current_gnorm = torch.sqrt(gnorm_vec[step % 100])
    vals, idx = torch.sort(gnorm_vec)
    clip_value = torch.sqrt(vals[percentile])
    gnorm_scale = 1.0

    if current_gnorm > clip_value:
        gnorm_scale = clip_value / current_gnorm

    return current_gnorm, clip_value, gnorm_scale


def histogram_scatter_add_2d(
    histogram: Tensor, index1: Tensor, index2: Tensor, source: Tensor
):
    assert len(histogram.shape) == 2
    assert histogram.dtype == torch.float32
    assert source.dtype == torch.float32
    assert index1.dtype == torch.int32
    assert index2.dtype == torch.int32

    assert histogram.device.type == "cuda"
    assert index1.device.type == "cuda"
    assert index2.device.type == "cuda"
    assert source.device.type == "cuda"

    maxdim1 = ct.c_int32(histogram.shape[0])
    n = ct.c_int32(index1.numel())
    is_on_gpu([histogram, index1, index2, source])
    lib.chistogram_scatter_add_2d(get_ptr(histogram), get_ptr(index1), get_ptr(index2), get_ptr(source), maxdim1, n)

def check_matmul(A, B, out, transposed_A, transposed_B, expected_type=torch.int8):
    if not torch.cuda.is_initialized(): torch.cuda.init()
    if A.dtype != expected_type or B.dtype != expected_type:
        raise TypeError(
            f"Expected torch.int8 input tensors A and B, but got {A.dtype} and {B.dtype}"
        )

    sA = A.shape
    sB = B.shape
    tA = transposed_A
    tB = transposed_B

    correct = True

    if len(sA) == 2 and len(sB) == 2:
        if not tA and not tB and A.shape[1] != B.shape[0]:
            correct = False
        elif tA and not tB and A.shape[0] != B.shape[0]:
            correct = False
        elif tA and tB and A.shape[0] != B.shape[1]:
            correct = False
        elif not tA and tB and A.shape[1] != B.shape[1]:
            correct = False
    elif len(sA) == 3 and len(sB) == 2:
        if not tA and not tB and A.shape[2] != B.shape[0]:
            correct = False
        elif tA and not tB and A.shape[1] != B.shape[0]:
            correct = False
        elif tA and tB and A.shape[1] != B.shape[1]:
            correct = False
        elif not tA and tB and A.shape[2] != B.shape[1]:
            correct = False
    elif len(sA) == 3 and len(sB) == 3:
        if not tA and not tB and A.shape[2] != B.shape[1]:
            correct = False
        elif tA and not tB and A.shape[1] != B.shape[1]:
            correct = False
        elif tA and tB and A.shape[1] != B.shape[2]:
            correct = False
        elif not tA and tB and A.shape[2] != B.shape[2]:
            correct = False

    if out is not None:
        sout = out.shape
        # special case common in backprop
        if not correct and len(sA) == 3 and len(sB) == 3:
            if (
                sout[0] == sA[2]
                and sout[1] == sB[2]
                and sA[0] == sB[0]
                and sA[1] == sB[1]
            ):
                correct = True
    else:
        if len(sA) == 2 and len(sB) == 2:
            if not tA and not tB:
                sout = (sA[0], sB[1])
            elif tA and tB:
                sout = (sA[1], sB[0])
            elif tA and not tB:
                sout = (sA[1], sB[1])
            elif not tA and tB:
                sout = (sA[0], sB[0])
        elif len(sA) == 3 and len(sB) == 2:
            if not tA and not tB:
                sout = (sA[0], sA[1], sB[1])
            elif tA and tB:
                sout = (sA[0], sA[2], sB[0])
            elif tA and not tB:
                sout = (sA[0], sA[2], sB[1])
            elif not tA and tB:
                sout = (sA[0], sA[1], sB[0])
        elif len(sA) == 3 and len(sB) == 3:
            if not tA and not tB:
                sout = (sA[0], sA[1], sB[2])
            elif tA and tB:
                sout = (sA[0], sA[2], sB[1])
            elif tA and not tB:
                sout = (sA[0], sA[2], sB[2])
            elif not tA and tB:
                sout = (sA[0], sA[1], sB[1])

    if not correct:
        raise ValueError(
            f"Tensor dimensions incorrect for matrix mulitiplication: A x B: {sA} x {sB} with transpose for A x B: {tA} x {tB}."
        )

    return sout


def igemm(
    A: Tensor,
    B: Tensor,
    out: Tensor = None,
    transposed_A=False,
    transposed_B=False,
):
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]:
            return batched_igemm(A, B, out)

    sA = A.shape
    sB = B.shape
    if transposed_A and len(sA) == 2:
        sA = (sA[1], sA[0])
    elif transposed_A and len(sA) == 3:
        sA = (sA[0], sA[2], sA[0])
    if transposed_B and len(sB) == 2:
        sB = (sB[1], sB[0])
    elif transposed_B and len(sB) == 3:
        sB = (sB[0], sB[2], sB[0])
    # this is a mess: cuBLAS expect column major, but PyTorch is row major.
    # So to perform the matrix multiplication, we have to treat A, B, and C matrices
    # (transpose of row major is column major)
    # This means we compute B^T A^T = C^T and we explicitly switch the dimensions of each of these

    # matrices in the input arguments for cuBLAS
    # column major: A @ B = C: [m, k] @ [k, n] = [m, n]
    # row major: B^T @ A^T = C^T: [m, k] @ [k, n] = [m, n]
    # column major with row major layout: B^T @ A^T = C^T: [k, m] @ [n, k] = [n, m]
    if len(sB) == 2:
        if B.stride()[0] == B.shape[1]:
            transposed_B = False
        elif B.stride()[1] == B.shape[0]:
            transposed_B = True
        if len(A.shape) == 2:
            if A.stride()[0] == A.shape[1]:
                transposed_A = False
            elif A.stride()[1] == A.shape[0]:
                transposed_A = True
        else:
            if A.stride()[1] == A.shape[2]:
                transposed_A = False
            elif A.stride()[2] == A.shape[1]:
                transposed_A = True

        if len(sA) == 2:
            n = sA[0]
            ldb = A.stride()[1 if transposed_A else 0]
        elif len(sA) == 3 and len(sB) == 2:
            n = sA[0] * sA[1]
            ldb = sA[2]

        m = sB[1]
        k = sB[0]
        lda = B.stride()[(1 if transposed_B else 0)]
        ldc = sB[1]
    elif len(sB) == 3:
        # special case
        assert len(sA) == 3
        if not (sA[0] == sB[0] and sA[1] == sB[1]):
            raise ValueError(
                f"Only bsi,bso->io supported for tensor contractions, but dims for A x B were: {sA} x {sB}"
            )

        transposed_A = True
        transposed_B = False

        m = sB[2]
        n = sA[2]
        k = sB[0] * sB[1]

        lda = m
        ldb = sA[2]
        ldc = m

    ptr = CUBLAS_Context.get_instance().get_context(A.device)

    # B^T @ A^T = C^T
    # [km, nk -> mn]
    is_on_gpu([B, A, out])
    lib.cigemm(ptr, ct.c_bool(transposed_B), ct.c_bool(transposed_A), ct.c_int32(m), ct.c_int32(n), ct.c_int32(k),
               get_ptr(B), get_ptr(A), get_ptr(out), ct.c_int32(lda), ct.c_int32(ldb), ct.c_int32(ldc))
    return out


def batched_igemm(
    A: Tensor,
    B: Tensor,
    out: Tensor = None,
    transposed_A=False,
    transposed_B=False,
):
    if not len(A.shape) == 3 or not len(B.shape) == 3:
        raise ValueError(
            f"Expected 3-dimensional tensors for bmm, but got shapes A and B: {A.shape} and {B.shape}"
        )
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)

    if B.is_contiguous():
        lda = B.stride()[1]
        transposed_A = False
    else:
        s = B.stride()
        if s[0] != B.shape[0]:
            B = B.contiguous()
            lda = B.stride()[1]
        elif s[2] == B.shape[1]:
            transposed_A = True
            lda = B.stride()[2]
        else:
            if s[2] == 1:
                B = B.contiguous()
                lda = B.stride()[1]
            elif s[1] == 1:
                B = B.contiguous()
                lda = B.stride()[1]
            else:
                B = B.contiguous()
                lda = B.stride()[1]

    if A.is_contiguous():
        ldb = A.stride()[1]
        transposed_B = False
    else:
        s = A.stride()
        if s[0] != A.shape[0]:
            A = A.contiguous()
            ldb = A.stride()[1]
            transposed_B = False
        elif s[2] == A.shape[1]:
            ldb = A.stride()[2]
            transposed_B = True
        else:
            A = A.contiguous()
            ldb = A.stride()[1]
            transposed_B = False

    # this is a mess: cuBLAS expect column major, but PyTorch is row major.
    # So to perform the matrix multiplication, we have to treat A, B, and C matrices
    # (transpose of row major is column major)
    # This means we compute B^T A^T = C^T and we explicitly switch the dimensions of each of these
    # matrices in the input arguments for cuBLAS

    # column major: A @ B = C: [batch, m, k] @ [batch, k, n] = [batch, m, n]
    # row major: B^T @ A^T = C^T: [batch, m, k] @ [batch, k, n] = [batch, m, n]
    # column major with row major layout: B^T @ A^T = C^T: [batch, k, m] @ [batch, n, k] = [batch, n, m]
    num_batch = A.shape[0]
    n = A.shape[1]
    m = B.shape[2]
    k = B.shape[1]

    ldc = m

    strideA = B.shape[1] * B.shape[2]
    strideB = A.shape[1] * A.shape[2]
    strideC = A.shape[1] * B.shape[2]

    ptr = CUBLAS_Context.get_instance().get_context(A.device)

    is_on_gpu([B, A, out])
    lib.cbatched_igemm(ptr, ct.c_bool(transposed_B), ct.c_bool(transposed_A), ct.c_int32(m), ct.c_int32(n), ct.c_int32(k),
               get_ptr(B), get_ptr(A), get_ptr(out), ct.c_int32(lda), ct.c_int32(ldb), ct.c_int32(ldc),
               ct.c_long(strideA), ct.c_long(strideB), ct.c_long(strideC), ct.c_uint32(num_batch))
    return out


def igemmlt(A, B, SA, SB, out=None, Sout=None, dtype=torch.int32):
    shapeA = SA[0]
    shapeB = SB[0]
    dimsA = len(shapeA)
    dimsB = len(shapeB)
    assert dimsB == 2, 'Only two dimensional matrices are supported for argument B'
    if dimsA == 2:
        m = shapeA[0]
    elif dimsA == 3:
        m = shapeA[0] * shapeA[1]

    rows = n = shapeB[0]
    assert prod(list(shapeA)) > 0, f'Input tensor dimensions need to be > 0: {shapeA}'

    # if the tensor is empty, return a transformed empty tensor with the right dimensions
    if shapeA[0] == 0 and dimsA == 2:
        return torch.empty((0, shapeB[0]), device=A.device, dtype=torch.float16)
    elif shapeA[1] == 0 and dimsA == 3:
        return torch.empty(tuple(shapeA[:2] + [shapeB[0]]), device=A.device, dtype=torch.float16)

    if dimsA == 2 and out is None:
        out, Sout = get_transform_buffer(
            (shapeA[0], shapeB[0]), dtype, A.device, "col32", "row"
        )
    elif dimsA == 3 and out is None:
        out, Sout = get_transform_buffer(
            (shapeA[0], shapeA[1], shapeB[0]), dtype, A.device, "col32", "row"
        )

    assert dimsB != 3, "len(B.shape)==3 not supported"
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype == torch.int8
    assert B.dtype == torch.int8
    assert out.dtype == dtype
    assert SA[1] == "col32"
    assert SB[1] in ["col_turing", "col_ampere"]
    assert Sout[1] == "col32"
    assert (
        shapeA[-1] == shapeB[-1]
    ), f"Matmullt only supports A @ B^T. Inner matrix dimensions do not match: A @ B = {shapeA} @ {shapeB}"
    formatB = SB[1]
    prev_device = A.device
    torch.cuda.set_device(A.device)

    ptr = CUBLAS_Context.get_instance().get_context(A.device)
    ptrA = get_ptr(A)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)

    k = shapeA[-1]
    lda = ct.c_int32(m * 32)
    if formatB == "col_turing":
        # turing: tiles with rows filled up to multiple of 8 rows by 32 columns
        # n = rows
        ldb = ct.c_int32(((rows + 7) // 8) * 8 * 32)
    else:
        # ampere: tiles with rows filled up to multiple of 32 rows by 32 columns
        # n = rows
        ldb = ct.c_int32(((rows + 31) // 32) * 32 * 32)

    ldc = ct.c_int32(m * 32)
    m = ct.c_int32(m)
    n = ct.c_int32(n)
    k = ct.c_int32(k)

    has_error = 0
    ptrRowScale = get_ptr(None)
    is_on_gpu([A, B, out])
    if formatB == 'col_turing':
        if dtype == torch.int32:
            has_error = lib.cigemmlt_turing_32(
                ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
            )
        else:
            has_error = lib.cigemmlt_turing_8(
                ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
            )
    elif formatB == "col_ampere":
        if dtype == torch.int32:
            has_error = lib.cigemmlt_ampere_32(
                ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
            )
        else:
            has_error = lib.cigemmlt_ampere_8(
                ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
            )

    if has_error == 1:
        print(f'A: {shapeA}, B: {shapeB}, C: {Sout[0]}; (lda, ldb, ldc): {(lda, ldb, ldc)}; (m, n, k): {(m, n, k)}')
        raise Exception('cublasLt ran into an error!')

    torch.cuda.set_device(prev_device)

    return out, Sout


def mm_dequant(
    A,
    quant_state,
    row_stats,
    col_stats,
    out=None,
    new_row_stats=None,
    new_col_stats=None,
    bias=None
):
    assert A.dtype == torch.int32
    if bias is not None: assert bias.dtype == torch.float16
    out_shape = quant_state[0]
    if len(out_shape) == 3:
        out_shape = (out_shape[0] * out_shape[1], out_shape[2])

    if out is None:
        out = torch.empty(out_shape, dtype=torch.float16, device=A.device)
    if new_row_stats is None:
        new_row_stats = torch.empty(
            out_shape[0], dtype=torch.float32, device=A.device
        )
    if new_col_stats is None:
        new_col_stats = torch.empty(
            out_shape[1], dtype=torch.float32, device=A.device
        )
    assert (
        new_row_stats.shape[0] == row_stats.shape[0]
    ), f"{new_row_stats.shape} vs {row_stats.shape}"
    assert (
        new_col_stats.shape[0] == col_stats.shape[0]
    ), f"{new_col_stats.shape} vs {col_stats.shape}"

    prev_device = pre_call(A.device)
    ptrA = get_ptr(A)
    ptrOut = get_ptr(out)
    ptrRowStats = get_ptr(row_stats)
    ptrColStats = get_ptr(col_stats)
    ptrNewRowStats = get_ptr(new_row_stats)
    ptrNewColStats = get_ptr(new_col_stats)
    ptrBias = get_ptr(bias)
    numRows = ct.c_int32(out_shape[0])
    numCols = ct.c_int32(out_shape[1])

    is_on_gpu([A, row_stats, col_stats, out, new_row_stats, new_col_stats, bias])
    lib.cdequant_mm_int32_fp16(ptrA, ptrRowStats, ptrColStats, ptrOut, ptrNewRowStats, ptrNewColStats, ptrBias, numRows, numCols)
    post_call(prev_device)

    return out


def get_colrow_absmax(
    A, row_stats=None, col_stats=None, nnz_block_ptr=None, threshold=0.0
):
    assert A.dtype == torch.float16
    device = A.device

    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    col_tiles = (cols + 255) // 256
    tiled_rows = ((rows + 15) // 16) * 16
    if row_stats is None:
        row_stats = torch.empty(
            (rows,), dtype=torch.float32, device=device
        ).fill_(-50000.0)
    if col_stats is None:
        col_stats = torch.empty(
            (cols,), dtype=torch.float32, device=device
        ).fill_(-50000.0)

    if nnz_block_ptr is None and threshold > 0.0:
        nnz_block_ptr = torch.zeros(
            ((tiled_rows * col_tiles) + 1,), dtype=torch.int32, device=device
        )

    ptrA = get_ptr(A)
    ptrRowStats = get_ptr(row_stats)
    ptrColStats = get_ptr(col_stats)
    ptrNnzrows = get_ptr(nnz_block_ptr)
    rows = ct.c_int32(rows)
    cols = ct.c_int32(cols)

    prev_device = pre_call(A.device)
    is_on_gpu([A, row_stats, col_stats, nnz_block_ptr])
    lib.cget_col_row_stats(ptrA, ptrRowStats, ptrColStats, ptrNnzrows, ct.c_float(threshold), rows, cols)
    post_call(prev_device)

    if threshold > 0.0:
        nnz_block_ptr.cumsum_(0)

    return row_stats, col_stats, nnz_block_ptr


class COOSparseTensor:
    def __init__(self, rows, cols, nnz, rowidx, colidx, values):
        assert rowidx.dtype == torch.int32
        assert colidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colidx.numel() == nnz

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowidx = rowidx
        self.colidx = colidx
        self.values = values


class CSRSparseTensor:
    def __init__(self, rows, cols, nnz, rowptr, colidx, values):
        assert rowptr.dtype == torch.int32
        assert colidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert colidx.numel() == nnz
        assert rowptr.numel() == rows + 1

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowptr = rowptr
        self.colidx = colidx
        self.values = values


class CSCSparseTensor:
    def __init__(self, rows, cols, nnz, colptr, rowidx, values):
        assert colptr.dtype == torch.int32
        assert rowidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colptr.numel() == cols + 1

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.colptr = colptr
        self.rowidx = rowidx
        self.values = values


def coo2csr(cooA):
    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    values.add_(1)
    rowptr = torch.zeros(
        (cooA.rows + 1,), dtype=torch.int32, device=cooA.rowidx.device
    )
    rowptr.scatter_(index=values.long(), src=counts.int(), dim=0)
    rowptr.cumsum_(0)
    return CSRSparseTensor(
        cooA.rows, cooA.cols, cooA.nnz, rowptr, cooA.colidx, cooA.values
    )


def coo2csc(cooA):
    val, col2rowidx = torch.sort(cooA.colidx)
    rowidx = cooA.rowidx[col2rowidx]
    values = cooA.values[col2rowidx]
    colvalues, counts = torch.unique(val, return_counts=True)
    colvalues.add_(1)
    colptr = torch.zeros(
        (cooA.cols + 1,), dtype=torch.int32, device=cooA.colidx.device
    )
    colptr.scatter_(index=colvalues.long(), src=counts.int(), dim=0)
    colptr.cumsum_(0)
    return CSCSparseTensor(
        cooA.rows, cooA.cols, cooA.nnz, colptr, rowidx, values
    )


def coo_zeros(rows, cols, nnz, device, dtype=torch.half):
    rowidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
    colidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
    values = torch.zeros((nnz,), dtype=dtype, device=device)
    return COOSparseTensor(rows, cols, nnz, rowidx, colidx, values)


def double_quant(
    A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0
):
    device = A.device
    assert A.dtype == torch.half
    assert device.type == "cuda"
    prev_device = pre_call(A.device)

    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    if row_stats is None or col_stats is None:
        row_stats, col_stats, nnz_row_ptr = get_colrow_absmax(
            A, threshold=threshold
        )

    if out_col is None:
        out_col = torch.zeros(A.shape, device=device, dtype=torch.int8)
    if out_row is None:
        out_row = torch.zeros(A.shape, device=device, dtype=torch.int8)

    coo_tensor = None
    ptrA = get_ptr(A)
    ptrColStats = get_ptr(col_stats)
    ptrRowStats = get_ptr(row_stats)
    ptrOutCol = get_ptr(out_col)
    ptrOutRow = get_ptr(out_row)

    is_on_gpu([A, col_stats, row_stats, out_col, out_row])
    if threshold > 0.0:
        nnz = nnz_row_ptr[-1].item()
        if nnz > 0:
            coo_tensor = coo_zeros(
                A.shape[0], A.shape[1], nnz_row_ptr[-1].item(), device
            )
            ptrRowIdx = get_ptr(coo_tensor.rowidx)
            ptrColIdx = get_ptr(coo_tensor.colidx)
            ptrVal = get_ptr(coo_tensor.values)
            ptrRowPtr = get_ptr(nnz_row_ptr)

            lib.cdouble_rowcol_quant(
                ptrA,
                ptrRowStats,
                ptrColStats,
                ptrOutCol,
                ptrOutRow,
                ptrRowIdx,
                ptrColIdx,
                ptrVal,
                ptrRowPtr,
                ct.c_float(threshold),
                ct.c_int32(rows),
                ct.c_int32(cols),
            )
            val, idx = torch.sort(coo_tensor.rowidx)
            coo_tensor.rowidx = val
            coo_tensor.colidx = coo_tensor.colidx[idx]
            coo_tensor.values = coo_tensor.values[idx]
        else:
            lib.cdouble_rowcol_quant(
                ptrA,
                ptrRowStats,
                ptrColStats,
                ptrOutCol,
                ptrOutRow,
                None,
                None,
                None,
                None,
                ct.c_float(0.0),
                ct.c_int32(rows),
                ct.c_int32(cols),
            )
    else:
        lib.cdouble_rowcol_quant(
            ptrA,
            ptrRowStats,
            ptrColStats,
            ptrOutCol,
            ptrOutRow,
            None,
            None,
            None,
            None,
            ct.c_float(threshold),
            ct.c_int32(rows),
            ct.c_int32(cols),
        )
    post_call(prev_device)

    return out_row, out_col, row_stats, col_stats, coo_tensor


def transform(A, to_order, from_order='row', out=None, transpose=False, state=None, ld=None):
    prev_device = pre_call(A.device)
    if state is None: state = (A.shape, from_order)
    else: from_order = state[1]
    if out is None: out, new_state = get_transform_buffer(state[0], A.dtype, A.device, to_order, state[1], transpose)
    else: new_state = (state[0], to_order) # (shape, order)

    shape = state[0]
    if len(shape) == 2:
        dim1 = ct.c_int32(shape[0])
        dim2 = ct.c_int32(shape[1])
    else:
        dim1 = ct.c_int32(shape[0] * shape[1])
        dim2 = ct.c_int32(shape[2])

    is_on_gpu([A, out])
    if to_order == 'col32':
        if transpose:
            lib.ctransform_row2col32T(get_ptr(A), get_ptr(out), dim1, dim2)
        else:
            lib.ctransform_row2col32(get_ptr(A), get_ptr(out), dim1, dim2)
    elif to_order == "col_turing":
        if transpose:
            lib.ctransform_row2turingT(get_ptr(A), get_ptr(out), dim1, dim2)
        else:
            lib.ctransform_row2turing(get_ptr(A), get_ptr(out), dim1, dim2)
    elif to_order == "col_ampere":
        if transpose:
            lib.ctransform_row2ampereT(get_ptr(A), get_ptr(out), dim1, dim2)
        else:
            lib.ctransform_row2ampere(get_ptr(A), get_ptr(out), dim1, dim2)
    elif to_order == "row":
        if from_order == "col_turing":
            lib.ctransform_turing2row(get_ptr(A), get_ptr(out), dim1, dim2)
        elif from_order == "col_ampere":
            lib.ctransform_ampere2row(get_ptr(A), get_ptr(out), dim1, dim2)
    else:
        raise NotImplementedError(f'Transform function not implemented: From {from_order} to {to_order}')

    post_call(prev_device)

    return out, new_state


def spmm_coo(cooA, B, out=None):
    if out is None:
        out = torch.empty(
            (cooA.rows, B.shape[1]), device=B.device, dtype=B.dtype
        )
    nnz = cooA.nnz
    assert cooA.rowidx.numel() == nnz
    assert cooA.colidx.numel() == nnz
    assert cooA.values.numel() == nnz
    assert cooA.cols == B.shape[0]

    transposed_B = False if B.is_contiguous() else True

    ldb = B.stride()[(1 if transposed_B else 0)]
    ldc = B.shape[1]

    ptr = Cusparse_Context.get_instance().context

    ptrRowidx = get_ptr(cooA.rowidx)
    ptrColidx = get_ptr(cooA.colidx)
    ptrValues = get_ptr(cooA.values)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)
    cnnz = ct.c_int32(cooA.nnz)
    crowsA = ct.c_int32(cooA.rows)
    ccolsA = ct.c_int32(cooA.cols)
    ccolsB = ct.c_int32(B.shape[1])
    cldb = ct.c_int32(ldb)
    cldc = ct.c_int32(ldc)

    is_on_gpu([cooA.rowidx, cooA.colidx, cooA.values, B, out])
    lib.cspmm_coo(ptr, ptrRowidx, ptrColidx, ptrValues, cnnz, crowsA, ccolsA, ccolsB, cldb, ptrB, cldc, ptrC, ct.c_bool(transposed_B))

    return out


def spmm_coo_very_sparse(cooA, B, dequant_stats=None, out=None):
    if out is None:
        out = torch.zeros(
            (cooA.rows, B.shape[1]), device=B.device, dtype=cooA.values.dtype
        )
    nnz = cooA.nnz
    assert cooA.rowidx.numel() == nnz
    assert cooA.colidx.numel() == nnz
    assert cooA.values.numel() == nnz
    assert cooA.cols == B.shape[0], f"{cooA.cols} vs {B.shape}"

    transposed_B = False if B.is_contiguous() else True

    ldb = B.stride()[(1 if transposed_B else 0)]
    ldc = B.shape[1]

    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    offset = counts.cumsum(0).int()
    max_count, max_idx = torch.sort(counts, descending=True)
    max_idx = max_idx.int()
    max_count = max_count.int()
    assert (
        max_count[0] <= 32
    ), f"Current max count per row is 8 but found {max_count[0]}."
    assert B.dtype in [torch.float16, torch.int8]
    ptrOffset = get_ptr(offset)
    ptrMaxCount = get_ptr(max_count)
    ptrMaxIdx = get_ptr(max_idx)

    ptrRowidx = get_ptr(cooA.rowidx)
    ptrColidx = get_ptr(cooA.colidx)
    ptrValues = get_ptr(cooA.values)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)
    ptrDequantStats = get_ptr(dequant_stats)
    cnnz_rows = ct.c_int32(counts.numel())
    cnnz = ct.c_int32(cooA.nnz)
    crowsA = ct.c_int32(cooA.rows)
    ccolsA = ct.c_int32(cooA.cols)
    crowsB = ct.c_int32(B.shape[1])
    ccolsB = ct.c_int32(B.shape[1])
    cldb = ct.c_int32(ldb)
    cldc = ct.c_int32(ldc)
    # print(cooA.rowidx[:64])
    # print(cooA.colidx[:64].sort()[0])

    is_on_gpu([cooA.rowidx, cooA.colidx, cooA.values, B, out, dequant_stats])
    if B.dtype == torch.float16:
        lib.cspmm_coo_very_sparse_naive_fp16(
            ptrMaxCount,
            ptrMaxIdx,
            ptrOffset,
            ptrRowidx,
            ptrColidx,
            ptrValues,
            ptrB,
            ptrC,
            ptrDequantStats,
            cnnz_rows,
            cnnz,
            crowsA,
            crowsB,
            ccolsB,
        )
    elif B.dtype == torch.int8:
        lib.cspmm_coo_very_sparse_naive_int8(
            ptrMaxCount,
            ptrMaxIdx,
            ptrOffset,
            ptrRowidx,
            ptrColidx,
            ptrValues,
            ptrB,
            ptrC,
            ptrDequantStats,
            cnnz_rows,
            cnnz,
            crowsA,
            crowsB,
            ccolsB,
        )
    # else: assertion error

    return out


C = 127.0


def vectorwise_quant(x, dim=1, quant_type="vector"):
    if quant_type == "linear":
        max1 = torch.abs(x).max().float()
        xq = torch.round(x / max1 * 127).to(torch.int8)
        return xq, max1
    elif quant_type in ["vector", "row"]:
        max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        xq = torch.round(x * (C / max1)).to(torch.int8)
        return xq, max1
    elif quant_type == "zeropoint":
        dtype = x.dtype
        x = x.float()
        dyna = x.max() - x.min()
        if dyna == 0:
            dyna = 1
        qx = 255.0 / dyna
        minx = x.min()
        zpx = torch.round(minx * qx)
        x = torch.round(qx * x - zpx) + zpx
        return x, qx
    elif quant_type in ["vector-zeropoint", "row-zeropoint"]:
        dtype = x.dtype
        x = x.float()
        dyna = torch.amax(x, dim=dim, keepdim=True) - torch.amin(
            x, dim=dim, keepdim=True
        )
        dyna[dyna == 0] = 1
        qx = 255.0 / dyna
        minx = torch.amin(x, dim=dim, keepdim=True)
        zpx = torch.round(minx * qx)
        x = torch.round(qx * x - zpx) + zpx
        return x, qx
    elif quant_type == "truncated-vector":
        with torch.no_grad():
            absx = torch.abs(x)
            max1 = torch.amax(absx, dim=dim, keepdim=True)
            max1 = max1 * 0.7
            idx = absx > max1.expand_as(absx)
            sign = torch.sign(x[idx])
            x[idx] = max1.expand_as(absx)[idx] * sign
            xq = torch.round(x / max1 * C).to(torch.int8)
        return xq, max1
    else:
        return None


def vectorwise_dequant(xq, max1, quant_type="vector"):
    if quant_type == "vector":
        x = (xq / C * max1).to(torch.float32)
        return x
    else:
        return None


def vectorwise_mm_dequant(xq, S1, S2, dtype=torch.half, quant_type="vector"):
    if quant_type == "linear":
        norm = S1 * S2 / (C * C)
        # double cast needed to prevent overflows
        return (xq.float() * norm).to(dtype)
    elif quant_type == "zeropoint":
        norm = 1.0 / (S1 * S2)
        return (xq.float() * norm).to(dtype)
    elif quant_type == "row-zeropoint":
        norm = 1.0 / (S1 * S2)
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= norm
        else:
            x *= norm
        return x.to(dtype)
    elif quant_type == "vector-zeropoint":
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= 1.0 / S1
        else:
            x *= 1.0 / S1
        x *= 1.0 / S2.t()
        return x.to(dtype)
    elif quant_type == "row":
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= S1 * S2 / (C * C)
        else:
            x *= S1 * S2 / (C * C)
        return x.to(dtype)
    elif quant_type in ["truncated-vector", "vector"]:
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= S1 / C
        else:
            x *= S1 / C
        x *= S2 / C
        return x.to(dtype)
    else:
        return None


def dequant_min_max(xq, A, B, SA, SB, dtype=torch.half):
    offset = B.float().t().sum(0) * (SA[0] + SA[1])
    x = xq.float()
    if len(xq.shape) == 2 and len(SB.shape) == 3:
        SB = SB.squeeze(0)
    if len(SB.shape) == 2:
        x *= SB.t() / 127
    else:
        x *= SB / 127
    x *= SA[1] / 127
    x += offset
    return x.to(dtype)


def extract_outliers(A, SA, idx):
    shapeA = SA[0]
    formatA = SA[1]
    assert formatA in ["col_turing", "col_ampere"]
    assert A.device.type == "cuda"

    out = torch.zeros(
        (shapeA[0], idx.numel()), dtype=torch.int8, device=A.device
    )

    idx_size = ct.c_int32(idx.numel())
    rows = ct.c_int32(shapeA[0])
    cols = ct.c_int32(shapeA[1])
    ptrA = get_ptr(A)
    ptrIdx = get_ptr(idx)
    ptrOut = get_ptr(out)

    prev_device = pre_call(A.device)
    if formatA == 'col_turing':
        lib.cextractOutliers_turing(ptrA, ptrIdx, ptrOut, idx_size, rows, cols)
    elif formatA == "col_ampere":
        lib.cextractOutliers_ampere(ptrA, ptrIdx, ptrOut, idx_size, rows, cols)
    post_call(prev_device)

    return out
