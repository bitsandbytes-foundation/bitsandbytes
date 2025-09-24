# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Iterable
import ctypes as ct
import itertools
from math import prod
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import Tensor
from typing_extensions import deprecated

from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict

from .cextension import HIP_ENVIRONMENT, lib

name2qmap = {}

"""C FUNCTIONS FOR OPTIMIZERS"""
str2optimizer8bit = {
    "adam": (
        lib.cadam_static_8bit_grad_32,
        lib.cadam_static_8bit_grad_16,
    ),
    "momentum": (
        lib.cmomentum_static_8bit_grad_32,
        lib.cmomentum_static_8bit_grad_16,
    ),
    "rmsprop": (
        lib.crmsprop_static_8bit_grad_32,
        lib.crmsprop_static_8bit_grad_16,
    ),
    "lion": (
        lib.clion_static_8bit_grad_32,
        lib.clion_static_8bit_grad_16,
    ),
    "lamb": (
        lib.cadam_static_8bit_grad_32,
        lib.cadam_static_8bit_grad_16,
    ),
    "lars": (
        lib.cmomentum_static_8bit_grad_32,
        lib.cmomentum_static_8bit_grad_16,
    ),
}


class GlobalPageManager:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.paged_tensors = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def prefetch_all(self, to_cpu=False):
        # assume the first added, will be the
        # ones that are used first, so swap them in last
        # in the case they are evicted again
        for t in self.paged_tensors[::-1]:
            prefetch_tensor(t, to_cpu)


class CUBLAS_Context:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.context = {}

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


FIRST_CUDA_DEVICE = torch.device("cuda", index=0)

# When multiple GPUs are present, we use a context manager to
# switch to the correct device of a tensor before invoking our CUDA
# kernels in the C++ library. However, when there's only one device
# there is no need to incur the overhead of cudaGetDevice/cudaSetDevice.
if torch.cuda.device_count() > 1:

    def _cuda_device_of(a: torch.Tensor):
        return torch.cuda.device_of(a)
else:
    import contextlib

    def _cuda_device_of(a: torch.Tensor):
        return contextlib.nullcontext()


def get_paged(*shape, dtype=torch.float32, device=FIRST_CUDA_DEVICE):
    num_bytes = dtype.itemsize * prod(shape)
    cuda_ptr = lib.cget_managed_ptr(ct.c_size_t(num_bytes))
    c_ptr = ct.cast(cuda_ptr, ct.POINTER(ct.c_int))
    new_array = np.ctypeslib.as_array(c_ptr, shape=shape)
    out = torch.frombuffer(new_array, dtype=dtype, count=prod(shape)).view(shape)
    out.is_paged = True
    out.page_deviceid = device.index
    return out


def prefetch_tensor(A: torch.Tensor, to_cpu=False):
    assert A.is_paged, "Only paged tensors can be prefetched!"
    if to_cpu:
        deviceid = -1
    else:
        deviceid = A.page_deviceid

    lib.cprefetch(get_ptr(A), ct.c_size_t(A.nbytes), ct.c_int32(deviceid))


def elementwise_func(func_name, A, B, value, prefetch=True):
    func = None
    if A.dtype == torch.float32:
        func = getattr(lib, f"c{func_name}_fp32", None)
        cvalue = ct.c_float(value)
    elif A.dtype == torch.uint8:
        func = getattr(lib, f"c{func_name}_uint8", None)
        cvalue = ct.c_uint8(value)

    if func is None:
        raise NotImplementedError(f"Function not implemented: {func_name}")

    is_managed = getattr(A, "is_managed", False)
    if is_managed and prefetch:
        prefetch_tensor(A)
        if B is not None:
            prefetch_tensor(B)

    func(get_ptr(A), get_ptr(B), cvalue, ct.c_int64(A.numel()))
    if A.is_paged or B.is_paged:
        # paged function are fully asynchronous
        # if we return from this function, we want to the tensor
        # to be in the correct state, that is the final state after the
        # operation occurred. So we synchronize.
        torch.cuda.synchronize()


def fill(A, value, device=None, prefetch=True):
    elementwise_func("fill", A, None, value)


def _mul(A, B, device=None):
    elementwise_func("_mul", A, B, 0)


def create_linear_map(signed=True, total_bits=8, add_zero=True):
    sign = -1.0 if signed else 0.0
    total_values = 2**total_bits
    if add_zero or total_bits < 8:
        # add a zero
        # since we simulate less bits by having zeros in the data type, we
        # we need to center the quantization around zero and as such lose
        # a single value
        total_values = 2**total_bits if not signed else 2**total_bits - 1

    values = torch.linspace(sign, 1.0, total_values)
    gap = 256 - values.numel()
    if gap == 0:
        return values
    else:
        l = values.numel() // 2  # noqa: E741
        return torch.Tensor(values[:l].tolist() + [0] * gap + values[l:].tolist())


def create_normal_map(offset=0.9677083, use_extra_value=True):
    try:
        from scipy.stats import norm
    except ImportError as ie:
        raise ImportError(
            "Scipy is required for `create_normal_map`. Install `bitsandbytes` with the `[test]` extra.",
        ) from ie

    if use_extra_value:
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1]).tolist()
        v2 = [0] * (256 - 15)  ## we have 15 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, 8)[:-1]).tolist()
        v2 = [0] * (256 - 14)  ## we have 14 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()

    v = v1 + v2 + v3

    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == 256

    return values


def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e + p == total_bits - has_sign
    # the exponent is biased to 2^(e-1) -1 == 0
    evalues = []
    for i, val in enumerate(range(-(2 ** (exponent_bits - has_sign)), 2 ** (exponent_bits - has_sign), 1)):
        evalues.append(2**val)

    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    # for ev in evalues:
    bias = 2 ** (exponent_bits - 1)
    for evalue in range(2 ** (exponent_bits)):
        for bit_pattern in lst:
            value = 1 if evalue != 0 else 0
            for i, pval in enumerate(list(bit_pattern)):
                value += pval * (2 ** -(i + 1))
            if evalue == 0:
                # subnormals
                value = value * 2**-(bias)
            else:
                # normals
                value = value * 2 ** -(evalue - bias - 1)
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
    code = torch.tensor(values)
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
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )
        boundaries = torch.linspace(0.1, 1, fraction_items, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits

    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    data.sort()
    return torch.tensor(data, dtype=torch.float32)


def is_on_gpu(tensors: Iterable[Optional[torch.Tensor]]):
    """Verifies that the input tensors are all on the same device.

    An input tensor may also be marked as `paged`, in which case the device placement is ignored.

    Args:
        tensors (`Iterable[Optional[torch.Tensor]]`): A list of tensors to verify.

    Raises:
        `RuntimeError`: Raised when the verification fails.

    Returns:
        `Literal[True]`
    """

    on_gpu = True
    gpu_ids = set()

    for t in tensors:
        # NULL pointers and paged tensors are OK.
        if t is not None and not getattr(t, "is_paged", False):
            on_gpu &= t.device.type != "cpu"
            gpu_ids.add((t.device.type, t.device.index))

    if not on_gpu:
        raise RuntimeError(
            f"All input tensors need to be on the same GPU, but found some tensors to not be on a GPU:\n {[(t.shape, t.device) for t in tensors]}",
        )

    if len(gpu_ids) > 1:
        raise RuntimeError(
            f"Input tensors need to be on the same GPU, but found the following tensor and device combinations:\n {[(t.shape, t.device) for t in tensors]}",
        )
    return on_gpu


def _get_tensor_stream(tensor: Tensor) -> ct.c_void_p:
    # We use the raw stream for performance reasons.
    if tensor.device.type == "xpu":
        return ct.c_void_p(torch._C._xpu_getCurrentRawStream(tensor.device.index))
    return ct.c_void_p(torch._C._cuda_getCurrentRawStream(tensor.device.index))


def get_ptr(A: Optional[Tensor]) -> Optional[ct.c_void_p]:
    """Gets the memory address of the first element of a tenso

    Args:
        A (`Optional[Tensor]`): A PyTorch tensor.

    Returns:
        `Optional[ct.c_void_p]`: A pointer to the underlying tensor data.
    """
    if A is None:
        return None

    return ct.c_void_p(A.data_ptr())


class QuantState:
    """container for quantization state components to work with Params4bit and similar classes"""

    valid_quant_types = ("fp4", "nf4")
    valid_qs_type_keys = [f"bitsandbytes__{x}" for x in valid_quant_types]
    valid_qs_keys = [
        "absmax",
        "quant_map",
        "nested_absmax",
        "nested_quant_map",
        "quant_state",
        "quant_type",
        "blocksize",
        "dtype",
        "shape",
        "nested_blocksize",
        "nested_dtype",
        "nested_offset",
    ]

    def __init__(
        self,
        absmax,
        shape=None,
        code=None,
        blocksize=None,
        quant_type=None,
        dtype=None,
        offset=None,
        state2=None,
    ):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None

    def __getitem__(self, idx):
        """
        ensures compatibility with older quant state scheme with nested lists.
        assumes the following layout:
        state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
        state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
        """
        if self.nested:
            list_repr = [
                self.absmax,
                self.shape,
                self.dtype,
                self.blocksize,
                [self.offset, self.state2],
                self.quant_type,
            ]
        else:
            list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, None, self.quant_type]
        return list_repr[idx]

    @classmethod
    def from_dict(cls, qs_dict: dict[str, Any], device: torch.device) -> "QuantState":
        """
        unpacks components of state_dict into QuantState
        where necessary, convert into strings, torch.dtype, ints, etc.

        qs_dict: based on state_dict, with only relevant keys, striped of prefixes.

        item with key `quant_state.bitsandbytes__[nf4/fp4]` may contain minor and non-tensor quant state items.
        """

        # unpacking tensor with non-tensor components
        qs_key = [k for k, v in qs_dict.items() if "quant_state" in k and isinstance(v, torch.Tensor)]
        if not len(qs_key) and "quant_type" not in qs_dict:
            raise ValueError("Expected packed or unpacked quant_state items, found neither")
        elif len(qs_key) != 1 or qs_key[0].split(".")[-1] not in cls.valid_qs_type_keys:
            raise ValueError(
                f"There should be exactly one `quant_state` item with ending from {cls.valid_qs_type_keys}.\nDetected {qs_key}.",
            )

        # unpacking minor and non-tensor quant state items if necessary
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))

        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}  # strip prefixes
        assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)

        if "nested_absmax" in qs_dict:
            offset = torch.tensor(float(qs_dict["nested_offset"])).to(device)
            state2 = cls(
                absmax=qs_dict["nested_absmax"].to(device),
                blocksize=qs_dict["nested_blocksize"],
                code=qs_dict["nested_quant_map"].to(device),
                dtype=getattr(torch, qs_dict["nested_dtype"]),
            )
        else:
            offset, state2 = None, None

        quant_state = cls(
            quant_type=qs_dict["quant_type"],
            absmax=qs_dict["absmax"].to(device),
            blocksize=qs_dict["blocksize"],
            code=qs_dict["quant_map"].to(device),
            dtype=getattr(torch, qs_dict["dtype"]),
            shape=torch.Size(qs_dict["shape"]) if qs_dict["shape"] is not None else None,
            offset=offset,
            state2=state2,
        )
        return quant_state

    def as_dict(self, packed=False):
        """
        returns dict of tensors and strings to use in serialization via _save_to_state_dict()
        param: packed -- returns dict[str, torch.Tensor] for state_dict fit for safetensors saving
        """
        qs_dict = {
            "quant_type": self.quant_type,
            "absmax": self.absmax,
            "blocksize": self.blocksize,
            "quant_map": self.code,
            "dtype": str(self.dtype).strip("torch."),
            "shape": tuple(self.shape),
        }
        if self.nested:
            qs_dict.update(
                {
                    "nested_absmax": self.state2.absmax,
                    "nested_blocksize": self.state2.blocksize,
                    "nested_quant_map": self.state2.code.clone(),  # un-shared to avoid restoring it after shared tensors are removed by safetensors
                    "nested_dtype": str(self.state2.dtype).strip("torch."),
                    "nested_offset": self.offset.item(),
                },
            )
        if not packed:
            return qs_dict

        # packed format allows serialization of non-tensor components, critical for saving in safetensors format
        qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
        non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
        qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = pack_dict_to_tensor(non_tensor_dict)
        return qs_packed_dict

    def to(self, device):
        # make sure the quantization state is on the right device
        self.code = self.code.to(device)
        self.absmax = self.absmax.to(device)
        if self.nested:
            self.offset = self.offset.to(device)
            self.state2.absmax = self.state2.absmax.to(device)
            self.state2.code = self.state2.code.to(device)

    def __eq__(self, other):
        if not isinstance(other, QuantState):
            return False

        return (
            torch.allclose(self.absmax, other.absmax, atol=1e-6)
            and self.shape == other.shape
            and torch.allclose(self.code, other.code, atol=1e-6)
            and self.dtype == other.dtype
            and self.blocksize == other.blocksize
            and self.quant_type == other.quant_type
            and (
                self.offset == other.offset
                if self.offset is not None and other.offset is not None
                else self.offset is other.offset
            )
            and (
                self.state2 == other.state2
                if self.state2 is not None and other.state2 is not None
                else self.state2 is other.state2
            )
        )


def quantize_blockwise(
    A: torch.Tensor,
    code: Optional[torch.Tensor] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=4096,
    nested=False,
) -> tuple[torch.Tensor, QuantState]:
    """Quantize a tensor in blocks of values.

    The input tensor is quantized by dividing it into blocks of `blocksize` values.
    The the absolute maximum value within these blocks is calculated for scaling
    the non-linear quantization.

    Args:
        A (`torch.Tensor`): The input tensor. Supports `float16`, `bfloat16`, or `float32` datatypes.
        code (`torch.Tensor`, *optional*):
            A mapping describing the low-bit data type. Defaults to a signed 8-bit dynamic type.
            For more details, see  (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561].
        absmax (`torch.Tensor`, *optional*): A tensor to use to store the absmax values.
        out (`torch.Tensor`, *optional*): A tensor to use to store the result.
        blocksize (`int`, *optional*):
            The size of the blocks. Defaults to 4096.
            Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
        nested (`bool`, *optional*): Whether to additionally quantize the absmax values. Defaults to False.

    Raises:
        ValueError: Raised when the input data type is not supported.

    Returns:
        `Tuple[torch.Tensor, QuantState]`: A tuple containing the quantization results.
        - `torch.Tensor`: The quantized tensor.
        - [`QuantState`]: The state object used to undo the quantization.
    """

    if code is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]

    _out, _absmax = torch.ops.bitsandbytes.quantize_blockwise.default(
        A,
        code.to(A.device),
        blocksize,
    )

    if nested:
        offset = _absmax.mean()
        _absmax -= offset
        qabsmax, state2 = quantize_blockwise(_absmax, blocksize=blocksize, nested=False)
        quant_state = QuantState(
            absmax=qabsmax,
            code=code.to(A.device, copy=True),
            blocksize=blocksize,
            dtype=A.dtype,
            offset=offset,
            state2=state2,
        )
    else:
        quant_state = QuantState(absmax=_absmax, code=code.to(A.device, copy=True), blocksize=blocksize, dtype=A.dtype)

    # TODO(matthewdouglas): Deprecate out kwarg
    out = out.copy_(_out) if out is not None else _out

    # TODO(matthewdouglas): Deprecate absmax kwarg
    if absmax is not None:
        quant_state.absmax = absmax.copy_(quant_state.absmax)

    return out, quant_state


def dequantize_blockwise(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 4096,
    nested=False,
) -> torch.Tensor:
    """Dequantize a tensor in blocks of values.

    The input tensor is dequantized by dividing it into blocks of `blocksize` values.
    The the absolute maximum value within these blocks is used for scaling
    the non-linear dequantization.

    Args:
        A (`torch.Tensor`): The quantized input tensor.
        quant_state ([`QuantState`], *optional*):
            The quantization state as returned by [`quantize_blockwise`].
            Required if `absmax` is not provided.
        absmax (`torch.Tensor`, *optional*):
            A tensor containing the scaling values.
            Required if `quant_state` is not provided and ignored otherwise.
        code (`torch.Tensor`, *optional*):
            A mapping describing the low-bit data type. Defaults to a signed 8-bit dynamic type.
            For more details, see  (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561].
            Ignored when `quant_state` is provided.
        out (`torch.Tensor`, *optional*): A tensor to use to store the result.
        blocksize (`int`, *optional*):
            The size of the blocks. Defaults to 4096.
            Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
            Ignored when `quant_state` is provided.

    Raises:
        ValueError: Raised when the input data type is not supported.

    Returns:
        `torch.Tensor`:
            The dequantized tensor. The datatype is indicated by `quant_state.dtype` and defaults to `torch.float32`.
    """

    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]

    if quant_state is None:
        quant_state = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=torch.float32)

    absmax = quant_state.absmax
    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()

    if out is not None:
        torch.ops.bitsandbytes.dequantize_blockwise.out(
            A,
            absmax,
            quant_state.code.to(A.device),
            quant_state.blocksize,
            quant_state.dtype,
            out=out,
        )
        return out

    return torch.ops.bitsandbytes.dequantize_blockwise.default(
        A,
        absmax,
        quant_state.code.to(A.device),
        quant_state.blocksize,
        quant_state.dtype,
    )


def get_4bit_type(typename, device=None, blocksize=64):
    if device is None:
        device = "cuda"
    data = None
    if typename == "nf4":
        """ Implements the NF4 data type.

            Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
            is normalized into the range [-1, 1].

            For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

            Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
            the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
        """
        data = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ]
    elif typename == "fp4":
        # 0b000 = 0
        # 0b001 = 0.0625
        # 0b010 = 8
        # 0b011 = 12
        # 0b100 = 4
        # 0b101 = 6
        # 0b110 = 2
        # 0b111 = 3
        # can also be created with bnb.functional.create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4)
        data = [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
    elif typename == "int4":
        data = [7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7]
    elif typename == "af4":
        # Taken from: NF4 Isn't Information Theoretically Optimal (and that's Good)
        # https://arxiv.org/abs/2306.06965
        if blocksize == 64:
            data = [
                -1.0,
                -0.69441008,
                -0.51243739,
                -0.3736951,
                -0.25607552,
                -0.14982478,
                -0.04934812,
                0.0,
                0.04273164,
                0.12934483,
                0.21961274,
                0.31675666,
                0.42563882,
                0.55496234,
                0.72424863,
                1.0,
            ][::-1]
        else:
            raise NotImplementedError("4-bit AbnormalFloats currently only support blocksize 64.")

    if data is None:
        raise NotImplementedError(f"Typename {typename} not supported")

    data = torch.tensor(data, device=device)
    data.div_(data.abs().max())

    assert data.numel() == 16

    return data


def quantize_fp4(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=None,
    compress_statistics=False,
    quant_storage=torch.uint8,
):
    if blocksize is None:
        blocksize = 64 if not HIP_ENVIRONMENT else 128
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "fp4", quant_storage)


def quantize_nf4(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=None,
    compress_statistics=False,
    quant_storage=torch.uint8,
):
    if blocksize is None:
        blocksize = 64 if not HIP_ENVIRONMENT else 128
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "nf4", quant_storage)


def quantize_4bit(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=None,
    compress_statistics=False,
    quant_type="fp4",
    quant_storage=torch.uint8,
) -> tuple[torch.Tensor, QuantState]:
    """Quantize tensor A in blocks of 4-bit values.

    Quantizes tensor A by dividing it into blocks which are independently quantized.

    Args:
        A (`torch.Tensor`): The input tensor. Supports `float16`, `bfloat16`, or `float32` datatypes.
        absmax (`torch.Tensor`, *optional*): A tensor to use to store the absmax values.
        out (`torch.Tensor`, *optional*): A tensor to use to store the result.
        blocksize (`int`, *optional*):
            The size of the blocks. Defaults to 128 on ROCm and 64 otherwise.
            Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
        compress_statistics (`bool`, *optional*): Whether to additionally quantize the absmax values. Defaults to False.
        quant_type (`str`, *optional*): The data type to use: `nf4` or `fp4`. Defaults to `fp4`.
        quant_storage (`torch.dtype`, *optional*): The dtype of the tensor used to store the result. Defaults to `torch.uint8`.

    Raises:
        ValueError: Raised when the input data type is not supported.

    Returns:
        Tuple[`torch.Tensor`, `QuantState`]: A tuple containing the quantization results.
        - `torch.Tensor`: The quantized tensor with packed 4-bit values.
        - [`QuantState`]: The state object used to undo the quantization.
    """

    if blocksize is None:
        blocksize = 64 if not HIP_ENVIRONMENT else 128

    input_shape = A.shape

    _out, _absmax = torch.ops.bitsandbytes.quantize_4bit.default(
        A,
        blocksize,
        quant_type,
        quant_storage,
    )

    code = get_4bit_type(quant_type, device=A.device)

    if compress_statistics:
        offset = _absmax.mean()
        qabsmax, state2 = quantize_blockwise(_absmax - offset, blocksize=256)
        del _absmax
        state = QuantState(
            absmax=qabsmax,
            shape=input_shape,
            dtype=A.dtype,
            blocksize=blocksize,
            code=code,
            quant_type=quant_type,
            offset=offset,
            state2=state2,
        )
    else:
        state = QuantState(
            absmax=_absmax,
            shape=input_shape,
            dtype=A.dtype,
            blocksize=blocksize,
            code=code,
            quant_type=quant_type,
        )

    # TODO(matthewdouglas): Deprecate out kwarg
    out = out.copy_(_out) if out is not None else _out

    # TODO(matthewdouglas): Deprecate absmax kwarg
    if absmax is not None:
        state.absmax = absmax.copy_(state.absmax)

    return out, state


def dequantize_fp4(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
) -> torch.Tensor:
    if blocksize is None:
        blocksize = 64 if not HIP_ENVIRONMENT else 128
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "fp4")


def dequantize_nf4(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
) -> torch.Tensor:
    if blocksize is None:
        blocksize = 64 if not HIP_ENVIRONMENT else 128
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "nf4")


def dequantize_4bit(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    quant_type="fp4",
) -> torch.Tensor:
    """Dequantizes a packed 4-bit quantized tensor.

    The input tensor is dequantized by dividing it into blocks of `blocksize` values.
    The the absolute maximum value within these blocks is used for scaling
    the non-linear dequantization.

    Args:
        A (`torch.Tensor`): The quantized input tensor.
        quant_state ([`QuantState`], *optional*):
            The quantization state as returned by [`quantize_4bit`].
            Required if `absmax` is not provided.
        absmax (`torch.Tensor`, *optional*):
            A tensor containing the scaling values.
            Required if `quant_state` is not provided and ignored otherwise.
        out (`torch.Tensor`, *optional*): A tensor to use to store the result.
        blocksize (`int`, *optional*):
            The size of the blocks. Defaults to 128 on ROCm and 64 otherwise.
            Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
        quant_type (`str`, *optional*): The data type to use: `nf4` or `fp4`. Defaults to `fp4`.

    Raises:
        ValueError: Raised when the input data type or blocksize is not supported.

    Returns:
        `torch.Tensor`: The dequantized tensor.
    """

    if blocksize is None:
        blocksize = 64 if not HIP_ENVIRONMENT else 128

    if quant_state is None:
        assert absmax is not None and out is not None

        quant_state = QuantState(
            absmax=absmax,
            shape=out.shape,
            dtype=out.dtype,
            blocksize=blocksize,
            quant_type=quant_type,
        )

    else:
        absmax = quant_state.absmax

    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()

    if out is not None:
        torch.ops.bitsandbytes.dequantize_4bit.out(
            A, absmax, quant_state.blocksize, quant_state.quant_type, quant_state.shape, quant_state.dtype, out=out
        )
    else:
        out = torch.ops.bitsandbytes.dequantize_4bit.default(
            A,
            absmax,
            quant_state.blocksize,
            quant_state.quant_type,
            quant_state.shape,
            quant_state.dtype,
        )

    if A.shape[0] == 1:  # is transposed, transpose back
        return out.t()
    return out


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def quantize(
    A: Tensor,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    if code is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
        code = code.to(A.device)

    absmax = torch.abs(A).max()
    if absmax.dtype != torch.float32:
        absmax = absmax.float()
    inp = A / absmax
    out = quantize_no_absmax(inp, code, out)
    return out, (absmax, code)


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def dequantize(
    A: Tensor,
    state: Optional[tuple[Tensor, Tensor]] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tensor:
    assert state is not None or absmax is not None
    if code is None and state is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
        code = code.to(A.device)

    if state is None:
        state = (absmax, code)
    out = dequantize_no_absmax(A, state[1], out)
    return out * state[0]


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def quantize_no_absmax(A: Tensor, code: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    """
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
    """
    with _cuda_device_of(A):
        if out is None:
            out = torch.zeros_like(A, dtype=torch.uint8)
        is_on_gpu([A, out])
        lib.cquantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))

    return out


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def dequantize_no_absmax(A: Tensor, code: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    """
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
    """
    with _cuda_device_of(A):
        if out is None:
            out = torch.zeros_like(A, dtype=torch.float32)
        is_on_gpu([code, A, out])
        stream = _get_tensor_stream(A)
        lib.cdequantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()), stream)

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
    state2: Optional[torch.Tensor] = None,
    beta2: float = 0.0,
    beta3: float = 0.0,
    alpha: float = 0.0,
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Optional[torch.Tensor] = None,
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
    beta3 : float
        Optimizer beta3.
    alpha : float
        Optimizer alpha.
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

    is_on_gpu([g, p, state1, state2, unorm_vec])
    torch.ops.bitsandbytes.optimizer_update_32bit(
        optimizer_name,
        g,
        p,
        state1,
        state2,
        unorm_vec,
        max_unorm,
        param_norm,
        beta1,
        beta2,
        beta3,
        alpha,
        eps,
        weight_decay,
        step,
        lr,
        gnorm_scale,
        skip_zeros,
    )


@deprecated(
    "This function is deprecated and will be removed in a future release. "
    "Please use optimizer_update_8bit_blockwise instead. ",
    category=FutureWarning,
)
def optimizer_update_8bit(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: Tensor,
    qmap2: Optional[torch.Tensor],
    max1: Tensor,
    max2: Optional[torch.Tensor],
    new_max1: Tensor,
    new_max2: Optional[torch.Tensor],
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Optional[torch.Tensor] = None,
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

    with _cuda_device_of(g):
        is_on_gpu([g, p, state1, state2, unorm_vec, qmap1, qmap2, max1, max2, new_max1, new_max2])
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
                f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}",
            )


def optimizer_update_8bit_blockwise(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: Tensor,
    qmap2: Optional[torch.Tensor],
    absmax1: Tensor,
    absmax2: Optional[torch.Tensor],
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    skip_zeros=False,
) -> None:
    is_on_gpu([p, g, state1, state2, qmap1, qmap2, absmax1, absmax2])

    torch.ops.bitsandbytes.optimizer_update_8bit_blockwise(
        optimizer_name,
        g,
        p,
        state1,
        state2,
        beta1,
        beta2,
        beta3,
        alpha,
        eps,
        step,
        lr,
        qmap1,
        qmap2,
        absmax1,
        absmax2,
        weight_decay,
        gnorm_scale,
        skip_zeros,
    )


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def percentile_clipping(grad: Tensor, gnorm_vec: Tensor, step: int, percentile: int = 5):
    """Applies percentile clipping

    grad: torch.Tensor
        The gradient tensor.
    gnorm_vec: torch.Tensor
        Vector of gradient norms. 100 elements expected.
    step: int
        The current optimization steps (number of past gradient norms).

    """
    with _cuda_device_of(grad):
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


def check_matmul(A, B, out, transposed_A, transposed_B, expected_type=torch.int8):
    if not torch.cuda.is_initialized():
        torch.cuda.init()
    if A.dtype != expected_type or B.dtype != expected_type:
        raise TypeError(f"Expected torch.int8 input tensors A and B, but got {A.dtype} and {B.dtype}")

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
            if sout[0] == sA[2] and sout[1] == sB[2] and sA[0] == sB[0] and sA[1] == sB[1]:
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
            f"Tensor dimensions incorrect for matrix mulitiplication: A x B: {sA} x {sB} with transpose for A x B: {tA} x {tB}.",
        )

    return sout


def gemv_4bit(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
    state=None,
):
    if state is None:
        raise ValueError("state cannot be None. gemv_4bit() requires the state from quantize_4bit()")

    absmax = state.absmax
    if state.nested:
        absmax = dequantize_blockwise(absmax, state.state2) + state.offset

    if out is not None:
        torch.ops.bitsandbytes.gemv_4bit.out(
            A,
            B,
            state.shape,
            absmax,
            state.code,
            state.blocksize,
            out=out,
        )
        return out

    return torch.ops.bitsandbytes.gemv_4bit.default(
        A,
        B,
        state.shape,
        absmax,
        state.code,
        state.blocksize,
    )


def igemm(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
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
                f"Only bsi,bso->io supported for tensor contractions, but dims for A x B were: {sA} x {sB}",
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
    lib.cigemm(
        ptr,
        ct.c_bool(transposed_B),
        ct.c_bool(transposed_A),
        ct.c_int32(m),
        ct.c_int32(n),
        ct.c_int32(k),
        get_ptr(B),
        get_ptr(A),
        get_ptr(out),
        ct.c_int32(lda),
        ct.c_int32(ldb),
        ct.c_int32(ldc),
    )
    return out


def batched_igemm(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
):
    if not len(A.shape) == 3 or not len(B.shape) == 3:
        raise ValueError(f"Expected 3-dimensional tensors for bmm, but got shapes A and B: {A.shape} and {B.shape}")
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
    lib.cbatched_igemm(
        ptr,
        ct.c_bool(transposed_B),
        ct.c_bool(transposed_A),
        ct.c_int32(m),
        ct.c_int32(n),
        ct.c_int32(k),
        get_ptr(B),
        get_ptr(A),
        get_ptr(out),
        ct.c_int32(lda),
        ct.c_int32(ldb),
        ct.c_int32(ldc),
        ct.c_long(strideA),
        ct.c_long(strideB),
        ct.c_long(strideC),
        ct.c_uint32(num_batch),
    )
    return out


def int8_linear_matmul(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    """Performs an 8-bit integer matrix multiplication.

    A linear transformation is applied such that `out = A @ B.T`. When possible, integer tensor core hardware is
    utilized to accelerate the operation.

    Args:
        A (`torch.Tensor`): The first matrix operand with the data type `torch.int8`.
        B (`torch.Tensor`): The second matrix operand with the data type `torch.int8`.
        out (`torch.Tensor`, *optional*): A pre-allocated tensor used to store the result.
        dtype (`torch.dtype`, *optional*): The expected data type of the output. Defaults to `torch.int32`.

    Raises:
        `NotImplementedError`: The operation is not supported in the current environment.
        `RuntimeError`: Raised when the cannot be completed for any other reason.

    Returns:
        `torch.Tensor`: The result of the operation.
    """
    if out is not None:
        torch.ops.bitsandbytes.int8_linear_matmul.out(A, B, out)
        return out

    return torch.ops.bitsandbytes.int8_linear_matmul.default(A, B)


def int8_mm_dequant(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
):
    """Performs dequantization on the result of a quantized int8 matrix multiplication.

    Args:
        A (`torch.Tensor` with dtype `torch.int32`): The result of a quantized int8 matrix multiplication.
        row_stats (`torch.Tensor`): The row-wise quantization statistics for the lhs operand of the matrix multiplication.
        col_stats (`torch.Tensor`): The column-wise quantization statistics for the rhs operand of the matrix multiplication.
        out (`torch.Tensor`, *optional*): A pre-allocated tensor to store the output of the operation.
        bias (`torch.Tensor`, *optional*): An optional bias vector to add to the result.

    Returns:
        `torch.Tensor`: The dequantized result with an optional bias, with dtype `torch.float16`.
    """
    result = torch.ops.bitsandbytes.int8_mm_dequant.default(A, row_stats, col_stats, dtype=torch.float16, bias=bias)

    # TODO(matthewdouglas): Deprecate out kwarg
    if out is not None:
        return out.copy_(result)

    return result


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def get_colrow_absmax(
    A: torch.Tensor,
    row_stats: Optional[torch.Tensor] = None,
    col_stats: Optional[torch.Tensor] = None,
    nnz_block_ptr: Optional[torch.Tensor] = None,
    threshold=0.0,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """ "Determine the quantization statistics for input matrix `A` in accordance to the `LLM.int8()` algorithm.

    The row-wise and column-wise absmax values are determined.

    For more information, see the [LLM.int8() paper](https://arxiv.org/abs/2208.07339).

    <Tip>
    This function is useful for training, but for inference it is advised to use [`get_row_absmax`] instead.
    The column-wise quantization scales are not typically needed in inference scenarios.
    </Tip>

    Args:
        A (`torch.Tensor` with dtype `torch.float16`): Input tensor.
        row_stats (`torch.Tensor`, *optional*): If provided, calculation of row statistics is skipped.
        col_stats (`torch.Tensor`, *optional*): If provided, calculation of column statistics is skipped.
        nnz_block_ptr (`torch.Tensor`, *optional*): Not used.
        threshold (`float`, *optional*):
            An optional threshold for sparse decomposition of outlier features.
            No outliers are held back when 0.0. Defaults to 0.0.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: A tuple containing quantization statistics.
        - `torch.Tensor` with dtype `torch.float32`: The row-wise quantization statistics.
        - `torch.Tensor` with dtype `torch.float32`: The column-wise quantization statistics.
        - `torch.Tensor` with dtype `torch.bool`, *optional*: A mask indicating the locations of outliers in the input tensor.
    """
    assert A.is_floating_point()

    outlier_mask = None

    if row_stats is None or col_stats is None:
        absA = A.abs().view(-1, A.shape[-1])

        if threshold > 0.0:
            # Filter outliers from stats when enabled
            outlier_mask = absA >= threshold
            absA.masked_fill_(outlier_mask, 0.0)

        if row_stats is None:
            # shape [rows]; unsqueeze(-1) gives [rows,1]
            # We have a CUDA kernel for row max, but not yet for cols.
            row_stats = get_row_absmax(A, threshold)

        if col_stats is None:
            # shape [cols]; unsqueeze(0) gives [1,cols]
            col_stats = absA.amax(dim=0, keepdim=False).float()

    return row_stats, col_stats, outlier_mask


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def get_row_absmax(A: torch.Tensor, threshold=0.0):
    """Determine the quantization statistics for input matrix `A` in accordance to the `LLM.int8()` algorithm.

    For more information, see the [LLM.int8() paper](https://arxiv.org/abs/2208.07339).

    Args:
        A (`torch.Tensor` with dtype `torch.float16`): The input matrix.
        threshold (`float`, *optional*):
            An optional threshold for sparse decomposition of outlier features.
            No outliers are held back when 0.0. Defaults to 0.0.

    Returns:
        `torch.Tensor` with dtype `torch.float32`: The absolute maximum value for each row, with outliers ignored.
    """

    assert A.dtype == torch.float16

    rows = prod(A.shape[:-1])
    cols = A.shape[-1]

    row_stats = torch.empty((rows,), dtype=torch.float32, device=A.device)

    is_on_gpu([A])

    with _cuda_device_of(A):
        lib.cget_row_stats(
            get_ptr(A),
            get_ptr(row_stats),
            ct.c_float(threshold),
            ct.c_int32(rows),
            ct.c_int32(cols),
            _get_tensor_stream(A),
        )

    return row_stats


class COOSparseTensor:
    def __init__(
        self, rows: int, cols: int, nnz: int, rowidx: torch.Tensor, colidx: torch.Tensor, values: torch.Tensor
    ):
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
    rowptr = torch.zeros((cooA.rows + 1,), dtype=torch.int32, device=cooA.rowidx.device)
    rowptr.scatter_(index=values.long(), src=counts.int(), dim=0)
    rowptr.cumsum_(0)
    return CSRSparseTensor(cooA.rows, cooA.cols, cooA.nnz, rowptr, cooA.colidx, cooA.values)


def coo2csc(cooA):
    val, col2rowidx = torch.sort(cooA.colidx)
    rowidx = cooA.rowidx[col2rowidx]
    values = cooA.values[col2rowidx]
    colvalues, counts = torch.unique(val, return_counts=True)
    colvalues.add_(1)
    colptr = torch.zeros((cooA.cols + 1,), dtype=torch.int32, device=cooA.colidx.device)
    colptr.scatter_(index=colvalues.long(), src=counts.int(), dim=0)
    colptr.cumsum_(0)
    return CSCSparseTensor(cooA.rows, cooA.cols, cooA.nnz, colptr, rowidx, values)


def coo_zeros(rows, cols, nnz, device, dtype=torch.half):
    rowidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
    colidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
    values = torch.zeros((nnz,), dtype=dtype, device=device)
    return COOSparseTensor(rows, cols, nnz, rowidx, colidx, values)


def int8_double_quant(
    A: torch.Tensor,
    col_stats: Optional[torch.Tensor] = None,
    row_stats: Optional[torch.Tensor] = None,
    out_col: Optional[torch.Tensor] = None,
    out_row: Optional[torch.Tensor] = None,
    threshold=0.0,
):
    """Determine the quantization statistics for input matrix `A` in accordance to the `LLM.int8()` algorithm.

    The statistics are determined both row-wise and column-wise (transposed).

    For more information, see the [LLM.int8() paper](https://arxiv.org/abs/2208.07339).

    <Tip>
    This function is useful for training, but for inference it is advised to use [`int8_vectorwise_quant`] instead.
    This implementation performs additional column-wise transposed calculations which are not optimized.
    </Tip>

    Args:
        A (`torch.Tensor` with dtype `torch.float16`): The input matrix.
        col_stats (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the column-wise quantization scales.
        row_stats (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the row-wise quantization scales.
        out_col (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the column-wise quantized data.
        out_row (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the row-wise quantized data.
        threshold (`float`, *optional*):
            An optional threshold for sparse decomposition of outlier features.

            No outliers are held back when 0.0. Defaults to 0.0.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: A tuple containing the quantized tensor and relevant statistics.
        - `torch.Tensor` with dtype `torch.int8`: The row-wise quantized data.
        - `torch.Tensor` with dtype `torch.int8`: The column-wise quantized data.
        - `torch.Tensor` with dtype `torch.float32`: The row-wise quantization scales.
        - `torch.Tensor` with dtype `torch.float32`: The column-wise quantization scales.
        - `torch.Tensor` with dtype `torch.int32`, *optional*: A list of column indices which contain outlier features.
    """

    if row_stats is not None:
        raise ValueError("row_stats must be None. int8_double_quant() does not support pre-allocated row_stats.")
    if col_stats is not None:
        raise ValueError("col_stats must be None. int8_double_quant() does not support pre-allocated col_stats.")
    if out_col is not None:
        raise ValueError("out_col must be None. int8_double_quant() does not support pre-allocated out_col.")
    if out_row is not None:
        raise ValueError("out_row must be None. int8_double_quant() does not support pre-allocated out_row.")

    return torch.ops.bitsandbytes.int8_double_quant.default(A, threshold=threshold)


def int8_vectorwise_dequant(A: torch.Tensor, stats: torch.Tensor):
    """Dequantizes a tensor with dtype `torch.int8` to `torch.float32`.

    Args:
        A (`torch.Tensor` with dtype `torch.int8`): The quantized int8 tensor.
        stats (`torch.Tensor` with dtype `torch.float32`): The row-wise quantization statistics.

    Returns:
        `torch.Tensor` with dtype `torch.float32`: The dequantized tensor.
    """
    # To dequantize we divide by 127, or multiply by the reciprocal.
    return torch.ops.bitsandbytes.int8_vectorwise_dequant.default(A, stats)


def int8_vectorwise_quant(A: torch.Tensor, threshold=0.0):
    """Quantizes a tensor with dtype `torch.float16` to `torch.int8` in accordance to the `LLM.int8()` algorithm.

    For more information, see the [LLM.int8() paper](https://arxiv.org/abs/2208.07339).

    Args:
        A (`torch.Tensor` with dtype `torch.float16`): The input tensor.
        threshold (`float`, *optional*):
            An optional threshold for sparse decomposition of outlier features.

            No outliers are held back when 0.0. Defaults to 0.0.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: A tuple containing the quantized tensor and relevant statistics.
        - `torch.Tensor` with dtype `torch.int8`: The quantized data.
        - `torch.Tensor` with dtype `torch.float32`: The quantization scales.
        - `torch.Tensor` with dtype `torch.int32`, *optional*: A list of column indices which contain outlier features.
    """
    return torch.ops.bitsandbytes.int8_vectorwise_quant.default(A, threshold)


def spmm_coo(
    cooA: Union[COOSparseTensor, torch.Tensor],
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
):
    if not isinstance(cooA, COOSparseTensor):
        assert cooA.is_sparse and cooA.layout == torch.sparse_coo, (
            "Tensor must be `COOSparseTensor or a PyTorch COO tensor."
        )

        # Convert to custom COOSparseTensor
        cooA = COOSparseTensor(
            rows=cooA.shape[0],
            cols=cooA.shape[1],
            nnz=cooA._nnz(),
            rowidx=cooA.indices()[0].int(),
            colidx=cooA.indices()[1].int(),
            values=cooA.values(),
        )

    if out is None:
        out = torch.empty((cooA.rows, B.shape[1]), device=B.device, dtype=B.dtype)
    nnz = cooA.nnz
    assert cooA.rowidx.numel() == nnz
    assert cooA.colidx.numel() == nnz
    assert cooA.values.numel() == nnz
    assert cooA.cols == B.shape[0]

    transposed_B = not B.is_contiguous()

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
    lib.cspmm_coo(
        ptr,
        ptrRowidx,
        ptrColidx,
        ptrValues,
        cnnz,
        crowsA,
        ccolsA,
        ccolsB,
        cldb,
        ptrB,
        cldc,
        ptrC,
        ct.c_bool(transposed_B),
    )

    return out


def spmm_coo_very_sparse(cooA, B, dequant_stats=None, out=None):
    if out is None:
        out = torch.zeros((cooA.rows, B.shape[1]), device=B.device, dtype=cooA.values.dtype)
    nnz = cooA.nnz

    assert cooA.rowidx.numel() == nnz
    assert cooA.colidx.numel() == nnz
    assert cooA.values.numel() == nnz
    assert cooA.cols == B.shape[0], f"{cooA.cols} vs {B.shape}"

    _, counts = torch.unique(cooA.rowidx, return_counts=True)
    offset = counts.cumsum(0).int()
    max_count, max_idx = torch.sort(counts, descending=True)
    max_idx = max_idx.int()
    max_count = max_count.int()
    assert max_count[0] <= 32, f"Current max count per row is 8 but found {max_count[0]}."
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
    crowsB = ct.c_int32(B.shape[1])
    ccolsB = ct.c_int32(B.shape[1])

    with _cuda_device_of(B):
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
