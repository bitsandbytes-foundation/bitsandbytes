# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import Any, Optional, TypeVar, Union, overload
import warnings

import torch
from torch import Tensor, device, dtype, nn
import torch.nn.functional as F

import bitsandbytes as bnb
from bitsandbytes.functional import (
    QuantState,
    _convert_weight_packed_for_cpu,
    _convert_weight_packed_for_cpu_inverse,
    has_avx512bf16,
)
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING, OutlierTracer

T = TypeVar("T", bound="torch.nn.Module")


class StableEmbedding(torch.nn.Embedding):
    """
    Custom embedding layer designed to improve stability during training for NLP tasks by using 32-bit optimizer states. It is designed to reduce gradient variations that can result from quantization. This embedding layer is initialized with Xavier uniform initialization followed by layer normalization.

    Example:

    ```
    # Initialize StableEmbedding layer with vocabulary size 1000, embedding dimension 300
    embedding_layer = StableEmbedding(num_embeddings=1000, embedding_dim=300)

    # Reset embedding parameters
    embedding_layer.reset_parameters()

    # Perform a forward pass with input tensor
    input_tensor = torch.tensor([1, 2, 3])
    output_embedding = embedding_layer(input_tensor)
    ```

    Attributes:
        norm (`torch.nn.LayerNorm`): Layer normalization applied after the embedding.

    Methods:
        reset_parameters(): Reset embedding parameters using Xavier uniform initialization.
        forward(input: Tensor) -> Tensor: Forward pass through the stable embedding layer.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        """
        Args:
            num_embeddings (`int`):
                The number of unique embeddings (vocabulary size).
            embedding_dim (`int`):
                The dimensionality of the embedding.
            padding_idx (`Optional[int]`):
                Pads the output with zeros at the given index.
            max_norm (`Optional[float]`):
                Renormalizes embeddings to have a maximum L2 norm.
            norm_type (`float`, defaults to `2.0`):
                The p-norm to compute for the `max_norm` option.
            scale_grad_by_freq (`bool`, defaults to `False`):
                Scale gradient by frequency during backpropagation.
            sparse (`bool`, defaults to `False`):
                Computes dense gradients. Set to `True` to compute sparse gradients instead.
            _weight (`Optional[Tensor]`):
                Pretrained embeddings.
        """
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device,
            dtype,
        )
        self.norm = torch.nn.LayerNorm(embedding_dim, device=device)
        GlobalOptimManager.get_instance().register_module_override(self, "weight", {"optim_bits": 32})

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)
        self._fill_padding_idx_with_zero()

    """ !!! This is a redefinition of _fill_padding_idx_with_zero in torch.nn.Embedding
        to make the Layer compatible with Pytorch < 1.9.
        This means that if this changes in future PyTorch releases this need to change too
        which is cumbersome. However, with this we can ensure compatibility with previous
        PyTorch releases.
    """

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        emb = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        # always apply layer norm in full precision
        emb = emb.to(torch.get_default_dtype())

        return self.norm(emb).to(self.weight.dtype)


class Embedding(torch.nn.Embedding):
    """
    Embedding class to store and retrieve word embeddings from their indices.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        device: Optional[device] = None,
    ) -> None:
        """
        Args:
            num_embeddings (`int`):
                The number of unique embeddings (vocabulary size).
            embedding_dim (`int`):
                The dimensionality of the embedding.
            padding_idx (`Optional[int]`):
                Pads the output with zeros at the given index.
            max_norm (`Optional[float]`):
                Renormalizes embeddings to have a maximum L2 norm.
            norm_type (`float`, defaults to `2.0`):
                The p-norm to compute for the `max_norm` option.
            scale_grad_by_freq (`bool`, defaults to `False`):
                Scale gradient by frequency during backpropagation.
            sparse (`bool`, defaults to `False`):
                Computes dense gradients. Set to `True` to compute sparse gradients instead.
            _weight (`Optional[Tensor]`):
                Pretrained embeddings.
        """
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device=device,
        )
        GlobalOptimManager.get_instance().register_module_override(self, "weight", {"optim_bits": 32})

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)
        self._fill_padding_idx_with_zero()

    """ !!! This is a redefinition of _fill_padding_idx_with_zero in torch.nn.Embedding
        to make the Layer compatible with Pytorch < 1.9.
        This means that if this changes in future PyTorch releases this need to change too
        which is cumbersome. However, with this we can ensure compatibility with previous
        PyTorch releases.
    """

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        emb = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        return emb


class Params4bit(torch.nn.Parameter):
    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad=False,  # quantized weights should be frozen by default
        quant_state: Optional[QuantState] = None,
        blocksize: Optional[int] = None,
        compress_statistics: bool = True,
        quant_type: str = "fp4",
        quant_storage: torch.dtype = torch.uint8,
        module: Optional["Linear4bit"] = None,
        bnb_quantized: bool = False,
    ) -> "Params4bit":
        if data is None:
            data = torch.empty(0)

        if blocksize is None:
            blocksize = 64

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.bnb_quantized = bnb_quantized
        self.data = data
        self.module = module
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data"] = self.data
        state["requires_grad"] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.requires_grad = state["requires_grad"]
        self.blocksize = state["blocksize"]
        self.compress_statistics = state["compress_statistics"]
        self.quant_type = state["quant_type"]
        self.quant_state = state["quant_state"]
        self.data = state["data"]
        self.quant_storage = state["quant_storage"]
        self.bnb_quantized = state["bnb_quantized"]
        self.module = state["module"]

    # Map from state_dict key names (as produced by QuantState.as_dict) to
    # the actual QuantState attribute/access path. FSDP's _get_fqns() resolves
    # dotted FQN keys via getattr, so "weight.quant_map" becomes
    # getattr(weight, "quant_map") — we must map that to quant_state.code.
    _QUANT_STATE_ATTR_MAP = {
        # Direct QuantState attributes
        "absmax": lambda qs: qs.absmax,
        "code": lambda qs: qs.code,
        "blocksize": lambda qs: qs.blocksize,
        "dtype": lambda qs: qs.dtype,
        "shape": lambda qs: qs.shape,
        "offset": lambda qs: qs.offset,
        "state2": lambda qs: qs.state2,
        # as_dict serializes code → "quant_map"
        "quant_map": lambda qs: qs.code,
        "quant_type": lambda qs: qs.quant_type,
        # as_dict serializes nested state2 attributes under "nested_*" keys
        "nested_absmax": lambda qs: qs.state2.absmax,
        "nested_blocksize": lambda qs: qs.state2.blocksize,
        "nested_quant_map": lambda qs: qs.state2.code,
        "nested_dtype": lambda qs: qs.state2.dtype,
        "nested_offset": lambda qs: qs.offset,
    }

    def __getattr__(self, name):
        # Proxy known QuantState attributes so that PyTorch's FSDP state_dict
        # machinery (which traverses FQN paths via getattr) can find them.
        accessor = self._QUANT_STATE_ATTR_MAP.get(name)
        if accessor is not None:
            quant_state = self.__dict__.get("quant_state")
            if quant_state is not None:
                try:
                    return accessor(quant_state)
                except AttributeError:
                    pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.quant_state = copy.deepcopy(state["quant_state"])
        new_instance.data = copy.deepcopy(state["data"])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    @classmethod
    def from_prequantized(
        cls,
        data: torch.Tensor,
        quantized_stats: dict[str, Any],
        requires_grad: bool = False,
        device="cuda",
        module: Optional["Linear4bit"] = None,
        **kwargs,
    ) -> "Params4bit":
        self = torch.Tensor._make_subclass(cls, data.to(device))
        self.requires_grad = requires_grad
        self.quant_state = QuantState.from_dict(qs_dict=quantized_stats, device=device)
        self.blocksize = self.quant_state.blocksize
        self.compress_statistics = self.quant_state.nested
        self.quant_type = self.quant_state.quant_type
        self.bnb_quantized = True

        self.quant_storage = data.dtype
        self.module = module

        if self.module is not None:
            self.module.quant_state = self.quant_state

        return self

    def _quantize(self, device):
        w = self.data.contiguous().to(device)
        w_4bit, quant_state = bnb.functional.quantize_4bit(
            w,
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
        )
        self.data = w_4bit
        self.quant_state = quant_state
        if self.module is not None:
            self.module.quant_state = quant_state
        self.bnb_quantized = True
        return self

    def cpu(self):
        return self.to(device="cpu")

    def cuda(self, device: Optional[int | device | str] = None, non_blocking: bool = False):
        if getattr(self.quant_state, "packing_format_for_cpu", False):
            self.data, self.quant_state = _convert_weight_packed_for_cpu_inverse(self.data, self.quant_state)
        return self.to(device="cuda" if device is None else device, non_blocking=non_blocking)

    def xpu(self, device: Optional[int | device | str] = None, non_blocking: bool = False):
        if getattr(self.quant_state, "packing_format_for_cpu", False):
            self.data, self.quant_state = _convert_weight_packed_for_cpu_inverse(self.data, self.quant_state)
        return self.to(device="xpu" if device is None else device, non_blocking=non_blocking)

    @overload
    def to(
        self: T,
        device: Optional[int | device] = ...,
        dtype: Optional[dtype | str] = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: dtype | str, non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T: ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None and device.type != "meta" and not self.bnb_quantized:
            return self._quantize(device)
        else:
            if self.quant_state is not None:
                self.quant_state.to(device)

            new_param = Params4bit(
                super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
                quant_state=self.quant_state,
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                bnb_quantized=self.bnb_quantized,
            )

            return new_param

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in [torch.chunk, torch.split]:
            tensor = args[0]

            result = super().__torch_function__(func, types, args, kwargs)

            if isinstance(result, tuple):
                return tuple(
                    cls(
                        data=chunk,
                        requires_grad=tensor.requires_grad,
                        quant_state=tensor.quant_state,
                        blocksize=tensor.blocksize,
                        compress_statistics=tensor.compress_statistics,
                        quant_type=tensor.quant_type,
                        quant_storage=tensor.quant_storage,
                        module=tensor.module,
                        bnb_quantized=tensor.bnb_quantized,
                    )
                    for chunk in result
                )
            else:
                return cls(
                    data=result,
                    requires_grad=tensor.requires_grad,
                    quant_state=tensor.quant_state,
                    blocksize=tensor.blocksize,
                    compress_statistics=tensor.compress_statistics,
                    quant_type=tensor.quant_type,
                    quant_storage=tensor.quant_storage,
                    module=tensor.module,
                    bnb_quantized=tensor.bnb_quantized,
                )

        return super().__torch_function__(func, types, args, kwargs)


def fix_4bit_weight_quant_state_from_module(module: Union["Embedding4bit", "Linear4bit"]):
    if getattr(module.weight, "quant_state", None) is not None:
        return

    if getattr(module, "quant_state", None) is None:
        warnings.warn(
            "FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.",
        )

    # the quant state got lost when the parameter got converted. This happens for example for fsdp
    # since we registered the module, we can recover the state here
    assert module.weight.shape[1] == 1
    if not isinstance(module.weight, Params4bit):
        module.weight = Params4bit(module.weight, quant_storage=module.quant_storage, bnb_quantized=True)
    module.weight.quant_state = module.quant_state


class Linear4bit(nn.Linear):
    """
    This class is the base module for the 4-bit quantization algorithm presented in [QLoRA](https://arxiv.org/abs/2305.14314).
    QLoRA 4-bit linear layers uses blockwise k-bit quantization under the hood, with the possibility of selecting various
    compute datatypes such as FP4 and NF4.

    In order to quantize a linear layer one should first load the original fp16 / bf16 weights into
    the Linear4bit module, then call `quantized_module.to("cuda")` to quantize the fp16 / bf16 weights.

    Example:

    ```python
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit

    fp16_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64)
    )

    quantized_model = nn.Sequential(
        Linear4bit(64, 64),
        Linear4bit(64, 64)
    )

    quantized_model.load_state_dict(fp16_model.state_dict())
    quantized_model = quantized_model.to(0) # Quantization happens here
    ```
    """

    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype=None,
        compress_statistics=True,
        quant_type="fp4",
        quant_storage=torch.uint8,
        device=None,
    ):
        """
        Initialize Linear4bit class.

        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(input_features, output_features, bias, device)
        self.weight = Params4bit(
            self.weight.data,
            requires_grad=False,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
            quant_storage=quant_storage,
            module=self,
        )
        # self.persistent_buffers = []  # TODO consider as way to save quant state
        self.compute_dtype = compute_dtype
        self.compute_type_is_set = compute_dtype is not None
        self.quant_state = None
        self.quant_storage = quant_storage
        self.support_avx512bf16_for_cpu = has_avx512bf16()

    def set_compute_type(self, x):
        if x.dtype in [torch.float32, torch.bfloat16]:
            # the input is in a dtype that is safe to compute in, we switch
            # to this type for speed and stability
            self.compute_dtype = x.dtype
        elif x.dtype == torch.float16:
            # we take the compoute dtype passed into the layer
            if self.compute_dtype in [None, torch.float32] and (x.numel() == x.shape[-1]):
                # single batch inference with input torch.float16 and compute_dtype float32 -> slow inference when it could be fast
                # warn the user about this
                warnings.warn(
                    "Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference.",
                )
                warnings.filterwarnings("ignore", message=".*inference.")
            if self.compute_dtype in [None, torch.float32] and (x.numel() != x.shape[-1]):
                warnings.warn(
                    "Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed.",
                )
                warnings.filterwarnings("ignore", message=".*inference or training")

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        save weight and bias,
        then fill state_dict with components of quant_state
        """
        if getattr(self.weight, "quant_state", None) is not None and getattr(
            self.weight.quant_state, "packing_format_for_cpu", False
        ):
            self.weight.data, self.weight.quant_state = _convert_weight_packed_for_cpu_inverse(
                self.weight.data, self.weight.quant_state
            )
        super()._save_to_state_dict(destination, prefix, keep_vars)  # saving weight and bias
        if getattr(self.weight, "quant_state", None) is not None:
            for k, v in self.weight.quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()

    def forward(self, x: torch.Tensor):
        fix_4bit_weight_quant_state_from_module(self)
        quant_state = self.weight.quant_state

        if (
            not getattr(quant_state, "packing_format_for_cpu", False)
            and x.device.type == "cpu"
            and self.support_avx512bf16_for_cpu
            and not self.training
            and x.requires_grad == False
        ):
            self.weight.data, quant_state = _convert_weight_packed_for_cpu(self.weight.data, quant_state)

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        weight = self.weight if getattr(quant_state, "packing_format_for_cpu", False) else self.weight.t()

        return bnb.matmul_4bit(x, weight, bias=bias, quant_state=quant_state).to(inp_dtype)


class LinearFP4(Linear4bit):
    """
    Implements the FP4 data type.
    """

    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype=None,
        compress_statistics=True,
        quant_storage=torch.uint8,
        device=None,
    ):
        """
        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(
            input_features,
            output_features,
            bias,
            compute_dtype,
            compress_statistics,
            "fp4",
            quant_storage,
            device,
        )


class LinearNF4(Linear4bit):
    """Implements the NF4 data type.

    Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
    is normalized into the range [-1, 1].

    For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

    Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
    the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
    """

    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype=None,
        compress_statistics=True,
        quant_storage=torch.uint8,
        device=None,
    ):
        """
        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(
            input_features,
            output_features,
            bias,
            compute_dtype,
            compress_statistics,
            "nf4",
            quant_storage,
            device,
        )


class LinearNVFP4(nn.Linear):
    """NVFP4 (E2M1) quantized linear layer for Blackwell GPUs (SM_120).

    Quantizes weights to NVFP4 on first forward pass. Uses the hardware
    block-scaled MMA instruction for inference. Supports optional Hadamard
    rotation for improved accuracy.

    Args:
        input_features: Number of input features.
        output_features: Number of output features.
        bias: Whether to use bias. Defaults to True.
        device: Device for initialization.
    """

    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        device=None,
    ):
        super().__init__(input_features, output_features, bias, device)
        self.weight_quantized = False
        self.weight_packed = None
        self.weight_state = None

    def _quantize_weight(self):
        """Quantize the weight tensor to NVFP4."""
        from bitsandbytes.functional import quantize_nvfp4

        # Weight is (out_features, in_features) = (N, K) in GEMM terms
        w = self.weight.data.to(torch.bfloat16).contiguous()
        packed, state = quantize_nvfp4(w)
        self.weight_packed = packed
        self.weight_state = state
        self.weight_quantized = True
        # Free the original weight to save memory
        self.weight = nn.Parameter(torch.empty(0, device=w.device, dtype=w.dtype), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.weight_quantized:
            self._quantize_weight()

        from bitsandbytes.functional import gemm_nvfp4, quantize_nvfp4

        inp_dtype = x.dtype
        input_shape = x.shape

        # Reshape input: (*, K) -> (M, K). Use BF16 for CUTLASS fused quantize.
        x_2d = x.reshape(-1, input_shape[-1]).to(torch.bfloat16).contiguous()
        N = self.weight_state.shape[0]  # out_features

        # Quantize activations to NVFP4
        x_packed, x_state = quantize_nvfp4(x_2d)

        # Run NVFP4 GEMM: x @ weight^T
        out = gemm_nvfp4(x_packed, x_state, self.weight_packed, self.weight_state)

        # Reshape output back: (M, N) -> (*, N)
        out = out.reshape(*input_shape[:-1], N)

        # Add bias
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(inp_dtype)


class LinearNVFP4MoE(nn.Module):
    """NVFP4 (E2M1) quantized MoE linear layer for Blackwell GPUs (SM_120).

    Wraps multiple expert weight matrices and fuses their GEMMs into a single
    grouped kernel launch. Each expert has shape (output_features, input_features).

    Usage:
        layer = LinearNVFP4MoE(num_experts=128, input_features=2048, output_features=1536)
        # Load weights: layer.experts[i].weight = ...
        # Or from an existing list of nn.Linear:
        # layer = LinearNVFP4MoE.from_linear_experts(expert_linears)
        out = layer(x, expert_offsets)

    Args:
        num_experts: Number of experts.
        input_features: Input dimension (K) per expert.
        output_features: Output dimension (N) per expert.
        bias: Whether experts have bias. Defaults to False.
        device: Device for initialization.
    """

    def __init__(
        self,
        num_experts: int,
        input_features: int,
        output_features: int,
        bias: bool = False,
        device=None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.input_features = input_features
        self.output_features = output_features
        self.has_bias = bias
        self._quantized = False

        # Store raw weights until first forward (or explicit quantize call)
        self.weight = nn.Parameter(
            torch.empty(num_experts, output_features, input_features, device=device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(num_experts, output_features, device=device, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.bias = None

        # Quantized state (populated by _quantize_weights)
        self.register_buffer("weight_packed", None)
        self.register_buffer("weight_scales", None)
        self.register_buffer("weight_scales_batched", None)
        self.weight_tensor_scale: float = 1.0

    @classmethod
    def from_linear_experts(cls, experts: list[nn.Linear], device=None) -> "LinearNVFP4MoE":
        """Create from a list of nn.Linear expert modules."""
        num_experts = len(experts)
        out_features, in_features = experts[0].weight.shape
        has_bias = experts[0].bias is not None
        dev = device or experts[0].weight.device

        layer = cls(num_experts, in_features, out_features, bias=has_bias, device=dev)
        with torch.no_grad():
            for i, expert in enumerate(experts):
                layer.weight.data[i] = expert.weight.data.to(torch.bfloat16)
                if has_bias and expert.bias is not None:
                    layer.bias.data[i] = expert.bias.data.to(torch.bfloat16)
        return layer

    def _quantize_weights(self):
        """Quantize all expert weights to NVFP4 and stack into contiguous buffers."""
        from bitsandbytes.functional import quantize_nvfp4

        N, K = self.output_features, self.input_features

        # Quantize all experts and find a shared tensor scale
        all_packed = []
        all_scales = []
        all_scales_blocked = []
        tensor_scales = []

        for i in range(self.num_experts):
            w = self.weight.data[i].to(torch.bfloat16).contiguous()
            packed, state = quantize_nvfp4(w)
            all_packed.append(state.packed_data)
            all_scales.append(state.block_scales)
            all_scales_blocked.append(state.block_scales_blocked)
            tensor_scales.append(state.tensor_scale)

        # Stack into contiguous buffers: [num_experts * N, K/2] packed data
        self.weight_packed = torch.cat(all_packed, dim=0).contiguous()

        # Globally-swizzled scales for grouped GEMM (SM_120)
        weight_scales_flat = torch.cat(all_scales, dim=0).contiguous()
        self.weight_scales = torch.ops.bitsandbytes.scale_to_blocked(
            weight_scales_flat, self.num_experts * N, K // 16,
        )

        # Per-expert swizzled scales for batched GEMM (SM_100)
        self.weight_scales_batched = torch.cat(all_scales_blocked, dim=0).contiguous()

        self.weight_tensor_scale = max(tensor_scales)

        self._quantized = True
        # Free original weights
        self.weight = nn.Parameter(
            torch.empty(0, device=self.weight_packed.device, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        token_ids: Optional[torch.Tensor] = None,
        gating_weights: Optional[torch.Tensor] = None,
        num_dest_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Run NVFP4 GEMM across all experts.

        Uses batched GEMM on SM_100 (datacenter Blackwell) or grouped GEMM
        on SM_120 (consumer Blackwell).

        Args:
            x: Concatenated activations from all experts [total_tokens, K] in token order.
                Tokens for expert 0 come first, then expert 1, etc.
            expert_offsets: Cumulative token offsets [num_experts + 1], int32.
                expert_offsets[i] is the starting token index for expert i.
                expert_offsets[-1] = total_tokens.
            token_ids: Optional mapping from assignment index to output token index
                [total_tokens] (int32). Required for weighted gather.
            gating_weights: Optional per-assignment gating weights [total_tokens] (float32).
                Required for weighted gather.
            num_dest_tokens: Number of unique destination tokens in the output.
                Required when token_ids and gating_weights are provided.

        Returns:
            If token_ids and gating_weights are provided:
                Weighted output tensor [num_dest_tokens, N] with fused gather + weight + sum.
            Otherwise:
                Output tensor [total_tokens, N] with per-assignment expert results.
        """
        if not self._quantized:
            self._quantize_weights()

        major, _ = torch.cuda.get_device_capability(x.device)
        from bitsandbytes.cextension import lib
        if major == 10 and hasattr(lib, "cgemm_nvfp4_moe_sm100_init"):
            return self._forward_batched(
                x, expert_offsets,
                token_ids=token_ids, gating_weights=gating_weights,
                num_dest_tokens=num_dest_tokens,
            )
        return self._forward_grouped(x, expert_offsets)

    def _forward_grouped(self, x: torch.Tensor, expert_offsets: torch.Tensor) -> torch.Tensor:
        """Grouped GEMM path (SM_120 consumer Blackwell)."""
        from bitsandbytes.functional import gemm_nvfp4_grouped, quantize_nvfp4

        inp_dtype = x.dtype
        N, K = self.output_features, self.input_features

        x_2d = x.reshape(-1, K).to(torch.bfloat16).contiguous()
        x_packed, x_state = quantize_nvfp4(x_2d)

        out = gemm_nvfp4_grouped(
            x_packed,
            x_state,
            self.weight_packed,
            self.weight_scales,
            self.weight_tensor_scale,
            expert_offsets.to(torch.int32),
            N,
            K,
        )

        if self.bias is not None:
            expert_offsets_i32 = expert_offsets.to(torch.int32)
            tokens_per_expert = expert_offsets_i32[1:] - expert_offsets_i32[:-1]
            bias_expanded = torch.repeat_interleave(self.bias, tokens_per_expert, dim=0)
            out = out + bias_expanded.to(out.dtype)

        return out.to(inp_dtype)

    def _forward_batched(
        self,
        x: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        token_ids: Optional[torch.Tensor] = None,
        gating_weights: Optional[torch.Tensor] = None,
        num_dest_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Batched GEMM path (SM_100 datacenter Blackwell).

        Pipeline with init/run split for CUDA graph compatibility:
          1. abs().max()              — compute tensor scale (device-side)
          2. quantize_nvfp4_raw       — quantize all tokens in one launch
          3. cmoe_scatter_nvfp4       — FP4 data → persistent padded buffer
          4. scale_to_blocked_batched — scales → persistent swizzled buffer
          5. batched GEMM run()       — init-if-needed, then just run(stream)
          6. gather                   — weighted or unweighted depending on args

        All persistent buffers (A, SFA, D, alpha, gather workspace) are cached
        in the module so their addresses are stable for the CUTLASS init/run split.
        No .item() GPU-CPU sync on the common (decode) path.
        """
        import ctypes as ct

        from bitsandbytes.backends.cuda.ops import _gemm_nvfp4_batched_moe_sm100_raw
        from bitsandbytes.cextension import lib
        from bitsandbytes.functional import (
            _get_tensor_stream,
            get_ptr,
            quantize_nvfp4_raw,
        )

        inp_dtype = x.dtype
        N, K = self.output_features, self.input_features
        num_experts = self.num_experts
        total_tokens = x.shape[0]  # CPU int, no GPU sync
        use_weighted = token_ids is not None and gating_weights is not None
        dev = x.device

        expert_offsets_i32 = expert_offsets.to(torch.int32)
        tokens_per_expert = expert_offsets_i32[1:] - expert_offsets_i32[:-1]

        # Determine max_M without GPU sync on common path.
        # If cache exists and allocated_max_M >= total_tokens (upper bound on
        # any single expert's count), the buffers are guaranteed sufficient.
        if (hasattr(self, "_batched_cache")
                and total_tokens <= self._batched_cache.get("allocated_max_M", 0)):
            max_M = self._batched_cache["allocated_max_M"]
        else:
            # First call or total_tokens exceeds allocation: sync once
            raw_max_M = tokens_per_expert.max().item()
            max_M = ((raw_max_M + 127) // 128) * 128

        x_2d = x.reshape(-1, K).to(torch.bfloat16).contiguous()

        # 1. Compute tensor scale on GPU (no .item(), stays as device tensor)
        act_tensor_scale_dev = x_2d.abs().max()
        global_scale_dev = (1.0 / act_tensor_scale_dev).to(torch.float32)

        # 2. Quantize ALL concatenated tokens in one launch
        packed_all, scales_all = quantize_nvfp4_raw(x_2d, global_scale_dev)

        # 3. Ensure persistent cached buffers exist (stable pointers for init/run)
        cache_key = (max_M, N, K, num_experts)
        if not hasattr(self, "_batched_cache") or self._batched_cache.get("key") != cache_key:
            W = K // 16
            n_col_blocks = (W + 3) // 4
            n_row_blocks = (max_M + 127) // 128
            sfa_per_expert = n_row_blocks * n_col_blocks * 512
            sfa_total = num_experts * sfa_per_expert

            self._batched_cache = {
                "key": cache_key,
                "allocated_max_M": max_M,
                "A_batched": torch.empty(num_experts * max_M * (K // 2), dtype=torch.uint8, device=dev),
                "SFA_batched": torch.zeros(sfa_total, dtype=torch.uint8, device=dev),
                "D_out": torch.empty(num_experts * max_M, N, dtype=torch.bfloat16, device=dev),
                "alpha_dev": torch.empty(1, dtype=torch.float32, device=dev),
                # Pre-computed constants for scale swizzle
                "sfa_per_expert": sfa_per_expert,
                "n_row_blocks": n_row_blocks,
                "W": W,
                "expert_out_offsets": torch.arange(
                    num_experts, dtype=torch.int32, device=dev,
                ) * sfa_per_expert,
            }
        cache = self._batched_cache

        # Ensure weighted gather buffers exist if needed
        if use_weighted and num_dest_tokens is not None:
            if cache.get("gather_num_dest") != num_dest_tokens:
                cache["gather_workspace"] = torch.empty(
                    num_dest_tokens * N, dtype=torch.float32, device=dev,
                )
                cache["gather_output"] = torch.empty(
                    num_dest_tokens, N, dtype=torch.bfloat16, device=dev,
                )
                cache["gather_num_dest"] = num_dest_tokens

        stream = _get_tensor_stream(x_2d)

        # 4. Scatter FP4 data into persistent buffer
        lib.cmoe_scatter_nvfp4(
            get_ptr(packed_all),
            get_ptr(cache["A_batched"]),
            get_ptr(expert_offsets_i32),
            ct.c_int(max_M),
            ct.c_int(K),
            ct.c_int(num_experts),
            stream,
        )

        # 5. Swizzle scales per-expert into persistent buffer
        cache["SFA_batched"].zero_()
        lib.cscale_to_blocked_batched(
            get_ptr(scales_all),
            get_ptr(cache["SFA_batched"]),
            get_ptr(expert_offsets_i32[:-1]),
            get_ptr(tokens_per_expert),
            get_ptr(cache["expert_out_offsets"]),
            ct.c_int(cache["W"]),
            ct.c_int(num_experts),
            ct.c_int(cache["n_row_blocks"]),
            stream,
        )

        # 6. Set alpha (device-side, no .item() sync)
        cache["alpha_dev"].copy_(
            (act_tensor_scale_dev * self.weight_tensor_scale).to(torch.float32).reshape(1)
        )

        # 7. Batched GEMM (init-if-needed, then just run(stream))
        _gemm_nvfp4_batched_moe_sm100_raw(
            cache["A_batched"],
            self.weight_packed,
            cache["SFA_batched"],
            self.weight_scales_batched,
            cache["D_out"],
            cache["alpha_dev"],
            max_M, N, K, num_experts,
        )

        # 8. Add bias to GEMM output (before gather, included in weighted sum)
        if self.bias is not None:
            D_out_3d = cache["D_out"].view(num_experts, max_M, N)
            D_out_3d += self.bias.unsqueeze(1).to(D_out_3d.dtype)

        # 9. Gather: padded per-expert → output
        if use_weighted and num_dest_tokens is not None:
            # Derive expert_ids and slot_ids from expert_offsets (all on GPU)
            expert_ids = torch.repeat_interleave(
                torch.arange(num_experts, device=dev, dtype=torch.int32),
                tokens_per_expert,
            )
            starts_expanded = torch.repeat_interleave(
                expert_offsets_i32[:-1], tokens_per_expert,
            )
            slot_ids = (
                torch.arange(total_tokens, device=dev, dtype=torch.int32)
                - starts_expanded
            )

            # Fused weighted gather: gather + weight + FP32 accumulate + BF16 convert
            lib.cmoe_weighted_gather_bf16(
                get_ptr(cache["D_out"]),
                get_ptr(cache["gather_output"]),
                get_ptr(cache["gather_workspace"]),
                get_ptr(token_ids.to(torch.int32)),
                get_ptr(expert_ids),
                get_ptr(slot_ids),
                get_ptr(gating_weights.to(torch.float32)),
                ct.c_int(total_tokens),
                ct.c_int(num_dest_tokens),
                ct.c_int(max_M),
                ct.c_int(N),
                stream,
            )
            out = cache["gather_output"]
        else:
            # Unweighted gather (backwards compatible path)
            from bitsandbytes.functional import moe_gather_bf16
            out = moe_gather_bf16(
                cache["D_out"].view(-1), expert_offsets_i32,
                max_M, N, num_experts, total_tokens,
            )
            out = out.view(total_tokens, N)

        return out.to(inp_dtype)


class Int8Params(torch.nn.Parameter):
    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad=True,
        has_fp16_weights=False,
        CB: Optional[torch.Tensor] = None,
        SCB: Optional[torch.Tensor] = None,
    ):
        if data is None:
            data = torch.empty(0)
        obj = torch.Tensor._make_subclass(cls, data, requires_grad)
        obj.CB = CB
        obj.SCB = SCB
        obj.has_fp16_weights = has_fp16_weights
        return obj

    def _quantize(self, device):
        if self.has_fp16_weights:
            return super().to(device)

        # We quantize the weight and store in 8bit row-major
        B = self.data.contiguous().to(device=device, dtype=torch.float16)
        CB, SCB, _ = bnb.functional.int8_vectorwise_quant(B)
        self.data = CB
        self.CB = CB
        self.SCB = SCB

        return self

    def cpu(self):
        return self.to(device="cpu")

    def cuda(self, device: Optional[int | device | str] = None, non_blocking: bool = False):
        return self.to(device="cuda" if device is None else device, non_blocking=non_blocking)

    def xpu(self, device: Optional[int | device | str] = None, non_blocking: bool = False):
        return self.to(device="xpu" if device is None else device, non_blocking=non_blocking)

    def __deepcopy__(self, memo):
        # adjust this if new arguments are added to the constructor
        new_instance = type(self).__new__(
            type(self),
            data=copy.deepcopy(self.data, memo),
            requires_grad=self.requires_grad,
            has_fp16_weights=self.has_fp16_weights,
            CB=copy.deepcopy(self.CB, memo),
            SCB=copy.deepcopy(self.SCB, memo),
        )
        return new_instance

    @overload
    def to(
        self: T,
        device: Optional[int | device] = ...,
        dtype: Optional[dtype | str] = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: dtype | str, non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T: ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)

        is_quantized = self.data.dtype == torch.int8

        if not is_quantized and device is not None and device.type != "meta" and self.data.device.type == "cpu":
            # We're moving from a CPU device to a non-meta device.
            # In this circumstance, we want to quantize if we haven't already.
            return self._quantize(device)

        # Create a new parameter on the target device.
        new_param = Int8Params(
            super().to(device=device, dtype=dtype, non_blocking=non_blocking),
            requires_grad=self.requires_grad,
            has_fp16_weights=self.has_fp16_weights,
        )

        # If we had already quantized, move the statistics appropriately.
        if is_quantized:
            new_param.CB = new_param.data

            if device is not None and self.SCB is not None and self.SCB.device.type != "meta":
                new_param.SCB = self.SCB.to(device)

        return new_param


def maybe_rearrange_weight(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    weight = state_dict.get(f"{prefix}weight")
    if weight is None:
        # if the state dict has no weights for this layer (e.g., LoRA finetuning), do nothing
        return
    weight_format = state_dict.pop(f"{prefix}weight_format", "row")

    if isinstance(weight_format, torch.Tensor):
        weight_format = weight_format.item()

    # For new weights format storage type, we explicitly check
    # if weights_format is on the mapping
    if isinstance(weight_format, int) and weight_format not in INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING:
        raise ValueError(f"Expected supported weight format - got {weight_format}")
    elif isinstance(weight_format, int) and weight_format in INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING:
        weight_format = INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING[weight_format]

    if weight_format != "row":
        raise ValueError(f"Only 'row' weight format is supported, got {weight_format}")


class Embedding8bit(nn.Embedding):
    """
    This class implements [LLM.int8()](https://arxiv.org/abs/2208.07339) algorithm for embedding layer

    Quantization API is similar to Linear8bitLt:
    ```python
    import torch
    import torch.nn as nn

    from bitsandbytes.nn import Embedding8bit

    fp16_module = nn.Embedding(128, 64)
    int8_module = Embedding8bit(128, 64)

    int8_module.load_state_dict(fp16_module.state_dict())

    int8_module = int8_module.to(0) # Quantization happens here
    ```
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.dtype = self.weight.data.dtype

        self.weight = Int8Params(self.weight.data, has_fp16_weights=False, requires_grad=False)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        raise NotImplementedError("Saving Embedding8bit module is not implemented")

    def forward(self, input: Tensor) -> Tensor:
        if not hasattr(self.weight, "SCB"):
            raise RuntimeError("Embedding layer is not quantized. Please call .cuda() or .to(device) first.")

        rows = self.weight.data
        row_stats = self.weight.SCB

        assert rows.shape == (self.num_embeddings, self.embedding_dim)
        assert row_stats.shape == (self.num_embeddings,)

        compressed_output = F.embedding(input, rows)
        compressed_output_stats = F.embedding(input, row_stats.view(self.num_embeddings, 1))

        output = compressed_output * (compressed_output_stats / 127.0)

        return output.to(self.dtype)


class Embedding4bit(nn.Embedding):
    """
    This is the base class similar to Linear4bit. It implements the 4-bit quantization algorithm presented in
    [QLoRA](https://arxiv.org/abs/2305.14314) for embeddings.

    Quantization API is similar to Linear4bit:
    ```python
    import torch
    import torch.nn as nn

    from bitsandbytes.nn import Embedding4bit

    fp16_module = nn.Embedding(128, 64)
    quantized_module = Embedding4bit(128, 64)

    quantized_module.load_state_dict(fp16_module.state_dict())

    quantized_module = quantized_module.to(0) # Quantization happens here
    ```
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        dtype=None,
        quant_type="fp4",
        quant_storage=torch.uint8,
        device=None,
    ):
        super().__init__(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.dtype = self.weight.data.dtype

        self.weight = Params4bit(
            self.weight.data,
            requires_grad=False,
            compress_statistics=None,
            quant_type=quant_type,
            quant_storage=quant_storage,
            module=self,
        )

        blocksize = self.weight.blocksize

        if embedding_dim % blocksize != 0:
            warnings.warn(
                f"Embedding size {embedding_dim} is not divisible by block size {blocksize}. "
                "This will lead to slow inference.",
            )

    def _forward_with_partial_dequantize(self, input: Tensor):
        assert self.embedding_dim % self.weight.quant_state.blocksize == 0

        w_4bit_uint8 = self.weight.data.view(torch.uint8).view(self.num_embeddings * self.embedding_dim // 2, 1)

        output_4bit = torch.nn.functional.embedding(
            weight=w_4bit_uint8.view(self.num_embeddings, self.embedding_dim // 2),
            input=input,
        ).view(-1, 1)
        assert output_4bit.shape == (input.numel() * self.embedding_dim // 2, 1)

        blocks_per_emb = self.embedding_dim // self.weight.blocksize

        absmax = self.weight.quant_state.absmax
        assert absmax.shape == (self.num_embeddings * blocks_per_emb,)

        output_absmax = torch.nn.functional.embedding(
            weight=absmax.view(self.num_embeddings, blocks_per_emb),
            input=input,
        ).view(
            -1,
        )
        assert output_absmax.shape == (input.numel() * blocks_per_emb,)

        output_quant_state = copy.deepcopy(self.weight.quant_state)
        output_quant_state.absmax = output_absmax
        output_quant_state.shape = torch.Size((*input.shape, self.embedding_dim))

        output = bnb.functional.dequantize_4bit(output_4bit, output_quant_state)
        assert output.shape == (*input.shape, self.embedding_dim)

        return output.to(self.dtype)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        raise NotImplementedError("Saving Embedding4bit module is not implemented")

    def forward(self, input: Tensor) -> Tensor:
        fix_4bit_weight_quant_state_from_module(self)

        if self.embedding_dim % self.weight.quant_state.blocksize == 0:
            return self._forward_with_partial_dequantize(input)

        dequantized_weight = bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state)

        return torch.nn.functional.embedding(
            weight=dequantized_weight,
            input=input,
        ).to(self.dtype)


class EmbeddingFP4(Embedding4bit):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        dtype=None,
        quant_storage=torch.uint8,
        device=None,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            quant_type="fp4",
            quant_storage=quant_storage,
            device=device,
        )


class EmbeddingNF4(Embedding4bit):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        dtype=None,
        quant_storage=torch.uint8,
        device=None,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            quant_type="nf4",
            quant_storage=quant_storage,
            device=device,
        )


class Linear8bitLt(nn.Linear):
    """
    This class is the base module for the [LLM.int8()](https://arxiv.org/abs/2208.07339) algorithm.
    To read more about it, have a look at the paper.

    In order to quantize a linear layer one should first load the original fp16 / bf16 weights into
    the Linear8bitLt module, then call `int8_module.to("cuda")` to quantize the fp16 weights.

    Example:

    ```python
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt

    fp16_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64)
    )

    int8_model = nn.Sequential(
        Linear8bitLt(64, 64, has_fp16_weights=False),
        Linear8bitLt(64, 64, has_fp16_weights=False)
    )

    int8_model.load_state_dict(fp16_model.state_dict())
    int8_model = int8_model.to(0) # Quantization happens here
    ```
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias=True,
        has_fp16_weights=True,
        threshold=0.0,
        index=None,
        device=None,
    ):
        """
        Initialize Linear8bitLt class.

        Args:
            input_features (`int`):
                Number of input features of the linear layer.
            output_features (`int`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
            has_fp16_weights (`bool`, defaults to `True`):
                If False, weights are quantized to int8 on ``.to(device)``. If True,
                weights remain in fp16 and are quantized on-the-fly during each forward pass.
            threshold (`float`, defaults to `0.0`):
                Outlier threshold for mixed-precision decomposition (LLM.int8()). During the
                forward pass, activation columns where any value exceeds this threshold are
                computed in fp16, while the remaining columns use int8. This operates on
                **activations** (inputs), not on weight values. Set to 0.0 to disable
                mixed-precision decomposition and quantize all columns to int8.
            index: Indices for weight reordering (used internally).
            device: Device to initialize the layer on.
        """
        super().__init__(input_features, output_features, bias, device)
        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights

        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)
        self._register_load_state_dict_pre_hook(maybe_rearrange_weight)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)

        # we only need to save SCB as extra data, because CB for quantized weights is already stored in weight.data
        scb_name = "SCB"

        # case 1: .cuda was called, SCB is in self.weight
        param_from_weight = getattr(self.weight, scb_name)
        # case 2: self.init_8bit_state was called, SCB is in self.state
        param_from_state = getattr(self.state, scb_name)

        key_name = prefix + f"{scb_name}"

        # We now only save in row-major. This format information is stored for backwards compatibility.
        format_name = prefix + "weight_format"

        if not self.state.has_fp16_weights:
            if param_from_weight is not None:
                destination[key_name] = param_from_weight if keep_vars else param_from_weight.detach()
                destination[format_name] = torch.tensor(0, dtype=torch.uint8)
            elif param_from_state is not None:
                destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
                destination[format_name] = torch.tensor(0, dtype=torch.uint8)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        unexpected_copy = list(unexpected_keys)

        for key in unexpected_copy:
            input_name = key[len(prefix) :]
            if input_name == "SCB":
                if self.weight.SCB is None:
                    # buffers not yet initialized, can't access them directly without quantizing first
                    raise RuntimeError(
                        "Loading a quantized checkpoint into non-quantized Linear8bitLt is "
                        "not supported. Please call module.cuda() before module.load_state_dict()",
                    )

                input_param = state_dict[key]
                self.weight.SCB.copy_(input_param)

                if self.state.SCB is not None:
                    self.state.SCB = self.weight.SCB

                unexpected_keys.remove(key)

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def to(self, *args, **kwargs):
        # Call the parent to() method to handle standard parameter/buffer movement
        result = super().to(*args, **kwargs)

        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)

        # Handle state tensors if needed.
        if device is not None:
            if result.state.CB is not None:
                result.state.CB = result.state.CB.to(device)
            if result.state.SCB is not None:
                result.state.SCB = result.state.SCB.to(device)

        return result

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights and self.state.CB is not None:
            self.weight.data = self.state.CB

        return out


class OutlierAwareLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, device=None):
        super().__init__(input_features, output_features, bias, device)
        self.outlier_dim = None
        self.is_quantized = False

    def forward_with_outliers(self, x, outlier_idx):
        raise NotImplementedError("Please override the `forward_with_outliers(self, x, outlier_idx)` function")

    def quantize_weight(self, w, outlier_idx):
        raise NotImplementedError("Please override the `quantize_weights(self, w, outlier_idx)` function")

    def forward(self, x):
        if self.outlier_dim is None:
            tracer = OutlierTracer.get_instance()
            if not tracer.is_initialized():
                print("Please use OutlierTracer.initialize(model) before using the OutlierAwareLinear layer")
            outlier_idx = tracer.get_outliers(self.weight)
            # print(outlier_idx, tracer.get_hvalue(self.weight))
            self.outlier_dim = outlier_idx

        if not self.is_quantized:
            w = self.quantize_weight(self.weight, self.outlier_dim)
            self.weight.data.copy_(w)
            self.is_quantized = True


class SwitchBackLinearBnb(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
        device=None,
    ):
        super().__init__(input_features, output_features, bias, device)
        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x):
        self.state.is_training = self.training

        if self.weight.CB is not None:
            self.init_8bit_state()

        return bnb.matmul_mixed(x.half(), self.weight.half(), bias=None, state=self.state) + self.bias


# ============================================================================
# K-bit Training Classes (from QLORA-2 branch)
# ============================================================================

# K-bit quantization (generalized 2-5 bit, blocksize 32, E4M4 absmax)
# ---------------------------------------------------------------------------

KBIT_BLOCKSIZE = 32
KBIT_TILE_N = 128


def _pad_to_multiple(n: int, m: int) -> int:
    """Round *n* up to the next multiple of *m*."""
    return ((n + m - 1) // m) * m


class _GlobalWeightBuffer:
    """Per-device pre-allocated buffer for dequantized weights.

    Avoids repeated allocation/deallocation on every forward and backward call
    through kbit linear layers.  The buffer is lazily created and grows as
    needed but never shrinks.

    Thread-safety: PyTorch guarantees only one forward/backward is active per
    device at a time, so a single buffer per device suffices.
    """

    _buffers: dict[torch.device, torch.Tensor] = {}

    @classmethod
    def get_buffer(cls, device: torch.device, min_elements: int, dtype: torch.dtype) -> torch.Tensor:
        """Return a buffer with at least *min_elements* on *device*."""
        key = device
        buf = cls._buffers.get(key)
        if buf is None or buf.numel() < min_elements or buf.dtype != dtype:
            cls._buffers[key] = torch.empty(min_elements, dtype=dtype, device=device)
        return cls._buffers[key][:min_elements]

    @classmethod
    def clear(cls):
        cls._buffers.clear()


class ParamsKbit(torch.nn.Parameter):
    """Parameter subclass for k-bit blockwise quantized weights.

    Stores weights in bit-plane packed int32 format with E4M4 absmax scaling.
    Quantization and (optional) repacking happen lazily on the first
    ``.to(device)`` call, mirroring the ``Params4bit`` pattern.

    Attributes:
        k: Bit width (2-5).
        K_dim: Inner (reduction) dimension of the weight matrix.
        N: Output (row) dimension of the weight matrix.
        N_padded: N rounded up to the next multiple of 128.
        packed: int32 bit-plane packed data (flat layout).
        absmax: float32 per-block absmax values (flat layout).
        codebook: float32 codebook tensor (2^k entries).
        original_dtype: The dtype of the weight before quantization.
        kbit_quantized: Whether quantization has been applied.
    """

    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad: bool = False,
        k: int = 4,
        module: Optional["LinearKbit"] = None,
    ) -> "ParamsKbit":
        if data is None:
            data = torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.k = k
        self.module = module
        self.kbit_quantized = False
        # Populated during _quantize:
        self.packed = None
        self.absmax = None
        self.codebook = None
        self.K_dim = 0
        self.N = 0
        self.N_padded = 0
        self.original_dtype = data.dtype
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data"] = self.data
        state["requires_grad"] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.requires_grad = state["requires_grad"]
        self.k = state["k"]
        self.module = state["module"]
        self.kbit_quantized = state["kbit_quantized"]
        self.packed = state["packed"]
        self.absmax = state["absmax"]
        self.codebook = state["codebook"]
        self.K_dim = state["K_dim"]
        self.N = state["N"]
        self.N_padded = state["N_padded"]
        self.original_dtype = state["original_dtype"]
        self.data = state["data"]

    def __deepcopy__(self, memo):
        import copy as _copy

        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.packed = _copy.deepcopy(state["packed"])
        new_instance.absmax = _copy.deepcopy(state["absmax"])
        new_instance.codebook = _copy.deepcopy(state["codebook"])
        new_instance.data = _copy.deepcopy(state["data"])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    def _quantize(self, device):
        """Quantize fp16/bf16 weight to kbit format on *device*.

        The weight tensor ``self.data`` is expected to be shape ``(N, K_dim)``
        (standard ``nn.Linear`` weight layout: ``out_features × in_features``).
        """
        w = self.data.contiguous().to(device)
        N, K_dim = w.shape
        self.original_dtype = w.dtype
        self.N = N
        self.K_dim = K_dim
        self.N_padded = _pad_to_multiple(N, KBIT_TILE_N)

        # Pad N dimension to multiple of 128 for kernel alignment
        if self.N_padded != N:
            w = torch.nn.functional.pad(w, (0, 0, 0, self.N_padded - N))

        packed, absmax, codebook = bnb.functional.quantize_kbit(
            w.reshape(-1).float(),  # quantize_kbit expects flat input
            k=self.k,
            absmax_format="fp32",  # keep float32 for GEMV; E4M4 applied at repack
        )

        self.packed = packed
        self.absmax = absmax
        self.codebook = codebook
        self.kbit_quantized = True

        # Store a small sentinel in self.data so the Parameter has the right device
        self.data = torch.empty(0, device=device, dtype=self.original_dtype)

        if self.module is not None:
            self.module._sync_kbit_state(self)

        return self

    def cpu(self):
        return self.to(device="cpu")

    def cuda(self, device: Optional[int | device | str] = None, non_blocking: bool = False):
        return self.to(device="cuda" if device is None else device, non_blocking=non_blocking)

    @overload
    def to(
        self: T,
        device: Optional[int | device] = ...,
        dtype: Optional[dtype | str] = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: dtype | str, non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T: ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None and device.type != "meta" and not self.kbit_quantized:
            return self._quantize(device)
        else:
            # Already quantized — move packed data to new device
            new_param = ParamsKbit(
                super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
                k=self.k,
                module=self.module,
            )
            new_param.kbit_quantized = self.kbit_quantized
            new_param.packed = self.packed.to(device) if self.packed is not None else None
            new_param.absmax = self.absmax.to(device) if self.absmax is not None else None
            new_param.codebook = self.codebook.to(device) if self.codebook is not None else None
            new_param.K_dim = self.K_dim
            new_param.N = self.N
            new_param.N_padded = self.N_padded
            new_param.original_dtype = self.original_dtype
            return new_param


class LinearKbit(nn.Linear):
    """Linear layer using k-bit blockwise quantization.

    Supports generalized k-bit widths (k=2,3,4,5) with blocksize 32 and
    E4M4 absmax encoding.  Inference uses automatic kernel dispatch:

    - M <= 4 (decode): ``kbit_scalar_gemv`` (flat-layout, float32 absmax)
    - M > 4 (prefill/training): ``dequantize_kbit`` + ``torch.mm``

    Example::

        layer = LinearKbit(4096, 4096, k=4)
        layer.load_state_dict(fp16_layer.state_dict())
        layer = layer.to("cuda")  # quantization happens here
        out = layer(x)            # dispatches to optimal kernel
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias: bool = True,
        k: int = 4,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
    ):
        super().__init__(input_features, output_features, bias, device)
        self.weight = ParamsKbit(self.weight.data, requires_grad=False, k=k, module=self)
        self.k = k
        self.compute_dtype = compute_dtype

    def _sync_kbit_state(self, params: ParamsKbit):
        """Called by ParamsKbit after quantization to sync module metadata."""
        pass  # reserved for future use (e.g., registering buffer sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from bitsandbytes.autograd._functions import MatMulKbit

        w = self.weight
        if not w.kbit_quantized:
            raise RuntimeError("LinearKbit weight not quantized. Call .to(device) first.")

        inp_dtype = x.dtype
        compute_dtype = self.compute_dtype or x.dtype
        if compute_dtype not in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float16
        x = x.to(compute_dtype)

        # Flatten batch dimensions: (*, K_dim) -> (M, K_dim)
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        M = x_2d.shape[0]

        if M <= 4 and not self.training and not x.requires_grad:
            # Decode path: scalar GEMV (flat layout, float32 absmax)
            out = torch.ops.bitsandbytes.kbit_scalar_gemv(
                x_2d,
                w.packed,
                w.absmax,
                w.codebook,
                w.K_dim,
                w.N_padded,
                w.k,
            )
        elif x.requires_grad:
            # Training path: use autograd-aware MatMulKbit
            out = MatMulKbit.apply(
                x_2d,
                w.packed,
                w.absmax,
                w.codebook,
                w.k,
                w.K_dim,
                w.N_padded,
                w.N,
                compute_dtype,
            )
        else:
            # Prefill path (no grad): dequantize + cuBLAS matmul
            n_elements = w.N_padded * w.K_dim
            w_deq = bnb.functional.dequantize_kbit(
                w.packed,
                w.absmax,
                w.codebook,
                w.k,
                n_elements,
                compute_dtype,
            )
            w_mat = w_deq[:n_elements].reshape(w.N_padded, w.K_dim)
            out = torch.nn.functional.linear(x_2d, w_mat[: w.N, :])

        # Slice off N-padding (MatMulKbit handles this internally)
        if w.N_padded != w.N and not x.requires_grad:
            out = out[:, : w.N]

        # Add bias
        if self.bias is not None:
            out = out + self.bias.to(compute_dtype)

        # Restore batch dimensions
        out = out.reshape(*orig_shape[:-1], w.N)
        return out.to(inp_dtype)


def prepare_model_for_kbit_training(
    model: torch.nn.Module,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: Optional[dict] = None,
) -> torch.nn.Module:
    """Prepare a model with LinearKbit layers for QLoRA-style training.

    This function:
    1. Freezes all base model parameters (requires_grad=False)
    2. Casts LayerNorm and other normalization layers to float32
    3. Enables gradient checkpointing if requested
    4. Registers the global weight buffer size from the model's largest layer

    After calling this, add LoRA adapters (or any trainable parameters) and
    those will be the only parameters that receive gradients.

    Args:
        model: A model containing LinearKbit layers.
        use_gradient_checkpointing: Enable gradient checkpointing for memory savings.
        gradient_checkpointing_kwargs: Kwargs passed to model.gradient_checkpointing_enable().

    Returns:
        The modified model (in-place).
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Cast normalization layers to float32 for training stability
    for module in model.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            module.float()

    # Enable gradient checkpointing
    if use_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            kwargs = gradient_checkpointing_kwargs or {}
            model.gradient_checkpointing_enable(**kwargs)
        elif hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.is_gradient_checkpointing = True

    # Register global weight buffer for the largest LinearKbit layer
    max_elements = 0
    compute_dtype = torch.float16
    device = None
    for module in model.modules():
        if isinstance(module, LinearKbit) and module.weight.kbit_quantized:
            w = module.weight
            n = w.N_padded * w.K_dim
            if n > max_elements:
                max_elements = n
                device = w.packed.device
            if module.compute_dtype is not None:
                compute_dtype = module.compute_dtype

    if max_elements > 0 and device is not None:
        _GlobalWeightBuffer.get_buffer(device, max_elements, compute_dtype)

    return model


