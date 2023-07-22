# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, TypeVar, Union, overload

import warnings
import torch
import torch.nn.functional as F
from torch import Tensor, device, dtype, nn

import bitsandbytes as bnb
import bitsandbytes.functional
from bitsandbytes.autograd._functions import undo_layout, get_tile_inds
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer, find_outlier_dims

T = TypeVar("T", bound="torch.nn.Module")


class StableEmbedding(torch.nn.Embedding):
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
        GlobalOptimManager.get_instance().register_module_override(
            self, "weight", {"optim_bits": 32}
        )

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
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device=device
        )
        GlobalOptimManager.get_instance().register_module_override(
            self, "weight", {"optim_bits": 32}
        )

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
    def __new__(cls, data=None, requires_grad=True, quant_state=None, blocksize=64, compress_statistics=True, quant_type='fp4'):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.data = data
        return self

    def cuda(self, device):
        w = self.data.contiguous().half().cuda(device)
        w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=self.blocksize, compress_statistics=self.compress_statistics, quant_type=self.quant_type)
        self.data = w_4bit
        self.quant_state = quant_state

        return self

    @overload
    def to(self: T, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ..., non_blocking: bool = ...,) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if (device is not None and device.type == "cuda" and self.data.device.type == "cpu"):
            return self.cuda(device)
        else:
            s = self.quant_state
            if s is not None:
                # make sure the quantization state is on the right device
                s[0] = s[0].to(device)
                if self.compress_statistics:
                    # TODO: refactor this. This is a nightmare
                    # for 4-bit: 
                    # state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
                    # state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
                    #s[-2][0] = s[-2][0].to(device) # offset
                    #s[-2][1][0] = s[-2][1][0].to(device) # nested absmax

                    # for 8-bit
                    s[-3][0] = s[-3][0].to(device) # offset
                    s[-3][1][0] = s[-3][1][0].to(device) # nested quantiation state statitics
                    s[-3][1][1] = s[-3][1][1].to(device) # nested quantiation codebook
            new_param = Params4bit(super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                                  requires_grad=self.requires_grad, quant_state=self.quant_state,
                                   blocksize=self.blocksize, compress_statistics=self.compress_statistics,
                                   quant_type=self.quant_type)

            return new_param

class Linear4bit(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_type='fp4',device=None):
        super().__init__(input_features, output_features, bias, device)
        self.weight = Params4bit(self.weight.data, requires_grad=False, compress_statistics=compress_statistics, quant_type=quant_type)
        self.compute_dtype = compute_dtype
        self.compute_type_is_set = False

    def set_compute_type(self, x):
        if x.dtype in [torch.float32, torch.bfloat16]:
            # the input is in a dtype that is safe to compute in, we switch
            # to this type for speed and stability
            self.compute_dtype = x.dtype
        elif x.dtype == torch.float16:
            # we take the compoute dtype passed into the layer
            if self.compute_dtype == torch.float32 and (x.numel() == x.shape[-1]):
                # single batch inference with input torch.float16 and compute_dtype float32 -> slow inference when it could be fast
                # warn the user about this
                warnings.warn(f'Input type into Linear4bit is torch.float16, but bnb_4bit_compute_type=torch.float32 (default). This will lead to slow inference.')
                warnings.filterwarnings('ignore', message='.*inference.')
            if self.compute_dtype == torch.float32 and (x.numel() != x.shape[-1]):
                warnings.warn(f'Input type into Linear4bit is torch.float16, but bnb_4bit_compute_type=torch.float32 (default). This will lead to slow inference or training speed.')
                warnings.filterwarnings('ignore', message='.*inference or training')






    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if getattr(self.weight, 'quant_state', None) is None:
            print('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)

        out = out.to(inp_dtype)

        return out

class LinearFP4(Linear4bit):
    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True,device=None):
        super().__init__(input_features, output_features, bias, compute_dtype, compress_statistics, 'fp4', device)

class LinearNF4(Linear4bit):
    ''' Implements the NF4 data type.

        Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
        is normalized into the range [-1, 1].

        For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

        Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
        the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
    '''
    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True,device=None):
        super().__init__(input_features, output_features, bias, compute_dtype, compress_statistics, 'nf4', device)



class Int8Params(torch.nn.Parameter):
    def __new__(
        cls,
        data=None,
        requires_grad=True,
        has_fp16_weights=False,
        CB=None,
        SCB=None,
    ):
        cls.has_fp16_weights = has_fp16_weights
        cls.CB = None
        cls.SCB = None
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def cuda(self, device):
        if self.has_fp16_weights:
            return super().cuda(device)
        else:
            # we store the 8-bit rows-major weight
            # we convert this weight to the turning/ampere weight during the first inference pass
            B = self.data.contiguous().half().cuda(device)
            CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
            del CBt
            del SCBt
            self.data = CB
            setattr(self, "CB", CB)
            setattr(self, "SCB", SCB)

        return self

    @overload
    def to(
        self: T,
        device: Optional[Union[int, device]] = ...,
        dtype: Optional[Union[dtype, str]] = ...,
        non_blocking: bool = ...,
    ) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        if (
            device is not None
            and device.type == "cuda"
            and self.data.device.type == "cpu"
        ):
            return self.cuda(device)
        else:
            new_param = Int8Params(
                super().to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                ),
                requires_grad=self.requires_grad,
                has_fp16_weights=self.has_fp16_weights,
            )
            new_param.CB = self.CB
            new_param.SCB = self.SCB

            return new_param


def maybe_rearrange_weight(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    weight = state_dict.get(f"{prefix}weight")
    if weight is None:
        # if the state dict has no weights for this layer (e.g., LoRA finetuning), do nothing
        return
    weight_format = state_dict.pop(f"{prefix}weight_format", "row")

    if weight_format != "row":
        tile_indices = get_tile_inds(weight_format, weight.device)
        state_dict[f"{prefix}weight"] = undo_layout(weight, tile_indices)


class Linear8bitLt(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, has_fp16_weights=True,
                       memory_efficient_backward=False, threshold=0.0, index=None, device=None):
        super().__init__(input_features, output_features, bias, device)
        assert not memory_efficient_backward, "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
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
        # case 3: SCB is in self.state, weight layout reordered after first forward()
        layout_reordered = self.state.CxB is not None

        key_name = prefix + f"{scb_name}"
        format_name = prefix + "weight_format"

        if not self.state.has_fp16_weights:
            if param_from_weight is not None:
                destination[key_name] = param_from_weight if keep_vars else param_from_weight.detach()
                destination[format_name] = "row"
            elif param_from_state is not None and not layout_reordered:
                destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
                destination[format_name] = "row"
            elif param_from_state is not None:
                destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
                destination[format_name] = self.state.formatB

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)
        unexpected_copy = list(unexpected_keys)

        for key in unexpected_copy:
            input_name = key[len(prefix):]
            if input_name == "SCB":
                if self.weight.SCB is None:
                    # buffers not yet initialized, can't access them directly without quantizing first
                    raise RuntimeError("Loading a quantized checkpoint into non-quantized Linear8bitLt is "
                                       "not supported. Please call module.cuda() before module.load_state_dict()")

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

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out


class OutlierAwareLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, device=None):
        super().__init__(input_features, output_features, bias, device)
        self.outlier_dim = None
        self.is_quantized = False

    def forward_with_outliers(self, x, outlier_idx):
        raise NotImplementedError('Please override the `forward_with_outliers(self, x, outlier_idx)` function')

    def quantize_weight(self, w, outlier_idx):
        raise NotImplementedError('Please override the `quantize_weights(self, w, outlier_idx)` function')

    def forward(self, x):
        if self.outlier_dim is None:
            tracer = OutlierTracer.get_instance()
            if not tracer.is_initialized():
                print('Please use OutlierTracer.initialize(model) before using the OutlierAwareLinear layer')
            outlier_idx = tracer.get_outliers(self.weight)
            #print(outlier_idx, tracer.get_hvalue(self.weight))
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
        device=None
    ):
        super().__init__(
            input_features, output_features, bias, device
        )
        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(
            self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights
        )

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x):
        self.state.is_training = self.training

        if self.weight.CB is not None:
            self.init_8bit_state()

        out = bnb.matmul_mixed(x.half(), self.weight.half(), bias=None, state=self.state) + self.bias
