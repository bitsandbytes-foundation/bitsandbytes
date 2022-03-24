r"""Functional interface"""
from typing import Callable, List, Optional, Tuple
import math
import warnings

import torch
import bitsandbytes as bnb
from torch import _VF
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes

from torch._jit_internal import boolean_dispatch, _overload
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from torch.nn import _reduction as _Reduction
from torch.nn import grad  # noqa: F401
from torch.nn.modules import utils
from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default
from torch.nn.functional import linear, hardtanh
import torch.distributed as dist

Tensor = torch.Tensor


def linear8bit(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, matmul_func=None) -> Tensor:
    if matmul_func:
        out = matmul_func(input, weight.t())
    else:
        out = bnb.matmul(input, weight.t())
    if bias is not None:
        out += bias.unsqueeze(0).expand_as(out)
    return out

def truncate_tensor(x, low, high, mode='zero'):
    with torch.no_grad():
        if mode == 'clamp':
            with torch.no_grad():
                quantx = bnb.functional.estimate_quantiles(x)
                x.clamp_(quantx[low], quantx[high])
        elif mode == 'zero':
            assert low == 0
            with torch.no_grad():
                absx = torch.abs(x)
                quantx = bnb.functional.estimate_quantiles(absx)
                x[absx <= quantx[high]] = 0.0
        else:
            raise ValueError(f'Unknown truncation mode: {mode}')
    return x

def apply_squash_func(x, attention_func):
    if attention_func is None or attention_func == '':
        pass
    elif attention_func == 'hardtanh':
        x = torch.nn.functional.hardtanh(x).clone()
    elif attention_func == 'elu':
        x = torch.nn.functional.elu(x).clone()
    elif attention_func == 'hardswish':
        x = torch.nn.functional.hardswish(x).clone()
    elif attention_func == 'logistic':
        x = torch.nn.functional.sigmoid(x).clone()
    else:
        raise ValueError(f'Activation function not supported for attention: {attention_func}')

    return x


# copied from PyTorch
def multi_head_attention_forward8bit(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    attention_type='off',
    norms=None,
    quant_type='vector',
    attention_func=None,
    attention_bmm_bits='16,16,8',
    args = None
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)


    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear8bit(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear8bit(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear8bit(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear8bit(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear8bit(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear8bit(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            if 'linear' in attention_type:
                if args.attn_trunc_low > 0 or args.attn_trunc_high < 100:
                    query = truncate_tensor(query, args.attn_trunc_low, args.attn_trunc_high, args.attn_trunc_mode)
                    key = truncate_tensor(key, args.attn_trunc_low, args.attn_trunc_high, args.attn_trunc_mode)
                    value = truncate_tensor(value, args.attn_trunc_low, args.attn_trunc_high, args.attn_trunc_mode)
                q = linear8bit(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                k = linear8bit(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
                v = linear8bit(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
            else:
                if 'q' in attention_type:
                    query = apply_squash_func(query, args.attention_func)
                    q = linear8bit(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim], num_splits=args.num_splits)
                else:
                    q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                if 'k' in attention_type:
                    k = linear8bit(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
                else:
                    k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
                if 'v' in attention_type:
                    v = linear8bit(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
                else:
                    v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            if 'linear' in attention_type:
                if args.attn_trunc_low > 0 or args.attn_trunc_high < 100:
                    query = truncate_tensor(query, args.attn_trunc_low, args.attn_trunc_high, args.attn_trunc_mode)
                    key = truncate_tensor(key, args.attn_trunc_low, args.attn_trunc_high, args.attn_trunc_mode)
                    value = truncate_tensor(value, args.attn_trunc_low, args.attn_trunc_high, args.attn_trunc_mode)
                q = linear8bit(query, q_proj_weight_non_opt, in_proj_bias)
                k = linear8bit(key, k_proj_weight_non_opt, in_proj_bias)
                v = linear8bit(value, v_proj_weight_non_opt, in_proj_bias)
            else:
                if 'q' in attention_type:
                    q = linear8bit(query, q_proj_weight_non_opt, in_proj_bias)
                else:
                    q = linear(query, q_proj_weight_non_opt, in_proj_bias)
                if 'k' in attention_type:
                    k = linear8bit(key, k_proj_weight_non_opt, in_proj_bias)
                else:
                    k = linear(key, k_proj_weight_non_opt, in_proj_bias)
                if 'v' in attention_type:
                    v = linear8bit(value, v_proj_weight_non_opt, in_proj_bias)
                else:
                    v = linear(value, v_proj_weight_non_opt, in_proj_bias)

    if norms is not None:
        #q = norms[0](q)/3.0
        #k = norms[1](k)/3.0
        q = norms[0](q)
        k = norms[1](k)
        v = norms[2](v)
    else:
        q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    if 'bmm1' in attention_type:
        if attention_func is None or attention_func == '':
            pass
        elif attention_func == 'hardtanh':
            q = torch.nn.functional.hardtanh(q).clone()
            k = torch.nn.functional.hardtanh(k).clone()
        elif attention_func == 'elu':
            q = torch.nn.functional.elu(q).clone()
            k = torch.nn.functional.elu(k).clone()
        elif attention_func == 'hardswish':
            q = torch.nn.functional.hardswish(q).clone()
            k = torch.nn.functional.hardswish(k).clone()
        elif attention_func == 'logistic':
            q = torch.nn.functional.sigmoid(q).clone()
            k = torch.nn.functional.sigmoid(k).clone()
        else:
            raise ValueError(f'Activation function not supported for attention: {attention_func}')

        if args.attn_trunc_low > 0 or args.attn_trunc_high < 100:
            low, high, mode = args.attn_trunc_low, args.attn_trunc_high, args.attn_trunc_mode
            k = truncate_tensor(k, low, high, mode)
            q = truncate_tensor(q, low, high, mode)

        num_splits = getattr(args, 'num_splits', 1)
        if num_splits > 1:
            split_size = q.shape[2]//num_splits
            subq = torch.split(q, split_size, dim=2)
            subk = torch.split(k, split_size, dim=2)
            if num_splits == 2:
                attn_output_weights = bnb.bmm(subq[0], subk[0].transpose(1, 2)) + bnb.bmm(subq[1], subk[1].transpose(1, 2))
            elif num_splits == 4:
                attn_output_weights = bnb.bmm(subq[0], subk[0].transpose(1, 2)) + bnb.bmm(subq[1], subk[1].transpose(1, 2)) + \
                              bnb.bmm(subq[2], subk[2].transpose(1, 2)) + bnb.bmm(subq[3], subk[3].transpose(1, 2))
            else:
                raise ValueError(f'Num splits not supported: num_splite={num_splits}')
        else:
            #attn_output_weights = bnb.bmm(q, k.transpose(1, 2), None, quant_type, attention_bmm_kits)
            attn_output_weights = bnb.bmm(q, k.transpose(1, 2))
    else:
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
    attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=dropout_p, training=training)

    if 'bmm2' in attention_type:
        num_splits = getattr(args, 'num_splits', 1)
        if num_splits > 1:
            split_size = attn_output_weights.shape[2]//num_splits
            subattn = torch.split(attn_output_weights, split_size, dim=2)
            subv = torch.split(v, split_size, dim=1)
            if num_splits == 2:
                attn_output = bnb.bmm(subattn[0], subv[0]) + bnb.bmm(subattn[1], subv[1])
            elif num_splits == 4:
                attn_output = bnb.bmm(subattn[0], subv[0]) + bnb.bmm(subattn[1], subv[1]) + \
                              bnb.bmm(subattn[2], subv[2]) + bnb.bmm(subattn[3], subv[3])
            else:
                raise ValueError(f'Num splits not supported: num_splite={num_splits}')
        else:
            attn_output = bnb.bmm(attn_output_weights, v)
    else:
        attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    if 'linear' in attention_type or 'out' in attention_type:
        if args.attn_trunc_low > 0 or args.attn_trunc_high < 100:
            attn_output = truncate_tensor(attn_output, args.attn_trunc_low, args.attn_trunc_high, args.attn_trunc_mode)
        attn_output = linear8bit(attn_output, out_proj_weight, out_proj_bias)
    else:
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
