import torch
import bitsandbytes as bnb

from typing import Optional

from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from bitsandbytes.optim import GlobalOptimManager

class StableEmbedding(torch.nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = True, _weight: Optional[Tensor] = None) -> None:
        super(StableEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, False, _weight)
        self.norm = torch.nn.LayerNorm(embedding_dim)
        GlobalOptimManager.get_instance().register_parameters(self.weight)
        GlobalOptimManager.get_instance().override_config(self.weight, 'optim_bits', 32)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)
        self._fill_padding_idx_with_zero()

    ''' !!! This is a redefinition of _fill_padding_idx_with_zero in torch.nn.Embedding
        to make the Layer compatible with Pytorch < 1.9.
        This means that if this changes in future PyTorch releases this need to change too
        which is cumbersome. However, with this we can ensure compatibility with previous
        PyTorch releases.
    '''
    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        emb = F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

        return self.norm(emb)

class Linear8bit(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, quant_type='vector', index=None):
        super(Linear8bit, self).__init__(input_features, output_features, bias)
        self.quant_type = quant_type
        self.index = index

    def forward(self, x):
        #with torch.no_grad():
            #Cw, Sw = bnb.functional.quantize_blockwise(self.weight)
            #w = bnb.functional.dequantize_blockwise(Cw, Sw)
            #Cw, Sw = bnb.functional.vectorwise_quant(self.weight)
            #w = bnb.functional.vectorwise_dequant(Cw, Sw)
            #self.weight.copy_(w)
            #torch.cuda.synchronize()
        out = bnb.matmul(x, self.weight.t(), self.bias, self.quant_type, [8, 8, 8], self.index)
        if self.bias is not None:
            out += self.bias.unsqueeze(0).expand_as(out)
        return out



