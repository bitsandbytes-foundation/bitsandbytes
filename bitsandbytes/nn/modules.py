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
    def __init__(self, input_features, output_features, bias=True, quant_type='vector', index=None, args=None, sparse_decomp=False):
        super(Linear8bit, self).__init__(input_features, output_features, bias)
        self.quant_type = quant_type
        self.index = index
        self.args = args
        #if args.scale_mode == 'last':
        #    self.s = torch.ones((3,))*127*16
        #else:
        #    self.s = torch.ones((output_features, 10))*127*16
        #self.s[0] = index or 0
        #GlobalOptimManager.get_instance().override_config(self.s, 'lr', 1.0)
        #GlobalOptimManager.get_instance().register_parameters(self.s)
        self.iter = 0

    def forward(self, x):
        #if self.s is None:
        #    print('init s')
        #    self.s = torch.ones((x.shape[0]*x.shape[1],), device=x.device)*127*127
        #elif self.s.numel() != x.shape[0]*x.shape[1]:
        #    print('reinit s')
        #    self.s = torch.ones((x.shape[0]*x.shape[1],), device=x.device)*self.s.mean()

        if self.args is not None:
            out = bnb.nn.functional.sparse_decomposed_linear8bit(x, self.weight, self.bias, qval=self.args.sparse_decomp_val, quant_type=self.args.quant_type)
        else:
            out = bnb.nn.functional.linear8bit(x, self.weight, self.bias, quant_type=self.args.quant_type)

        return out



