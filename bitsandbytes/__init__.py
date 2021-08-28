from .optim import adam
from .nn import modules
from .autograd._functions import mm, bmm, matmul

__pdoc__ = {'libBitsNBytes' : False,
            'optim.optimizer.Optimizer8bit': False,
            'optim.optimizer.MockArgs': False
           }
