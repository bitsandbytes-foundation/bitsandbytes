# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

from .nn import modules
from .cextension import COMPILED_WITH_CUDA

if COMPILED_WITH_CUDA:
    from .optim import adam

__pdoc__ = {'libBitsNBytes': False,
            'optim.optimizer.Optimizer8bit': False,
            'optim.optimizer.MockArgs': False
            }
