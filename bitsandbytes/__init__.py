# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
from .optim import adam
from .nn import modules
__pdoc__ = {'libBitsNBytes' : False,
            'optim.optimizer.Optimizer8bit': False,
            'optim.optimizer.MockArgs': False
           }
