from bitsandbytes.optim.optimizer import Optimizer1State

class SGD(Optimizer1State):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, optim_bits=32, args=None,
            min_8bit_size=4096, percentile_clipping=100):
        if momentum == 0:
            raise NotImplementError(f'SGD without momentum is not supported!')
        super(SGD, self).__init__('momentum', params, lr, (momentum, dampening), 0.0,
                weight_decay, optim_bits, args, min_8bit_size, percentile_clipping)

class SGD8bit(Optimizer1State):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, args=None,
            min_8bit_size=4096, percentile_clipping=100):
        if momentum == 0:
            raise NotImplementError(f'SGD without momentum is not supported!')
        super(SGD8bit, self).__init__('momentum', params, lr, (momentum, dampening), 0.0,
                weight_decay, 8, args, min_8bit_size, percentile_clipping)

class SGD32bit(Optimizer1State):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, args=None,
            min_8bit_size=4096, percentile_clipping=100):
        if momentum == 0:
            raise NotImplementError(f'SGD without momentum is not supported!')
        super(SGD32bit, self).__init__('momentum', params, lr, (momentum, dampening), 0.0,
                weight_decay, 32, args, min_8bit_size, percentile_clipping)
