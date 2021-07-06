from bitsandbytes.optim.optimizer import Optimizer1State

class LARS(Optimizer1State):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, optim_bits=32, is_sparse=False, args=None,
                 min_8bit_size=4096, percentile_clipping=100, max_unorm=0.02):
        if momentum == 0:
            raise NotImplementError(f'LARS without momentum is not supported!')
        super(LARS, self).__init__('lars', params, lr, (momentum, dampening), 0.0,
                weight_decay, optim_bits, is_sparse, args, min_8bit_size, percentile_clipping, max_unorm=max_unorm)

class LARS8bit(Optimizer1State):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, is_sparse=False, args=None,
                 min_8bit_size=4096, percentile_clipping=100, max_unorm=0.02):
        if momentum == 0:
            raise NotImplementError(f'LARS without momentum is not supported!')
        super(LARS8bit, self).__init__('lars', params, lr, (momentum, dampening), 0.0,
                weight_decay, 8, is_sparse, args, min_8bit_size, percentile_clipping, max_unorm=max_unorm)

class LARS32bit(Optimizer1State):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, is_sparse=False, args=None,
                 min_8bit_size=4096, percentile_clipping=100, max_unorm=0.02):
        if momentum == 0:
            raise NotImplementError(f'LARS without momentum is not supported!')
        super(LARS32bit, self).__init__('lars', params, lr, (momentum, dampening), 0.0,
                weight_decay, 32, is_sparse, args, min_8bit_size, percentile_clipping, max_unorm=max_unorm)
