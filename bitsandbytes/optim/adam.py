from bitsandbytes.optim.optimizer import Optimizer2State

class Adam(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, optim_bits=32, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        super(Adam, self).__init__('adam', params, lr, betas, eps,
                weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise)

class Adam8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        super(Adam8bit, self).__init__('adam', params, lr, betas, eps,
                weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise)

class Adam32bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, args=None,
            min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        super(Adam32bit, self).__init__('adam', params, lr, betas, eps,
                weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise)


