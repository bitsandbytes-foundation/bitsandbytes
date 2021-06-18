import torch
from torch.optim import Optimizer
from bitsandbytes.optim.optimizer import Optimizer8bit, MockArgs
import bitsandbytes.functional as F

class Adam(Optimizer8bit):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, optim_bits=32, is_sparse=False, args=None, override_with_args=False,
            min_8bit_size=4096, percentile_clipping=100):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, is_sparse=is_sparse)
        super(Adam, self).__init__(params, defaults)

        if args is None:
            args = {}
            args['optim_bits'] = optim_bits
            args['adam8bits_offset'] = 1/512
            args['percentile_clipping'] = 100
            args['is_sparse'] = is_sparse
            args['min_8bit_size'] = min_8bit_size
            args['percentile_clipping'] = percentile_clipping

            self.args = MockArgs(args)
        else:
            self.args = args

        self.keep_32_bit = set()
        self.name2qmap = {}
        if self.args.optim_bits == 8: self.fill_qmap()

    def fill_qmap(self):
        self.name2qmap['dynamic'] = F.create_dynamic_map(signed=True)
        self.name2qmap['udynamic'] = F.create_dynamic_map(signed=False)

    def get_config(self, gindex, pindex, group):
        config = {}
        config['betas'] = group['betas']
        config['eps'] = group['eps']
        config['weight_decay'] = group['weight_decay']
        config['lr'] = group['lr']
        config['is_sparse'] = self.args.is_sparse
        config['optim_bits'] = self.args.optim_bits
        config['min_8bit_size'] = self.args.min_8bit_size
        config['percentile_clipping'] = self.args.percentile_clipping

        if (gindex, pindex) in self.mng.index2config:
            config.update(self.mng.index2config[(gindex, pindex)])
        return config

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)

        if config['optim_bits'] == 32:
            dtype = torch.float32
        elif config['optim_bits'] == 8:
            dtype = torch.uint8
        else: raise NotImplementedError(f'Amount of optimizer bits not supported: {config["optim_bits"]}')

        if p.numel() < config['min_8bit_size']: dtype = torch.float32

        state = self.state[p]
        state['step'] = 0

        if dtype == torch.float32 or (dtype == torch.uint8 and p.numel() < 4096):
            state['state1'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32, device=p.device)
            state['state2'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32, device=p.device)
        elif dtype == torch.uint8:
            if state['step'] == 0:
                if 'dynamic' not in self.name2qmap: self.fill_qmap()
                self.name2qmap['dynamic'] = self.name2qmap['dynamic'].to(p.device)
                self.name2qmap['udynamic'] = self.name2qmap['udynamic'].to(p.device)

            state['state1'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.uint8, device=p.device)
            state['state2'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.uint8, device=p.device)
            state['qmap1'] = self.name2qmap['dynamic']
            state['qmap2'] = self.name2qmap['udynamic']
            state['max1'] = torch.zeros((1,), dtype=torch.float32, device=p.device)
            state['max2'] = torch.zeros((1,), dtype=torch.float32, device=p.device)
            state['new_max1'] = torch.zeros((1,), dtype=torch.float32, device=p.device)
            state['new_max2'] = torch.zeros((1,), dtype=torch.float32, device=p.device)

        if config['percentile_clipping'] < 100:
            state['gnorm_vec'] = torch.zeros((100,), device=p.device)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        state = self.state[p]
        grad = p.grad

        config = self.get_config(gindex, pindex, group)

        state['step'] += 1
        step = state['step']

        if config['percentile_clipping'] < 100:
            current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(grad, state['gnorm_vec'], step, config['percentile_clipping'])
        else:
            gnorm_scale = 1.0

        if state['state1'].dtype == torch.float:
            F.adam_update_32bit(grad, p, state['state1'], state['state2'], config['betas'][0], config['betas'][1],
                          config['eps'], step, config['lr'],
                          config['weight_decay'], is_sparse=config['is_sparse'], gnorm_scale=gnorm_scale)
        elif state['state1'].dtype == torch.uint8:
            F.adam_update_8bit(grad, p, state['state1'], state['state2'], config['betas'][0], config['betas'][1],
                          config['eps'],  step, config['lr'],
                          state['qmap1'], state['qmap2'], state['max1'], state['max2'], state['new_max1'], state['new_max2'],
                          config['weight_decay'], is_sparse=config['is_sparse'], gnorm_scale=gnorm_scale)
            # swap maxes
            state['max1'], state['new_max1'] = state['new_max1'], state['max1']
            state['max2'], state['new_max2'] = state['new_max2'], state['max2']


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        overflows = []
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)

                self.update_step(group, p, gindex, pindex)

        return loss

class Adam32bit(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, args=None, override_with_args=False):
        super(Adam32bit, self).__init__(params, lr, betas, eps, weight_decay, amsgrad, args, override_with_args)
        self.args.optim_bits = 32

def Adam8bit(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0, amsgrad=False, optim_bits=8, is_sparse=False, args=None, override_with_args=False):
    return Adam(params, lr, betas, eps, weight_decay, amsgrad, 8, is_sparse, args, override_with_args)

