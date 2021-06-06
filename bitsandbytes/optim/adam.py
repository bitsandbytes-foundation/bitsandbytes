import torch
from torch.optim import Optimizer
from bitsandbytes.optim.optimizer import Optimizer8bit
import bitsandbytes.functional as F

class MockArgs(object):
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])

class Adam(Optimizer8bit):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, optim_bits=32, args=None, override_with_args=False):
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
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

        if args is None:
            args = {}
            args['optim_bits'] = optim_bits
            args['adam8bits_offset'] = 1/512
            args['percentile_clipping'] = 100

            self.args = MockArgs(args)
        else:
            self.args = args

        self.keep_32_bit = set()


    def set_state_bits(self, model, keep32type=[torch.nn.Embedding], keep32smaller=4096):
        for module, p in model.named_modules():
            if any([isinstance(module, t) for t in keep32type]):
                for p2 in module.parameters():
                    self.keep_32_bit.add(p2.data.storage().data_ptr())
            if p.numel() < keep32smaller:
                self.keep_32_bit.add(p.data.storage().data_ptr())

    @torch.no_grad()
    def init_state(self, group, p_id, p):
        if self.args.optim_bits == 32:
            dtype = torch.float32
        elif self.args.optim_bits == 8:
            dtype = torch.uint8
        else: raise NotImplementedError('Amount of Adam bits not supported')

        state = self.state[p]
        state['step'] = 0
        if p.numel() % 4 != 0:
            raise ValueError(f'Parameter tensors need to have a multiple of 4: {p.shape}')

        if dtype == torch.float32 or (dtype == torch.uint8 and p.numel() < 4096):
            state['state1'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32, device=p.device)
            state['state2'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32, device=p.device)
        elif dtype == torch.uint8:
            state['qtbl1'] = torch.zeros((256,), dtype=torch.float32, device=p.device)
            state['qtbl2'] = torch.zeros((256,), dtype=torch.float32, device=p.device)
            state['max1'] = torch.zeros((1,), dtype=torch.float32, device=p.device)
            state['max2'] = torch.zeros((1,), dtype=torch.float32, device=p.device)

        if self.args.percentile_clipping < 100:
            state['gnorm_vec'] = torch.zeros((100,), device=p.device)
        #if self.args.unorm != 'none':
            #state['unorm_vec'] = torch.zeros((100,), device=p.device)


    @torch.no_grad()
    def update_step(self, group, p_id, p):
        state = self.state[p]
        grad = p.grad
        beta1, beta2 = group['betas']

        state['step'] += 1
        step = state['step']

        F.adam_update(grad, p, state['state1'], state['state2'], beta1, beta2, group['eps'], group['weight_decay'], step, group['lr'])

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
        for group in self.param_groups:
            for p_id, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p_id, p)

                self.update_step(group, p_id, p)

        return loss

class Adam32bit(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, args=None, override_with_args=False):
        super(Adam32bit, self).__init__(params, lr, betas, eps, weight_decay, amsgrad, args, override_with_args)
        self.args.optim_bits = 32

class Adam8bit(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, args=None, override_with_args=False):
        super(Adam32bit, self).__init__(params, lr, betas, eps, weight_decay, amsgrad, args, override_with_args)
        self.args.optim_bits = 8

