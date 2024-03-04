import math
from typing import Dict, Any, Tuple, Optional
import torch
from torch import nn
from optimizers import GenericAdaptiveOptimizer,WeightDecay

class Adam(GenericAdaptiveOptimizer):
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-16,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 defaults: Optional[Dict[str,Any]] = None):
        
        '''
        params : list of parameters
        lr : learning rate
        betas : tuple of beta1,beta2
        eps : epsilon for div
        weight-decay : class WeightDecay
        optimized_update : wheter to optimize the bias correction of the second moment by doing it after adding epsilon
        defaults : dictionary of default for group values
        '''

        defaults = {} if defaults is None else defaults
        defaults.update(weight_decay.defaults()) # default dictionary에 weight_decay dict 추가
        super().__init__(params, defaults, lr, betas, eps)

        self.weight_decay = weight_decay
        self.optimized_update = optimized_update

    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        '''
        state = optimizer state of the parameter(tensor)
        group = optimizer attributes of the parameter group
        param = parameter tensor \theta_{t-1}
        '''
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format) # momentum
        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format) # rmsprop

    def get_mv(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor):
        beta1, beta2 = group['betas']

        m, v = state['exp_avg'], state['exp_avg_sq']

        # m_t <- beta1*m_{t-1} + (1-beta1)*g_t
        m.mul_(beta1).add_(grad,alpha=1-beta1)
        # v_t <- beta2*v_{t-1} + (1-beta2)*g_t^2
        v.mul_(beta2).addcmul_(grad,grad,value = 1-beta2)

        return m,v
    
    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        return group['lr']
    
    def adam_update(self, state: Dict[str, any], group: Dict[str, any], param: torch.nn.Parameter, m: torch.Tensor, v: torch.Tensor):
        
        beta1, beta2 = group['betas']

        # Adam의 bias correction
        bias_correction1 = 1-beta1**state['step']
        bias_correction2 = 1-beta2**state['step']
        lr = self.get_lr(state,group)

        if self.optimized_update:
            denominator = v.sqrt().add_(group['eps']) # \sqrt{v_t}+\eps
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            param.data.addcdiv_(m, denominator, value=-step_size)

        else:
            denominator = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            step_size = lr / bias_correction1
            param.data.addcdiv_(m, denominator, value=-step_size)

    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        grad = self.weight_decay(param, grad, group)
        m, v = self.get_mv(state, group, grad)
        state['step'] += 1
        self.adam_update(state, group, param, m, v)