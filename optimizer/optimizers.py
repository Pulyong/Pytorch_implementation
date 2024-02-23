import math
from typing import Dict, Any, Tuple, Optional

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

class GenericAdaptiveOptimizer(Optimizer):

    def __init__(self, params, defaults: Dict[str, Any], lr: float, betas: Tuple[float, float], eps: float):

        # check hyper-parameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults.update(dict(lr=lr, betas = betas, eps=eps)) # defaults라는 딕셔너리에 lr, beta, eps 추가

        super().__init__(params, defaults)

    # moving average등등 init
    def init_state(self, state: Dict[str, Any], group: Dict[str, Any], param: nn.Parameter):
        '''
        state: Optimizer의 parameter(moving average or num step ...)을 가지고 있는 Dict
        group: model의 param이 들어있는 Dict
        param: model의 parameters
        '''
        pass

    # optimizer를 통해 parameter update
    def step_param(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor, param: torch.Tensor):
        '''
        state: Optimizer의 parameter(moving average or num step ...)을 가지고 있는 Dict
        group: model의 param이 들어있는 Dict
        grad: parameter의 gradient
        param: model의 parameters
        '''
        pass

    @torch.no_grad()
    def step(self, closure=None):
        '''
        모든 Adam based optimizer가 필요한 step의 template
        '''
        loss = None # calculate loss
        if closure is not None:
            with torch.enable_grad(): # torch.no_grad등으로 gradient 계산이 불가능 할 때 gradient 계산을 가능하게 해줌
                loss = closure()

        for group in self.param_groups:
            for param in group['params']: # parameter만 가져오기
                if param.grad is None:
                    continue # freeze된 param은 건너 뛰기

                grad = param.grad.data

                if grad.is_sparse:
                    raise RuntimeError('GenericAdaptiveOptimizer does not support sparse gradients,'
                                       ' please consider SparseAdam instead')
                
                state = self.state[param]

                if len(state) == 0:
                    self.init_state(state, group, param)
                
                self.step_param(state, group, grad, param)

        return loss
    
# L2 weight decay
class WeightDecay:
    def __init__(self, weight_decay: float = 0., weight_decouple: bool = True, absolute: bool = False):
        '''
        weight_decay: weight_decay의 coefficient(weight 부여)
        weight_decouple: AdamW처럼 weight decay를 parameter update할 때 꽂아줄 것인지
        absolute: weight decay coefficient가 absolute인지 즉, learning rate를 사용할 것인지
        '''

        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        self.absolute = absolute
        self.weight_decouple = weight_decouple
        self.weight_decay = weight_decay

    def defaults(self):
        return dict(weight_decay=self.weight_decay)
    
    def __call__(self, param: torch.nn.Parameter, grad: torch.Tensor, group: Dict[str, any]):
        
        if self.weight_decouple:
            # absolute면 lr을 사용하지 않고 absolute한 weight decay로 업데이트
            if self.absolute:
                param.data.mul_(1.0 - group['weight_decay'])

            else:
                param.data.mul_(1.0 - group['lr'] * group['weight_decay'])

            return grad
        
        else:
            if group['weight_decay'] != 0:
                return grad.add(param.data, alpha=group['weight_decay'])
            else:
                return grad
            