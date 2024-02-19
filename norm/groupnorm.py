import torch
from torch import nn

class GroupNorm(nn.Module):
    def __init__(self, groups: int, channels: int, *,
                 eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = False):
        
        super().__init__()
        
        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
        self.channels = channels

        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

        if self.track_running_stats:
            self.register_buffer('exp_mean', None)
            self.register_buffer('exp_var', None)

    def forward(self, x: torch.Tensor):
        
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]

        x = x.view(batch_size,self.groups,-1)
        
        if self.training or not self.track_running_stats: # test 단계에서 running average를 사용하지 않아도 진입(mean, var 계산을 위해)
            mean = x.mean(dim=[-1],keepdim=True)
            mean_x2 = (x**2).mean(dim=[-1],keepdim=True)

            var = mean_x2 - mean**2

            if self.training and self.track_running_stats:
                if self.exp_mean == None:
                    self.exp_mean 
                else:
                    self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                    self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var
        
        else: # test 단계에서 running average사용시 진입(training에서 running average를 통해 mean, var가 계산되어있기 때문)
            mean = self.exp_mean
            var = self.exp_var

        x_norm = (x-mean) / torch.sqrt(var + self.eps)
        

        if self.affine:
            x_norm = x_norm.view(batch_size, self.channels, -1)
            x_norm = self.scale.view(1,-1,1)*x_norm + self.shift.view(1,-1,1)

        return x_norm.view(x_shape)
    
if __name__=='__main__':
    x = torch.zeros([2, 6, 2, 4])
    print(x.shape)
    bn = GroupNorm(2,6,track_running_stats=True)

    x = bn(x)
    print(x.shape)