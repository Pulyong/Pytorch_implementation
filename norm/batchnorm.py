import torch
from torch import nn

class BatchNorm(nn.Module):
    def __init__(self, channels: int, *,
                 eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_state: bool = True):
        '''
        channels: input의 feature 수. image면 channel의 수
        eps: normalize과정에서 분모가 0이되지 않게 만드는 작은 상수
        momentum: exponential moving average인 momentum의 사용 여부
        affine: gamma와 beta의 사용 여부
        track_running_stats: 계산에 moving averages를 사용할 것인지 mean,var를 사용할 것인지
        '''
        
        super().__init__()

        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_state

        if self.affine: # gamma, beta create
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

        if self.track_running_stats: # mean과 variance에 대한 moving averages를 위한 buffer 생성
            self.register_buffer('exp_mean', torch.zeros(channels))
            self.register_buffer('exp_var', torch.ones(channels))
    
    def forward(self, x: torch.Tensor):
        '''
        x의 shape는 [batch_size, channels, *]. *는 0이 아닌 dimension
        image의 경우 [batch_size, channels, H, W]
        mlp의 경우 [batch_size,features]
        embedding의 경우 [batch_size,token,embd dim]
        '''
        
        x_shape = x.shape
        batch_size = x_shape[0]

        assert self.channels == x_shape[1]

        x = x.view(batch_size,self.channels,-1) # channel 밑으로 flatten (mean, var 계산을 위해) shape: (batch,channel,나머지)

        if self.training or not self.track_running_stats: # test 단계에서 running average를 사용하지 않아도 진입(mean, var 계산을 위해)
            mean = x.mean(dim = [0,2]) # shape: (channel)
            mean_x2 = (x**2).mean(dim=[0,2])

            var = mean_x2 - mean**2

            if self.training and self.track_running_stats:
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var
        
        else: # test 단계에서 running average사용시 진입(training에서 running average를 통해 mean, var가 계산되어있기 때문)
            mean = self.exp_mean
            var = self.exp_var

        # batchnorm 계산
        x_norm = (x - mean.view(1,-1, 1)) / torch.sqrt(var + self.eps).view(1, -1, 1) # x와의 shape을 맞춰주기 위해 view. batch와 나머지 차원축에 대해 broadcast 진행되어 계산

        if self.affine: # gamma, beta 곱하고 더해주기
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1) # x와의 shape을 맞춰주기 위해 view. broadcast후 연산 진행

        return x_norm.view(x_shape) # 다시 원래 x shape로 복구

if __name__ == '__main__':
    x = torch.zeros([2,3,2,4])
    print(x.shape)
    bn = BatchNorm(3)

    x = bn(x)
    print(x.shape)
    print(bn.exp_var.shape)