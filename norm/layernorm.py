from typing import Union, List
import torch
from torch import nn, Size

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        '''
        normalized_shape: input x의 shape
        eps: normalize 분모가 0이 되는 것을 방지하기 위한 작은 상수
        elementwise_affine: gamma와 beta 사용 여부
        '''
        
        super().__init__()

        # input으로 들어온 shape 확인 후 tensor의 size로 변환
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape]) # torch.Size -> return size of tensor
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine: # gamma, beta 사용 여부
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor):

        assert self.normalized_shape == x.shape[-len(self.normalized_shape):] # x의 shape = (batch, token, feature)이면, normalized_shape = [token,feature] 즉, batch를 제외한 shape이 같아야한다.
        
        dims = [-(i + 1) for i in range(len(self.normalized_shape))] # len(self.normalized_shape) = 2 이면, dims = [-1,-2]

        mean = x.mean(dim=dims, keepdim=True) # 위의 예시를 이용하면 mean의 shape = (batch,1,1)
    
        mean_x2 = (x**2).mean(dim=dims, keepdim=True)

        var = mean_x2 - mean**2

        # norm 계산
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine: # gamma, beta 이용하면
            x_norm = self.gain * x_norm + self.bias

        return x_norm
    
if __name__ == '__main__':
    x = torch.zeros([2, 3, 2, 4])
    print(x.shape)
    ln = LayerNorm(x.shape[2:])
    x = ln(x)
    print(x.shape)
    print(ln.gain.shape)