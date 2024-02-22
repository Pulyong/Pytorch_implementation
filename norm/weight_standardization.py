import torch
from torch import nn
from torch.nn import functional as F

#################################### 기본 코드 ####################################

def weight_standardization(weight: torch.Tensor, eps: float):
    
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out,-1)

    var, mean = torch.var_mean(weight, dim=1,keep_dim = True)

    weight = (weight - mean) / torch.sqrt(var + eps)

    return torch.view(c_out, c_in, *kernel_shape)

#################################################################################


# Pytorch conv2d에 적용
class Conv2d(nn.Conv2d): # torch Conv2d 상속
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps: float = 1e-5): # 기본 input은 같으나 마지막 epsilon을 받음(normalize 분모에 사용)
        
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias) #nn.Conv2d에 파라미터 전달

        self.eps = eps

    def weight_standardization(self, weight: torch.Tensor, eps: float): 
        
        
        c_out, c_in, *kernel_shape = weight.shape # weight의 shape 
        weight = weight.view(c_out,-1) # c_out을 기준으로 flatten

        var, mean = torch.var_mean(weight, dim=1,keepdim=True)

        weight = (weight - mean) / torch.sqrt(var + eps)

        return weight.view(c_out, c_in, *kernel_shape)

    def forward(self, x):
        return F.conv2d(x, self.weight_standardization(self.weight,self.eps), self.bias, self.stride,
                          self.padding, self.dilation, self.groups) # weight에 normalized weight를 넣어줌
    
# Pytorch conv3d에 적용
class Conv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps: float = 1e-5):
        
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.eps = eps

    def weight_standardization(self, weight: torch.Tensor, eps: float):
        
        
        c_out, c_in, *kernel_shape = weight.shape # weight의 shape 
        weight = weight.view(c_out,-1) # c_out을 기준으로 flatten (Conv3d도 똑같다)

        var, mean = torch.var_mean(weight, dim=1,keepdim=True)

        weight = (weight - mean) / torch.sqrt(var + eps)

        return weight.view(c_out, c_in, *kernel_shape)

    def forward(self, x):
        return F.conv3d(x, self.weight_standardization(self.weight,self.eps), self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
    
if __name__ == '__main__':
    conv2d = Conv2d(10, 20, 5)
    print(conv2d.weight.shape)
    print(conv2d(torch.zeros(10, 10, 100, 100)))
    
    conv3d = Conv3d(10,20,5)
    print(conv3d.weight.shape)
    print(conv3d(torch.zeros(10, 10, 10, 100, 100)))
    
