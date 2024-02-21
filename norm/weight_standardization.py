import torch
from torch import nn
from torch.nn import functional as F

def weight_standardization(weight: torch.Tensor, eps: float):
    
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out,-1)

    var, mean = torch.var_mean(weight, dim=1,keep_dim = True)

    weight = (weight - mean) / torch.sqrt(var + eps)

    return torch.view(c_out, c_in, *kernel_shape)

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps: float = 1e-5):
        
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.eps = eps

    def weight_standardization(self, weight: torch.Tensor, eps: float):
        
        
        c_out, c_in, *kernel_shape = weight.shape
        weight = weight.view(c_out,-1)

        var, mean = torch.var_mean(weight, dim=1,keepdim=True)

        weight = (weight - mean) / torch.sqrt(var + eps)

        return weight.view(c_out, c_in, *kernel_shape)

    def forward(self, x):
        return F.conv2d(x, self.weight_standardization(self.weight,self.eps), self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
    
if __name__ == '__main__':
    conv2d = Conv2d(10, 20, 5)
    print(conv2d.weight.shape)
    print(conv2d(torch.zeros(10, 10, 100, 100)))
    
