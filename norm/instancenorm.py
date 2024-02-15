import torch
from torch import nn

class instancenorm(nn.Module):
    def __init__(self, channels: int, *,
                 eps: float = 1e-5, affine: bool = True):
        pass