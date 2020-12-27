import torch
from torch import nn
import math

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channels: Number of channels in the input tensor
        b: Hyper-parameter for adaptive kernel size formulation. Default: 1
        gamma: Hyper-parameter for adaptive kernel size formulation. Default: 2 
    """
    def __init__(self, channels, b=1, gamma=2):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out

    def forward(self, x):

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        