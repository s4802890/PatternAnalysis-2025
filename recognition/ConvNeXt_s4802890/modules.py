# modules.py - Start with just the ConvNeXtBlock skeleton
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block - Basic building block
    """
    def __init__(self, dim):
        super().__init__()
        # Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
    def forward(self, x):
        return self.dwconv(x)