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
        # Layer normalization
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Inverted bottleneck MLP
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Expand 4x
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # Compress back
        
    def forward(self, x):
        residual = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute for LayerNorm
        x = x.permute(0, 2, 3, 1)
        
        # LayerNorm + MLP
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Permute back
        x = x.permute(0, 3, 1, 2)
        
        # Residual connection
        x = residual + x
        return x
    