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
    
    class ConvNeXt(nn.Module):
        """
        ConvNeXt Model for ADNI binary classification
        Architecture: Stem -> 4 Stages -> Classification Head
        """
        def __init__(self, in_channels=1, num_classes=2):
            super().__init__()
            
            # Stem: Aggressive downsampling 224x224 -> 56x56
            # 4x4 conv with stride 4 (patchify operation)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 96, kernel_size=4, stride=4),
                nn.LayerNorm(96, eps=1e-6)
            )
            
        def forward(self, x):
            # Input: (B, 1, 224, 224)
            x = self.stem(x)
            # Output: (B, 96, 56, 56)
            return x