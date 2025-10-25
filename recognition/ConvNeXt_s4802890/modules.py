import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block - Basic building block
    """
    def __init__(self, dim, drop_rate = 0.1):
        super().__init__()
        # Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # Layer normalization
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Inverted bottleneck MLP
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Expand 4x
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # Compress back
        
        self.drop = nn.Dropout(drop_rate)

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
        x = self.drop(x)
        x = self.pwconv2(x)
        x = self.drop(x)

        # Permute back
        x = x.permute(0, 3, 1, 2)
        
        # Residual connection
        x = residual + x
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class ConvNeXt(nn.Module):
    """
    ConvNeXt Model for ADNI binary classification
    Architecture: Stem -> 4 Stages (with downsampling) -> Classification Head
    
    Based on ConvNeXt-Tiny configuration:
    - Stage depths: [3, 3, 9, 3]
    - Feature dimensions: [96, 192, 384, 768]
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        
        # Channel dimensions for each stage
        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]
        
        # Stem: Aggressive downsampling 224x224 -> 56x56
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        
        self.stages = nn.ModuleList()
        
        for i in range(4):
            # Downsampling layer (between stages, not before stage 0)
            if i > 0:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[i-1]),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            else:
                downsample = nn.Identity()
            
            # Stage with multiple ConvNeXt blocks
            stage = nn.Sequential(
                downsample,
                *[ConvNeXtBlock(dims[i], drop_rate = 0.2) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        
        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1], eps=1e-6),
            nn.Linear(dims[-1], num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        Input: (B, 1, 224, 224)
        Output: (B, num_classes)
        """
        # Stem
        x = self.stem(x)
        
        # 4 stages with downsampling
        for stage in self.stages:
            x = stage(x)
        
        # Classification head
        x = self.head(x)
        
        return x

if __name__ == "__main__":
    # Test the model
    model = ConvNeXt(in_channels=1, num_classes=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
