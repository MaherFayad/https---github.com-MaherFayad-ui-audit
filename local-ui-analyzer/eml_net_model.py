"""
EML-NET Architecture with EfficientNet-V2 Backbone and FPN Decoder.

Upgraded architecture for improved UI saliency prediction:
- EfficientNet-V2-S encoder for better multi-scale feature extraction
- FPN-style decoder with lateral connections for high-resolution detail
- Compatible with existing SaliencyEngine interface

Author: Refactored for UX-Heatmap v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network decoder with lateral connections.
    
    Progressively upsamples and fuses features from encoder stages
    to preserve high-resolution spatial details.
    """
    def __init__(self, encoder_channels: list, fpn_channels: int = 256):
        """
        Args:
            encoder_channels: List of channel counts from encoder stages (low to high res)
                              e.g., [24, 48, 64, 128, 256] for EfficientNet-V2-S
            fpn_channels: Unified channel count for FPN layers
        """
        super(FPNDecoder, self).__init__()
        
        # Lateral 1x1 convolutions (reduce encoder channels to fpn_channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, fpn_channels, kernel_size=1, bias=False)
            for in_ch in encoder_channels
        ])
        
        # Refinement 3x3 convolutions (after each fusion)
        self.refine_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            )
            for _ in encoder_channels
        ])
        
        # Final output head
        self.output_conv = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels // 2, 1, kernel_size=1)
        )
        
    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of encoder feature maps [P1, P2, P3, P4, P5]
                     from low-resolution to high-resolution
        Returns:
            Saliency map at original resolution
        """
        # Start from deepest (lowest resolution) level
        # features[-1] is the deepest, features[0] is shallowest
        
        # Apply lateral conv to deepest feature
        x = self.lateral_convs[-1](features[-1])
        x = self.refine_convs[-1](x)
        
        # Bottom-up pathway with lateral connections
        for i in range(len(features) - 2, -1, -1):
            # Upsample current feature
            x = F.interpolate(x, size=features[i].shape[2:], mode='bilinear', align_corners=True)
            # Add lateral connection
            lateral = self.lateral_convs[i](features[i])
            x = x + lateral
            # Refine
            x = self.refine_convs[i](x)
        
        # Final output
        return self.output_conv(x)


class EMLNet(nn.Module):
    """
    EML-NET v2 with EfficientNet-V2-S backbone and FPN decoder.
    
    Maintains API compatibility with original EMLNet class.
    Output is raw logits (apply sigmoid externally for [0,1] range).
    """
    def __init__(self, pretrained: bool = True):
        super(EMLNet, self).__init__()
        
        # Load EfficientNet-V2-S backbone
        if pretrained:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
        else:
            weights = None
            
        efficientnet = models.efficientnet_v2_s(weights=weights)
        
        # Extract feature stages from EfficientNet
        # EfficientNet-V2-S features structure:
        # features[0]: Conv stem (24 ch, stride 2)
        # features[1]: Stage 1 (24 ch)
        # features[2]: Stage 2 (48 ch)
        # features[3]: Stage 3 (64 ch)
        # features[4]: Stage 4 (128 ch)
        # features[5]: Stage 5 (160 ch)
        # features[6]: Stage 6 (256 ch)
        # features[7]: Conv head (1280 ch)
        
        self.stem = efficientnet.features[0]        # 24 ch, 1/2
        self.stage1 = efficientnet.features[1]      # 24 ch, 1/2
        self.stage2 = efficientnet.features[2]      # 48 ch, 1/4
        self.stage3 = efficientnet.features[3]      # 64 ch, 1/8
        self.stage4 = nn.Sequential(
            efficientnet.features[4],
            efficientnet.features[5]
        )                                           # 160 ch, 1/16
        self.stage5 = efficientnet.features[6]      # 256 ch, 1/32
        
        # FPN Decoder
        # Channel counts: [24, 48, 64, 160, 256]
        encoder_channels = [24, 48, 64, 160, 256]
        self.decoder = FPNDecoder(encoder_channels, fpn_channels=128)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, 3, H, W)
        Returns:
            Saliency map (B, 1, H, W) - raw logits
        """
        input_size = x.shape[2:]
        
        # Encoder forward pass
        c0 = self.stem(x)       # 24 ch, H/2 x W/2
        c1 = self.stage1(c0)    # 24 ch, H/2 x W/2
        c2 = self.stage2(c1)    # 48 ch, H/4 x W/4
        c3 = self.stage3(c2)    # 64 ch, H/8 x W/8
        c4 = self.stage4(c3)    # 160 ch, H/16 x W/16
        c5 = self.stage5(c4)    # 256 ch, H/32 x W/32
        
        # FPN Decoder (skip c0 as it's same resolution as c1)
        features = [c1, c2, c3, c4, c5]
        out = self.decoder(features)
        
        # Upsample to input resolution
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return out


# Backward compatibility alias
EMLNetV1 = EMLNet


if __name__ == "__main__":
    # Quick test
    model = EMLNet()
    x = torch.randn(2, 3, 480, 640)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
