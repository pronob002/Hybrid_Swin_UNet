"""
src/models/unet3d.py
====================
Standard 3D Convolutional U-Net Baseline for Multi-Organ Segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 14,
        base_channels: int = 24
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        self.enc0 = ConvBlock3D(in_channels, base_channels)
        self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock3D(base_channels, 2 * base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock3D(2 * base_channels, 4 * base_channels)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock3D(4 * base_channels, 8 * base_channels)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock3D(8 * base_channels, 16 * base_channels)

        self.up3 = nn.ConvTranspose3d(16 * base_channels, 8 * base_channels, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(16 * base_channels, 8 * base_channels)

        self.up2 = nn.ConvTranspose3d(8 * base_channels, 4 * base_channels, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(8 * base_channels, 4 * base_channels)

        self.up1 = nn.ConvTranspose3d(4 * base_channels, 2 * base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(4 * base_channels, 2 * base_channels)

        self.up0 = nn.ConvTranspose3d(2 * base_channels, base_channels, kernel_size=2, stride=2)
        self.dec0 = ConvBlock3D(2 * base_channels, base_channels)

        self.final_head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        p0 = self.pool0(e0)

        e1 = self.enc1(p0)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d0 = self.up0(d1)
        d0 = self.dec0(torch.cat([d0, e0], dim=1))

        logits = self.final_head(d0)
        return logits
