"""
src/models/hybrid_swin_unet.py
==============================
3D Hybrid Swin-UNet Architecture for Few-Shot Cross-Domain Multi-Organ Segmentation.

Key Features:
1. Hierarchical 3D Swin Transformer Encoder with shifted-window self-attention (W-MSA / SW-MSA).
2. Linear computational complexity O(N * M^3) with 3D cyclic shift and relative position bias.
3. Automatic spatial padding/unpadding for arbitrary resolutions (Stage 4 bottleneck 6x6x6).
4. Skip connections transferring multi-scale features from encoder stages to decoder.
5. 3D Convolutional U-Net Decoder with trilinear upsampling and GeLU activation.
6. Multi-organ segmentation head predicting C=14 classes (Background + 13 organs).
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition_3d(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """Partitions 5D tensor (B, D, H, W, C) into non-overlapping 3D windows."""
    B, D, H, W, C = x.shape
    Wd, Wh, Ww = window_size
    x = x.view(B, D // Wd, Wd, H // Wh, Wh, W // Ww, Ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, Wd * Wh * Ww, C)
    return windows


def window_reverse_3d(
    windows: torch.Tensor,
    window_size: Tuple[int, int, int],
    dims: Tuple[int, int, int, int]
) -> torch.Tensor:
    """Reconstructs 5D tensor (B, D, H, W, C) from 3D windows."""
    B, D, H, W = dims
    Wd, Wh, Ww = window_size
    x = windows.view(B, D // Wd, H // Wh, W // Ww, Wd, Wh, Ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    """3D Window-based Multi-head Self-Attention (W-MSA / SW-MSA) with relative positional bias."""
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP3D(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP3D(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        
        _, Dp, Hp, Wp, _ = x.shape

        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition_3d(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, (B, Dp, Hp, Wp))

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


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


class DecoderStage3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock3D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class HybridSwinUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 14,
        embed_dim: int = 24,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.window_size = window_size

        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=2, stride=2)
        self.stem_norm = nn.LayerNorm(embed_dim)
        self.input_stem = ConvBlock3D(in_channels, embed_dim)

        self.stage1_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=embed_dim,
                num_heads=num_heads[0],
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0]//2, window_size[1]//2, window_size[2]//2)
            ) for i in range(depths[0])
        ])
        self.merge1 = PatchMerging3D(embed_dim)

        self.stage2_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=2 * embed_dim,
                num_heads=num_heads[1],
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0]//2, window_size[1]//2, window_size[2]//2)
            ) for i in range(depths[1])
        ])
        self.merge2 = PatchMerging3D(2 * embed_dim)

        self.stage3_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=4 * embed_dim,
                num_heads=num_heads[2],
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0]//2, window_size[1]//2, window_size[2]//2)
            ) for i in range(depths[2])
        ])
        self.merge3 = PatchMerging3D(4 * embed_dim)

        self.stage4_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=8 * embed_dim,
                num_heads=num_heads[3],
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0]//2, window_size[1]//2, window_size[2]//2)
            ) for i in range(depths[3])
        ])

        self.dec3 = DecoderStage3D(in_channels=8 * embed_dim, skip_channels=4 * embed_dim, out_channels=4 * embed_dim)
        self.dec2 = DecoderStage3D(in_channels=4 * embed_dim, skip_channels=2 * embed_dim, out_channels=2 * embed_dim)
        self.dec1 = DecoderStage3D(in_channels=2 * embed_dim, skip_channels=embed_dim, out_channels=embed_dim)
        self.dec0 = DecoderStage3D(in_channels=embed_dim, skip_channels=embed_dim, out_channels=embed_dim)

        self.final_head = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.input_stem(x)
        stem_out = self.patch_embed(x)
        s1 = stem_out.permute(0, 2, 3, 4, 1).contiguous()
        s1 = self.stem_norm(s1)

        for blk in self.stage1_blocks:
            s1 = blk(s1)
        skip1 = s1.permute(0, 4, 1, 2, 3).contiguous()
        s2 = self.merge1(s1)

        for blk in self.stage2_blocks:
            s2 = blk(s2)
        skip2 = s2.permute(0, 4, 1, 2, 3).contiguous()
        s3 = self.merge2(s2)

        for blk in self.stage3_blocks:
            s3 = blk(s3)
        skip3 = s3.permute(0, 4, 1, 2, 3).contiguous()
        s4 = self.merge3(s3)

        for blk in self.stage4_blocks:
            s4 = blk(s4)
        d4 = s4.permute(0, 4, 1, 2, 3).contiguous()

        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        d0 = self.dec0(d1, skip0)

        logits = self.final_head(d0)
        return logits
