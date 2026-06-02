"""Spatial attention modules for mammography risk prediction.

Implements attention mechanisms that learn to focus on clinically relevant
regions of the mammogram -- dense tissue patterns, architectural distortions,
and microcalcification clusters -- without requiring explicit ROI annotations.

The attention maps are fully differentiable, enabling end-to-end training,
and can be extracted at inference time for interpretability overlays that
help radiologists understand model predictions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention.

    Learns to recalibrate channel-wise feature responses by explicitly
    modelling interdependencies between channels. Particularly useful for
    selecting which feature maps (texture, edges, density patterns) are
    most informative for risk prediction.

    Args:
        num_channels: Number of input channels.
        reduction_ratio: Reduction ratio for the bottleneck. Higher values
            use fewer parameters but may lose representational capacity.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        reduced = max(num_channels // reduction_ratio, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, num_channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Channel-recalibrated tensor of same shape.
        """
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module for localizing diagnostically relevant regions.

    Produces a 2D attention map that highlights spatial locations in the
    feature map that are most predictive of cancer risk. Uses both average
    and max pooling across channels to capture complementary spatial cues.

    Args:
        kernel_size: Size of the convolutional kernel for spatial attention.
            Larger kernels capture broader context but reduce spatial precision.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial attention and apply to input.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tuple of:
                - Attended features of shape (B, C, H, W)
                - Attention map of shape (B, 1, H, W) for visualization
        """
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)

        attention_map = self.sigmoid(self.conv(pooled))  # (B, 1, H, W)

        attended = x * attention_map
        return attended, attention_map


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention.

    Sequentially applies channel attention (what to focus on) followed by
    spatial attention (where to focus), providing a comprehensive attention
    mechanism that improves feature discrimination.

    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

    Args:
        num_channels: Number of input/output channels.
        reduction_ratio: Channel attention reduction ratio.
        spatial_kernel_size: Kernel size for spatial attention convolution.
    """

    def __init__(
        self,
        num_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CBAM attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tuple of (attended_features, spatial_attention_map).
        """
        x = self.channel_attention(x)
        x, attention_map = self.spatial_attention(x)
        return x, attention_map


class MultiScaleSpatialAttention(nn.Module):
    """Multi-scale spatial attention for capturing features at different resolutions.

    Mammographic findings span a wide range of scales -- from sub-millimeter
    microcalcifications to large-scale density patterns. This module applies
    spatial attention at multiple scales and fuses the results, enabling the
    model to attend to both fine-grained and coarse features simultaneously.

    Args:
        num_channels: Number of input channels.
        scales: Number of attention scales.
        reduction_ratio: Channel reduction ratio for each scale.
    """

    def __init__(
        self,
        num_channels: int,
        scales: int = 3,
        reduction_ratio: int = 16,
    ) -> None:
        super().__init__()
        self.scales = scales

        # Attention modules at each scale
        self.attention_modules = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()

        for i in range(scales):
            self.attention_modules.append(
                SpatialAttention(kernel_size=7)
            )
            if i > 0:
                self.downsample.append(
                    nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)
                )
                self.upsample_convs.append(
                    nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
                )

        # Fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(scales, scales, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(scales),
            nn.ReLU(inplace=True),
            nn.Conv2d(scales, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-scale attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tuple of (attended_features, fused_attention_map).
        """
        b, c, h, w = x.size()

        # Channel attention first
        x = self.channel_attention(x)

        attention_maps = []

        # Scale 0: original resolution
        _, att_0 = self.attention_modules[0](x)
        attention_maps.append(att_0)

        # Coarser scales: downsample -> attend -> upsample
        for i in range(1, self.scales):
            x_down = self.downsample[i - 1](x)
            _, att_i = self.attention_modules[i](x_down)
            att_i = F.interpolate(
                att_i, size=(h, w), mode="bilinear", align_corners=False
            )
            att_i = self.upsample_convs[i - 1](att_i)
            attention_maps.append(att_i)

        # Fuse multi-scale attention maps
        stacked = torch.cat(attention_maps, dim=1)  # (B, scales, H, W)
        fused_attention = self.fusion_conv(stacked)  # (B, 1, H, W)

        attended = x * fused_attention
        return attended, fused_attention


class GatedAttention(nn.Module):
    """Gated attention mechanism for mammography feature selection.

    Uses a gating mechanism to learn which spatial locations contain
    relevant information versus background/noise. The gating signal is
    learned jointly with the attention weights, providing a more expressive
    attention mechanism than simple softmax-based approaches.

    Particularly effective for mammography where large portions of the image
    (air background, pectoral muscle) are uninformative and should be
    suppressed aggressively.

    Args:
        in_channels: Number of input feature channels.
        attention_dim: Dimensionality of the attention embedding space.
    """

    def __init__(self, in_channels: int, attention_dim: int = 128) -> None:
        super().__init__()

        self.attention_fc = nn.Sequential(
            nn.Conv2d(in_channels, attention_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(attention_dim),
            nn.Tanh(),
        )

        self.gate_fc = nn.Sequential(
            nn.Conv2d(in_channels, attention_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(attention_dim),
            nn.Sigmoid(),
        )

        self.attention_weights = nn.Conv2d(attention_dim, 1, kernel_size=1, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply gated attention.

        Args:
            x: Input features of shape (B, C, H, W).

        Returns:
            Tuple of:
                - Weighted feature vector of shape (B, C) (global representation)
                - Attention map of shape (B, 1, H, W)
        """
        attention = self.attention_fc(x)   # (B, D, H, W)
        gate = self.gate_fc(x)             # (B, D, H, W)

        gated = attention * gate           # (B, D, H, W)
        scores = self.attention_weights(gated)  # (B, 1, H, W)

        # Normalize attention across spatial dimensions
        b, _, h, w = scores.size()
        scores_flat = scores.view(b, 1, -1)  # (B, 1, H*W)
        attention_weights = F.softmax(scores_flat, dim=-1)
        attention_map = attention_weights.view(b, 1, h, w)

        # Weighted spatial aggregation
        x_flat = x.view(b, x.size(1), -1)    # (B, C, H*W)
        weighted = torch.bmm(
            x_flat, attention_weights.squeeze(1).transpose(1, 2)
        )  # (B, C, 1)
        weighted = weighted.squeeze(-1)  # (B, C)

        return weighted, attention_map
