"""Mammography risk prediction model architecture.

Implements a multi-task deep learning model for simultaneous breast cancer risk
prediction and BI-RADS density classification. The model uses a pretrained
EfficientNet or ResNet backbone with a custom attention-augmented head that
learns to focus on diagnostically relevant breast tissue regions.

Architecture overview:
    1. Pretrained backbone (EfficientNet-B5 or ResNet-50) for feature extraction
    2. Spatial attention module for ROI focusing
    3. Global feature aggregation via attention-weighted pooling
    4. Multi-task heads:
       - Risk prediction: sigmoid output for cancer probability
       - Density classification: 4-class softmax for BI-RADS density (A-D)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.attention import CBAM, GatedAttention, MultiScaleSpatialAttention

logger = logging.getLogger(__name__)


class MammographyRiskModel(nn.Module):
    """Multi-task model for breast cancer risk prediction from mammograms.

    Combines a pretrained CNN backbone with spatial attention and multi-task
    prediction heads for joint risk scoring and density classification.

    Args:
        backbone: Name of the pretrained backbone ('efficientnet_b5', 'resnet50',
            'efficientnet_b3').
        num_density_classes: Number of density categories (default 4 for BI-RADS).
        dropout_rate: Dropout rate applied before prediction heads.
        use_attention: Whether to use spatial attention mechanism.
        attention_type: Type of attention ('cbam', 'multi_scale', 'gated').
        pretrained: Whether to load ImageNet pretrained weights.
        freeze_backbone_epochs: Number of initial epochs to freeze backbone
            weights (for gradual unfreezing during fine-tuning).
        input_channels: Number of input channels (1 for grayscale mammograms).
    """

    SUPPORTED_BACKBONES = {"efficientnet_b5", "efficientnet_b3", "resnet50", "resnet101"}

    def __init__(
        self,
        backbone: str = "efficientnet_b5",
        num_density_classes: int = 4,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        attention_type: str = "cbam",
        pretrained: bool = True,
        freeze_backbone_epochs: int = 0,
        input_channels: int = 1,
    ) -> None:
        super().__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from: {self.SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self._current_epoch = 0

        # Build backbone
        self.backbone, feature_dim = self._build_backbone(
            backbone, pretrained, input_channels
        )

        # Attention module
        if use_attention:
            self.attention = self._build_attention(feature_dim, attention_type)
        else:
            self.attention = None

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Shared feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
        )

        # Risk prediction head (binary: cancer within 5 years)
        self.risk_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1),
        )

        # Density classification head (4-class: BI-RADS A/B/C/D)
        self.density_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_density_classes),
        )

        # Initialize weights for new layers
        self._initialize_heads()

        logger.info(
            "MammographyRiskModel initialized: backbone=%s, attention=%s(%s), "
            "params=%.2fM",
            backbone,
            use_attention,
            attention_type if use_attention else "none",
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def _build_backbone(
        self, name: str, pretrained: bool, input_channels: int
    ) -> Tuple[nn.Module, int]:
        """Construct the feature extraction backbone.

        Handles conversion from 3-channel pretrained weights to single-channel
        input by averaging the first convolutional layer's weights across the
        channel dimension.
        """
        if name == "efficientnet_b5":
            weights = models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b5(weights=weights)
            feature_dim = base.classifier[1].in_features

            # Adapt first conv for single-channel input
            if input_channels != 3:
                old_conv = base.features[0][0]
                new_conv = nn.Conv2d(
                    input_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )
                if pretrained:
                    # Average across RGB channels
                    new_conv.weight.data = old_conv.weight.data.mean(
                        dim=1, keepdim=True
                    ).repeat(1, input_channels, 1, 1)
                base.features[0][0] = new_conv

            # Remove original classifier
            base.classifier = nn.Identity()
            base.avgpool = nn.Identity()
            backbone = base.features

        elif name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b3(weights=weights)
            feature_dim = base.classifier[1].in_features

            if input_channels != 3:
                old_conv = base.features[0][0]
                new_conv = nn.Conv2d(
                    input_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )
                if pretrained:
                    new_conv.weight.data = old_conv.weight.data.mean(
                        dim=1, keepdim=True
                    ).repeat(1, input_channels, 1, 1)
                base.features[0][0] = new_conv

            base.classifier = nn.Identity()
            base.avgpool = nn.Identity()
            backbone = base.features

        elif name in ("resnet50", "resnet101"):
            model_fn = models.resnet50 if name == "resnet50" else models.resnet101
            weights_cls = (
                models.ResNet50_Weights.DEFAULT
                if name == "resnet50"
                else models.ResNet101_Weights.DEFAULT
            )
            weights = weights_cls if pretrained else None
            base = model_fn(weights=weights)
            feature_dim = base.fc.in_features

            if input_channels != 3:
                old_conv = base.conv1
                new_conv = nn.Conv2d(
                    input_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )
                if pretrained:
                    new_conv.weight.data = old_conv.weight.data.mean(
                        dim=1, keepdim=True
                    ).repeat(1, input_channels, 1, 1)
                base.conv1 = new_conv

            # Extract feature layers (everything before avgpool and fc)
            backbone = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
        else:
            raise ValueError(f"Unknown backbone: {name}")

        return backbone, feature_dim

    def _build_attention(
        self, feature_dim: int, attention_type: str
    ) -> nn.Module:
        """Build the spatial attention module."""
        if attention_type == "cbam":
            return CBAM(feature_dim, reduction_ratio=16, spatial_kernel_size=7)
        elif attention_type == "multi_scale":
            return MultiScaleSpatialAttention(feature_dim, scales=3)
        elif attention_type == "gated":
            return GatedAttention(feature_dim, attention_dim=128)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def _initialize_heads(self) -> None:
        """Initialize prediction head weights with Kaiming initialization."""
        for module in [self.feature_proj, self.risk_head, self.density_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the full model.

        Args:
            x: Input tensor of shape (B, 1, H, W) -- single-channel mammograms.

        Returns:
            Dictionary containing:
                - 'risk_logit': Raw logit for cancer risk, shape (B, 1)
                - 'risk_prob': Sigmoid probability for cancer risk, shape (B, 1)
                - 'density_logits': Raw logits for density classes, shape (B, 4)
                - 'density_probs': Softmax probabilities, shape (B, 4)
                - 'attention_map': Spatial attention map, shape (B, 1, H', W')
                    (only present when use_attention=True)
                - 'features': Global feature vector, shape (B, 256)
        """
        # Extract backbone features
        features = self.backbone(x)  # (B, C, H', W')

        # Apply attention
        attention_map = None
        if self.attention is not None:
            if self.attention_type == "gated":
                # Gated attention returns pooled features directly
                pooled, attention_map = self.attention(features)
            else:
                features, attention_map = self.attention(features)
                pooled = self.global_pool(features).flatten(1)
        else:
            pooled = self.global_pool(features).flatten(1)

        # Shared feature projection
        shared_features = self.feature_proj(pooled)  # (B, 256)

        # Risk prediction
        risk_logit = self.risk_head(shared_features)  # (B, 1)
        risk_prob = torch.sigmoid(risk_logit)

        # Density classification
        density_logits = self.density_head(shared_features)  # (B, 4)
        density_probs = F.softmax(density_logits, dim=1)

        output = {
            "risk_logit": risk_logit,
            "risk_prob": risk_prob,
            "density_logits": density_logits,
            "density_probs": density_probs,
            "features": shared_features,
        }

        if attention_map is not None:
            output["attention_map"] = attention_map

        return output

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for backbone freezing schedule.

        If freeze_backbone_epochs > 0, the backbone is frozen for the first
        N epochs to allow the randomly initialized heads to converge before
        fine-tuning the pretrained backbone.
        """
        self._current_epoch = epoch

        if epoch < self.freeze_backbone_epochs:
            self._freeze_backbone()
        elif epoch == self.freeze_backbone_epochs:
            self._unfreeze_backbone()
            logger.info("Backbone unfrozen at epoch %d.", epoch)

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_attention_maps(
        self, x: torch.Tensor, upsample: bool = True
    ) -> Optional[torch.Tensor]:
        """Extract attention maps for visualization.

        Args:
            x: Input tensor of shape (B, 1, H, W).
            upsample: If True, upsample attention maps to input resolution.

        Returns:
            Attention maps of shape (B, 1, H, W) if use_attention=True,
            else None.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

        attention_map = output.get("attention_map")
        if attention_map is not None and upsample:
            attention_map = F.interpolate(
                attention_map,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return attention_map

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
