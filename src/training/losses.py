"""Custom loss functions for mammography risk prediction.

Implements loss functions tailored to the unique challenges of medical imaging:
- Extreme class imbalance (typically 3-7% positive rate in screening populations)
- Need for well-calibrated probability outputs for clinical decision support
- Multi-task learning with heterogeneous objectives (regression + classification)

References:
    - Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    - Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for handling extreme class imbalance in mammography screening.

    Focal loss down-weights the contribution of easy-to-classify examples,
    allowing the model to focus on the hard, ambiguous cases that are most
    informative for learning discriminative features. This is crucial for
    mammography where the vast majority of exams are normal.

    Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1 else (1-p)

    Args:
        alpha: Weighting factor for the positive class. Values > 0.5 increase
            the weight of positives (cancers). Set based on class prevalence.
        gamma: Focusing parameter. Higher values focus more aggressively on
            hard examples. gamma=0 recovers standard cross-entropy.
        reduction: Reduction mode ('mean', 'sum', or 'none').
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs (before sigmoid), shape (B,).
            targets: Binary labels (0 or 1), shape (B,).

        Returns:
            Scalar loss value (or per-sample if reduction='none').
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Binary cross-entropy (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Focal modulation
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CalibratedBCELoss(nn.Module):
    """Binary cross-entropy with calibration-aware regularization.

    Augments standard BCE with a regularization term that penalizes
    miscalibration, encouraging the model to produce probability estimates
    that match the true frequency of positive outcomes. This is critical
    for clinical deployment where physicians need to trust the output
    probabilities as genuine risk estimates.

    The calibration penalty uses a differentiable approximation to the
    Expected Calibration Error (ECE) that can be optimized via gradient descent.

    Args:
        num_bins: Number of bins for calibration estimation.
        calibration_weight: Weight of the calibration penalty term.
        pos_weight: Weight for the positive class in BCE.
    """

    def __init__(
        self,
        num_bins: int = 15,
        calibration_weight: float = 0.1,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.calibration_weight = calibration_weight
        self.pos_weight = pos_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute calibrated BCE loss.

        Args:
            logits: Raw model outputs (before sigmoid), shape (B,).
            targets: Binary labels (0 or 1), shape (B,).

        Returns:
            Scalar combined loss.
        """
        targets = targets.float()

        # Standard BCE
        if self.pos_weight is not None:
            pw = torch.tensor(
                [self.pos_weight], device=logits.device, dtype=logits.dtype
            )
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)

        # Differentiable calibration penalty
        calibration_penalty = self._soft_calibration_error(logits, targets)

        return bce + self.calibration_weight * calibration_penalty

    def _soft_calibration_error(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute a differentiable approximation to ECE.

        Uses soft binning with Gaussian kernels centered at bin midpoints
        to maintain gradient flow, unlike the hard-binned ECE which is
        piecewise constant and non-differentiable.
        """
        probs = torch.sigmoid(logits)
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, device=probs.device)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_width = 1.0 / self.num_bins

        # Soft assignment to bins (Gaussian kernel)
        sigma = bin_width * 0.5
        probs_expanded = probs.unsqueeze(-1)  # (B, num_bins)
        centers_expanded = bin_centers.unsqueeze(0)  # (1, num_bins)

        weights = torch.exp(
            -0.5 * ((probs_expanded - centers_expanded) / sigma) ** 2
        )
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted average of predictions and targets per bin
        bin_pred_mean = (weights * probs.unsqueeze(-1)).sum(dim=0) / (
            weights.sum(dim=0) + 1e-8
        )
        bin_target_mean = (weights * targets.unsqueeze(-1)).sum(dim=0) / (
            weights.sum(dim=0) + 1e-8
        )

        # Bin counts for weighted average
        bin_counts = weights.sum(dim=0)
        total = bin_counts.sum()

        ece = ((bin_pred_mean - bin_target_mean).abs() * bin_counts).sum() / (
            total + 1e-8
        )

        return ece


class MultiTaskLoss(nn.Module):
    """Combined loss for joint risk prediction and density classification.

    Balances the two task losses with configurable weights, optionally
    using learned uncertainty-based weighting (Kendall et al., 2018)
    to automatically balance the tasks.

    Args:
        risk_loss_fn: Loss function for cancer risk prediction.
        density_loss_fn: Loss function for density classification.
        risk_weight: Static weight for risk loss.
        density_weight: Static weight for density loss.
        use_uncertainty_weighting: If True, learn task weights via
            homoscedastic uncertainty.
    """

    def __init__(
        self,
        risk_loss_fn: nn.Module,
        density_loss_fn: nn.Module,
        risk_weight: float = 1.0,
        density_weight: float = 0.3,
        use_uncertainty_weighting: bool = False,
    ) -> None:
        super().__init__()
        self.risk_loss_fn = risk_loss_fn
        self.density_loss_fn = density_loss_fn
        self.risk_weight = risk_weight
        self.density_weight = density_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting

        if use_uncertainty_weighting:
            # Log-variance parameters for uncertainty weighting
            self.log_var_risk = nn.Parameter(torch.zeros(1))
            self.log_var_density = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        risk_logits: torch.Tensor,
        density_logits: torch.Tensor,
        risk_targets: torch.Tensor,
        density_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined multi-task loss.

        Args:
            risk_logits: Risk prediction logits, shape (B,).
            density_logits: Density classification logits, shape (B, 4).
            risk_targets: Binary risk labels, shape (B,).
            density_targets: Density class labels, shape (B,).

        Returns:
            Scalar combined loss.
        """
        # Filter out invalid density labels (may be -1 for unknown)
        valid_density = density_targets >= 0

        risk_loss = self.risk_loss_fn(risk_logits, risk_targets)

        if valid_density.any():
            density_loss = self.density_loss_fn(
                density_logits[valid_density], density_targets[valid_density]
            )
        else:
            density_loss = torch.tensor(0.0, device=risk_logits.device)

        if self.use_uncertainty_weighting:
            # Kendall et al. uncertainty weighting:
            # L = (1/2*sigma_1^2) * L_1 + (1/2*sigma_2^2) * L_2 + log(sigma_1) + log(sigma_2)
            precision_risk = torch.exp(-self.log_var_risk)
            precision_density = torch.exp(-self.log_var_density)

            total = (
                0.5 * precision_risk * risk_loss
                + self.log_var_risk
                + 0.5 * precision_density * density_loss
                + self.log_var_density
            )
        else:
            total = self.risk_weight * risk_loss + self.density_weight * density_loss

        return total


class LabelSmoothingBCE(nn.Module):
    """Binary cross-entropy with label smoothing.

    Applies label smoothing to binary targets, replacing hard 0/1 labels
    with soft labels [epsilon, 1-epsilon]. This acts as a regularizer and
    can improve calibration.

    Args:
        smoothing: Smoothing factor (0 = no smoothing, 0.1 = typical).
    """

    def __init__(self, smoothing: float = 0.05) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute smoothed BCE.

        Args:
            logits: Raw model outputs, shape (B,).
            targets: Binary labels, shape (B,).

        Returns:
            Scalar loss.
        """
        targets = targets.float()
        smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smoothed)
