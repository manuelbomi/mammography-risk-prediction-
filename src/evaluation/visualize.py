"""Visualization utilities for mammography risk prediction evaluation.

Generates publication-quality plots for model evaluation:
- ROC curves with AUC annotation and confidence bands
- Calibration (reliability) diagrams
- Risk score distributions by outcome
- Attention heatmap overlays on mammograms
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve

from src.evaluation.metrics import (
    compute_auc_with_ci,
    expected_calibration_error,
    sensitivity_at_specificity,
)

logger = logging.getLogger(__name__)

# Publication-quality matplotlib configuration
PLOT_STYLE = {
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "figure.figsize": (8, 6),
}


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 500,
    title: str = "ROC Curve -- Mammography Risk Prediction",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot ROC curve with AUC, confidence band, and clinical operating points.

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        n_bootstrap: Bootstrap samples for confidence band.
        title: Plot title.
        save_path: If provided, save figure to this path.
        show: Whether to display the plot.

    Returns:
        Matplotlib Figure object.
    """
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Main ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val, ci_lo, ci_hi = compute_auc_with_ci(
            y_true, y_score, n_bootstrap=n_bootstrap
        )

        ax.plot(
            fpr, tpr,
            color="#2196F3",
            linewidth=2.5,
            label=f"Model (AUC = {auc_val:.3f} [{ci_lo:.3f}, {ci_hi:.3f}])",
        )

        # Bootstrap confidence band
        rng = np.random.RandomState(42)
        interp_fpr = np.linspace(0, 1, 200)
        boot_tprs = []

        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(y_true), size=len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            b_fpr, b_tpr, _ = roc_curve(y_true[idx], y_score[idx])
            boot_tprs.append(np.interp(interp_fpr, b_fpr, b_tpr))

        boot_tprs = np.array(boot_tprs)
        tpr_lower = np.percentile(boot_tprs, 2.5, axis=0)
        tpr_upper = np.percentile(boot_tprs, 97.5, axis=0)

        ax.fill_between(
            interp_fpr, tpr_lower, tpr_upper,
            alpha=0.15, color="#2196F3",
            label="95% CI",
        )

        # Clinical operating points
        for spec_target, marker, color in [
            (0.90, "o", "#E91E63"),
            (0.95, "s", "#FF9800"),
        ]:
            sens, _ = sensitivity_at_specificity(y_true, y_score, spec_target)
            ax.plot(
                1 - spec_target, sens,
                marker=marker, markersize=10, color=color,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5,
                label=f"Sens={sens:.3f} @ {int(spec_target*100)}% Spec",
            )

        # Diagonal reference
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")

        ax.set_xlabel("1 - Specificity (False Positive Rate)")
        ax.set_ylabel("Sensitivity (True Positive Rate)")
        ax.set_title(title)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("ROC curve saved to %s", save_path)

        if show:
            plt.show()

        return fig


def plot_calibration_diagram(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Diagram",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot calibration (reliability) diagram.

    Shows the relationship between predicted probability and observed
    frequency. A perfectly calibrated model follows the diagonal. Also
    shows a histogram of prediction counts per bin.

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        n_bins: Number of bins.
        title: Plot title.
        save_path: Save path.
        show: Whether to display.

    Returns:
        Figure.
    """
    with plt.rc_context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Calibration curve
        ece, bin_acc, bin_conf, bin_counts = expected_calibration_error(
            y_true, y_score, n_bins=n_bins
        )

        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)

        ax1.plot(
            prob_pred, prob_true,
            "o-", color="#2196F3", linewidth=2, markersize=8,
            label=f"Model (ECE = {ece:.4f})",
        )
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect")

        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Observed Frequency")
        ax1.set_title(title)
        ax1.legend(loc="upper left")
        ax1.set_xlim([-0.02, 1.02])
        ax1.set_ylim([-0.02, 1.02])
        ax1.grid(True, alpha=0.3)

        # Histogram of predictions
        ax2.hist(
            y_score, bins=n_bins, range=(0, 1),
            color="#90CAF9", edgecolor="#1565C0", alpha=0.8,
        )
        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Count")
        ax2.set_xlim([-0.02, 1.02])

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig


def plot_risk_heatmap(
    image: np.ndarray,
    attention_map: np.ndarray,
    risk_score: float,
    title: str = "Risk Heatmap",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Overlay attention heatmap on mammogram image.

    Creates a clinical visualization showing which regions of the mammogram
    contributed most to the risk prediction, helping radiologists understand
    and verify the model's reasoning.

    Args:
        image: Mammogram image, shape (H, W), values in [0, 1].
        attention_map: Attention weights, shape (H, W), values in [0, 1].
        risk_score: Predicted risk probability.
        title: Plot title.
        save_path: Save path.
        show: Whether to display.

    Returns:
        Figure.
    """
    # Custom colormap: transparent -> yellow -> red
    colors = [
        (0.0, 0.0, 0.0, 0.0),  # Transparent
        (1.0, 1.0, 0.0, 0.3),  # Yellow, semi-transparent
        (1.0, 0.5, 0.0, 0.6),  # Orange
        (1.0, 0.0, 0.0, 0.8),  # Red
    ]
    risk_cmap = LinearSegmentedColormap.from_list("risk", colors)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(image, cmap="gray", aspect="auto")
        axes[0].set_title("Original Mammogram")
        axes[0].axis("off")

        # Attention map
        axes[1].imshow(attention_map, cmap="hot", aspect="auto")
        axes[1].set_title("Attention Map")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(image, cmap="gray", aspect="auto")
        axes[2].imshow(attention_map, cmap=risk_cmap, aspect="auto")
        risk_pct = risk_score * 100
        color = "red" if risk_score > 0.15 else "orange" if risk_score > 0.075 else "green"
        axes[2].set_title(f"Risk: {risk_pct:.1f}%", color=color, fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(title, fontsize=16, fontweight="bold")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig


def plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Risk Score Distribution",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot distribution of risk scores for positive and negative cases.

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        title: Plot title.
        save_path: Save path.
        show: Whether to display.

    Returns:
        Figure.
    """
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        neg_scores = y_score[y_true == 0]
        pos_scores = y_score[y_true == 1]

        bins = np.linspace(0, 1, 50)
        ax.hist(
            neg_scores, bins=bins, alpha=0.6, color="#4CAF50",
            label=f"No cancer (n={len(neg_scores)})", density=True,
        )
        ax.hist(
            pos_scores, bins=bins, alpha=0.6, color="#F44336",
            label=f"Cancer (n={len(pos_scores)})", density=True,
        )

        ax.set_xlabel("Predicted Risk Score")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig
