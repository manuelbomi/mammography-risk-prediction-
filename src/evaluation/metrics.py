"""Evaluation metrics for breast cancer risk prediction.

Implements clinically relevant metrics for assessing model performance:
- AUC with DeLong confidence intervals
- Sensitivity at fixed specificity thresholds (operating points relevant
  to screening mammography)
- Calibration metrics (ECE, reliability diagrams)
- Bootstrap confidence intervals for all metrics

These metrics are aligned with how breast cancer screening tools are
evaluated in the clinical literature, enabling direct comparison with
published results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_auc_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute AUC-ROC with bootstrap confidence interval.

    Uses the percentile bootstrap method to estimate confidence intervals,
    which is more robust than the DeLong method for small sample sizes and
    extreme class imbalance common in screening datasets.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted risk scores / probabilities.
        confidence_level: Confidence level for the interval (e.g., 0.95).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (auc_value, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)

    # Point estimate
    auc_value = roc_auc_score(y_true, y_score)

    # Bootstrap
    n = len(y_true)
    bootstrap_aucs = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        boot_true = y_true[indices]
        boot_score = y_score[indices]

        # Need both classes in bootstrap sample
        if len(np.unique(boot_true)) < 2:
            continue

        try:
            boot_auc = roc_auc_score(boot_true, boot_score)
            bootstrap_aucs.append(boot_auc)
        except ValueError:
            continue

    if len(bootstrap_aucs) < 100:
        logger.warning(
            "Only %d valid bootstrap samples (target: %d). "
            "CI may be unreliable.",
            len(bootstrap_aucs),
            n_bootstrap,
        )

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_aucs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))

    return float(auc_value), float(ci_lower), float(ci_upper)


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float,
) -> Tuple[float, float]:
    """Compute sensitivity (recall) at a given specificity threshold.

    This is the standard metric for evaluating screening tests: how many
    cancers do we detect while maintaining an acceptable false-positive rate?

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        target_specificity: Desired specificity (e.g., 0.90 for 90%).

    Returns:
        Tuple of (sensitivity, threshold) at the target specificity.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr

    # Find the threshold closest to target specificity (from above)
    valid = specificity >= target_specificity
    if not valid.any():
        return 0.0, float(thresholds[-1])

    # Among valid points, pick the one with highest sensitivity
    idx = np.where(valid)[0]
    best_idx = idx[np.argmax(tpr[idx])]

    return float(tpr[best_idx]), float(thresholds[min(best_idx, len(thresholds) - 1)])


def specificity_at_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float,
) -> Tuple[float, float]:
    """Compute specificity at a given sensitivity threshold.

    Complementary to sensitivity_at_specificity: how many unnecessary callbacks
    can we avoid while catching a target fraction of cancers?

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        target_sensitivity: Desired sensitivity (e.g., 0.90 for 90%).

    Returns:
        Tuple of (specificity, threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr

    valid = tpr >= target_sensitivity
    if not valid.any():
        return 0.0, float(thresholds[0])

    # Among valid points, pick the one with highest specificity
    idx = np.where(valid)[0]
    best_idx = idx[np.argmax(specificity[idx])]

    return float(specificity[best_idx]), float(
        thresholds[min(best_idx, len(thresholds) - 1)]
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE).

    ECE measures the average absolute gap between predicted probability
    and observed frequency across probability bins. Lower is better; a
    perfectly calibrated model has ECE = 0.

    For clinical use, well-calibrated probabilities are essential so that
    a predicted risk of 10% means approximately 10% of such patients
    actually develop cancer.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of equally spaced bins.

    Returns:
        Tuple of (ece, bin_accuracies, bin_confidences, bin_counts).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob >= lower) & (y_prob < upper)

        bin_counts[i] = in_bin.sum()
        if bin_counts[i] > 0:
            bin_accuracies[i] = y_true[in_bin].mean()
            bin_confidences[i] = y_prob[in_bin].mean()

    # Weighted average of |accuracy - confidence|
    total = bin_counts.sum()
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / max(total, 1)

    return float(ece), bin_accuracies, bin_confidences, bin_counts


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Maximum Calibration Error (MCE).

    The worst-case calibration error across all bins. Important for safety-
    critical applications where even one poorly calibrated bin could lead
    to harmful clinical decisions.
    """
    _, bin_accuracies, bin_confidences, bin_counts = expected_calibration_error(
        y_true, y_prob, n_bins
    )
    nonempty = bin_counts > 0
    if not nonempty.any():
        return 0.0
    return float(np.max(np.abs(bin_accuracies[nonempty] - bin_confidences[nonempty])))


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics.

    Generates a full set of metrics relevant for clinical evaluation of
    breast cancer risk prediction models.

    Args:
        y_true: Binary ground truth (1 = cancer).
        y_score: Predicted risk probabilities.
        n_bootstrap: Number of bootstrap iterations for CIs.
        seed: Random seed.

    Returns:
        Dictionary of metric name -> value.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    metrics: Dict[str, Any] = {}

    # AUC with confidence interval
    auc_val, auc_ci_lo, auc_ci_hi = compute_auc_with_ci(
        y_true, y_score, n_bootstrap=n_bootstrap, seed=seed
    )
    metrics["auc"] = auc_val
    metrics["auc_ci_lower"] = auc_ci_lo
    metrics["auc_ci_upper"] = auc_ci_hi

    # Average precision (precision-recall AUC)
    metrics["average_precision"] = float(average_precision_score(y_true, y_score))

    # Sensitivity at fixed specificity thresholds
    for spec_target in [0.80, 0.90, 0.95]:
        sens, thresh = sensitivity_at_specificity(y_true, y_score, spec_target)
        spec_pct = int(spec_target * 100)
        metrics[f"sensitivity_at_{spec_pct}_specificity"] = sens
        metrics[f"threshold_at_{spec_pct}_specificity"] = thresh

    # Specificity at fixed sensitivity thresholds
    for sens_target in [0.80, 0.90, 0.95]:
        spec, thresh = specificity_at_sensitivity(y_true, y_score, sens_target)
        sens_pct = int(sens_target * 100)
        metrics[f"specificity_at_{sens_pct}_sensitivity"] = spec

    # Calibration
    ece, bin_acc, bin_conf, bin_counts = expected_calibration_error(y_true, y_score)
    metrics["ece"] = ece
    metrics["mce"] = maximum_calibration_error(y_true, y_score)

    # Prevalence
    metrics["prevalence"] = float(y_true.mean())
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_total"] = len(y_true)

    logger.info(
        "Metrics computed: AUC=%.4f [%.4f, %.4f], ECE=%.4f, "
        "Sens@90Spec=%.4f, N=%d (pos=%d)",
        auc_val,
        auc_ci_lo,
        auc_ci_hi,
        ece,
        metrics["sensitivity_at_90_specificity"],
        len(y_true),
        int(y_true.sum()),
    )

    return metrics


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    **metric_kwargs,
) -> Tuple[float, float, float]:
    """Generic bootstrap confidence interval for any metric function.

    Args:
        y_true: Ground truth labels.
        y_score: Predicted scores.
        metric_fn: Function taking (y_true, y_score, **kwargs) returning float.
        n_bootstrap: Number of resamples.
        confidence_level: CI level.
        seed: Random seed.
        **metric_kwargs: Additional kwargs for metric_fn.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    point = metric_fn(y_true, y_score, **metric_kwargs)

    bootstrap_values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        bt, bs = y_true[idx], y_score[idx]
        if len(np.unique(bt)) < 2:
            continue
        try:
            val = metric_fn(bt, bs, **metric_kwargs)
            bootstrap_values.append(val)
        except Exception:
            continue

    alpha = 1 - confidence_level
    ci_lo = np.percentile(bootstrap_values, 100 * alpha / 2)
    ci_hi = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))

    return float(point), float(ci_lo), float(ci_hi)
