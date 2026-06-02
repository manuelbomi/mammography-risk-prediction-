#!/usr/bin/env python3
"""Evaluation script for mammography risk prediction model.

Loads a trained checkpoint, runs inference on a test set, and generates
comprehensive evaluation metrics and visualizations.

Usage:
    python scripts/evaluate.py \
        --checkpoint experiments/run_001/best_model.pt \
        --data-dir data/raw \
        --labels-csv data/labels_test.csv \
        --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import MammographyDataset
from src.evaluation.metrics import (
    bootstrap_metric,
    compute_all_metrics,
    compute_auc_with_ci,
    expected_calibration_error,
    sensitivity_at_specificity,
)
from src.evaluation.visualize import (
    plot_calibration_diagram,
    plot_roc_curve,
    plot_score_distribution,
)
from src.models.risk_model import MammographyRiskModel
from src.utils.config import TrainingConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mammography risk prediction model",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument("--data-dir", type=str, required=True, help="Test data directory.")
    parser.add_argument("--labels-csv", type=str, required=True, help="Test labels CSV.")
    parser.add_argument("--output-dir", type=str, default="results/", help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap iterations.")
    return parser.parse_args()


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[MammographyRiskModel, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    model = MammographyRiskModel(
        backbone=config.get("model", {}).get("backbone", "efficientnet_b5"),
        num_density_classes=config.get("model", {}).get("num_density_classes", 4),
        dropout_rate=config.get("model", {}).get("dropout_rate", 0.3),
        use_attention=config.get("model", {}).get("use_attention", True),
        attention_type=config.get("model", {}).get("attention_type", "cbam"),
        pretrained=False,
        input_channels=config.get("model", {}).get("input_channels", 1),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        "Loaded model from %s (epoch %d, AUC=%.4f)",
        checkpoint_path,
        checkpoint.get("epoch", -1),
        checkpoint.get("metrics", {}).get("auc", 0.0),
    )

    return model, checkpoint


@torch.no_grad()
def run_inference(
    model: MammographyRiskModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run inference on the full dataset.

    Returns:
        Tuple of (risk_probs, risk_labels, density_labels, patient_ids).
    """
    all_probs, all_labels, all_density, all_ids = [], [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)

        all_probs.append(outputs["risk_prob"].squeeze(-1).cpu().numpy())
        all_labels.append(batch["risk_label"].numpy())
        all_density.append(batch["density_label"].numpy())
        all_ids.extend(batch["patient_id"])

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_density),
        all_ids,
    )


def generate_report(
    metrics: dict,
    output_dir: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    """Generate evaluation report with metrics and visualizations."""
    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    serializable = {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Generate plots
    plot_roc_curve(
        y_true, y_score,
        save_path=output_dir / "roc_curve.png",
    )

    plot_calibration_diagram(
        y_true, y_score,
        save_path=output_dir / "calibration_diagram.png",
    )

    plot_score_distribution(
        y_true, y_score,
        save_path=output_dir / "score_distribution.png",
    )

    # Generate text summary
    summary_lines = [
        "=" * 60,
        "EVALUATION RESULTS -- Mammography Risk Prediction",
        "=" * 60,
        "",
        f"Test set size: {metrics['n_total']} (positive: {metrics['n_positive']})",
        f"Prevalence: {metrics['prevalence']:.4f}",
        "",
        "--- Discrimination ---",
        f"AUC:                        {metrics['auc']:.4f} [{metrics['auc_ci_lower']:.4f}, {metrics['auc_ci_upper']:.4f}]",
        f"Average Precision:          {metrics['average_precision']:.4f}",
        "",
        "--- Operating Points ---",
        f"Sensitivity @ 80% Spec:     {metrics['sensitivity_at_80_specificity']:.4f}",
        f"Sensitivity @ 90% Spec:     {metrics['sensitivity_at_90_specificity']:.4f}",
        f"Sensitivity @ 95% Spec:     {metrics['sensitivity_at_95_specificity']:.4f}",
        f"Specificity @ 90% Sens:     {metrics['specificity_at_90_sensitivity']:.4f}",
        "",
        "--- Calibration ---",
        f"Expected Calibration Error: {metrics['ece']:.4f}",
        f"Maximum Calibration Error:  {metrics['mce']:.4f}",
        "",
        "=" * 60,
    ]

    summary = "\n".join(summary_lines)
    logger.info("\n%s", summary)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load model
    model, checkpoint = load_model(args.checkpoint, device)

    # Load test data
    test_dataset = MammographyDataset(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        split="test",
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info("Test dataset: %d samples", len(test_dataset))

    # Run inference
    risk_probs, risk_labels, density_labels, patient_ids = run_inference(
        model, test_loader, device
    )

    # Compute metrics
    metrics = compute_all_metrics(
        risk_labels, risk_probs, n_bootstrap=args.n_bootstrap
    )

    # Generate report
    generate_report(metrics, output_dir, risk_labels, risk_probs)

    # Save per-patient predictions
    predictions = {
        pid: {"risk_prob": float(prob), "label": int(label)}
        for pid, prob, label in zip(patient_ids, risk_probs, risk_labels)
    }
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info("Evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
