#!/usr/bin/env python3
"""Generate clinical-style PDF risk assessment report for a patient.

Produces a professional report containing:
- Patient information and exam metadata
- Risk prediction score with clinical interpretation
- Attention heatmap visualization showing regions of concern
- Density classification result
- Recommendation based on risk stratification

Usage:
    python scripts/generate_report.py \
        --checkpoint experiments/run_001/best_model.pt \
        --dicom-dir data/raw/patient_042/ \
        --output report_patient_042.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dicom_loader import DicomLoader, load_mammogram_study
from src.models.risk_model import MammographyRiskModel
from src.preprocessing.mammogram_preprocessor import MammogramPreprocessor

logger = logging.getLogger(__name__)

# Risk stratification thresholds
RISK_THRESHOLDS = {
    "low": (0.0, 0.05),
    "average": (0.05, 0.10),
    "intermediate": (0.10, 0.20),
    "high": (0.20, 1.0),
}

DENSITY_LABELS = {
    0: "A - Almost entirely fatty",
    1: "B - Scattered fibroglandular densities",
    2: "C - Heterogeneously dense",
    3: "D - Extremely dense",
}


def classify_risk(score: float) -> tuple[str, str, str]:
    """Classify risk score into clinical category.

    Returns:
        Tuple of (category, color, recommendation).
    """
    for category, (low, high) in RISK_THRESHOLDS.items():
        if low <= score < high:
            break

    recommendations = {
        "low": "Routine screening per standard guidelines. No additional imaging recommended.",
        "average": "Routine screening. Consider discussing risk factors with patient.",
        "intermediate": "Consider supplemental screening with breast MRI or ultrasound. "
                       "Referral to high-risk clinic may be appropriate.",
        "high": "Recommend supplemental screening with contrast-enhanced breast MRI. "
               "Referral to high-risk clinic for risk management counseling.",
    }

    colors = {
        "low": "#4CAF50",
        "average": "#8BC34A",
        "intermediate": "#FF9800",
        "high": "#F44336",
    }

    return category.upper(), colors[category], recommendations[category]


def generate_pdf_report(
    views: dict,
    model: MammographyRiskModel,
    preprocessor: MammogramPreprocessor,
    output_path: str | Path,
    device: torch.device,
) -> None:
    """Generate a multi-page PDF clinical report.

    Args:
        views: Dictionary mapping view names to (image, metadata) tuples.
        model: Trained risk prediction model.
        preprocessor: Preprocessing pipeline.
        output_path: Path for the output PDF.
        device: Inference device.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get patient info from first available view
    first_meta = list(views.values())[0][1]
    patient_id = first_meta.patient_id

    # Process all views and get predictions
    view_results = {}
    for view_name, (image, meta) in views.items():
        is_mlo = "MLO" in view_name.upper()
        tensor, mask = preprocessor.process(image, is_mlo=is_mlo, is_left=True)
        tensor = tensor.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(tensor)

        risk_prob = outputs["risk_prob"].item()
        density_probs = outputs["density_probs"].squeeze().cpu().numpy()
        density_class = int(density_probs.argmax())

        attention_map = None
        if "attention_map" in outputs:
            att = outputs["attention_map"]
            attention_map = torch.nn.functional.interpolate(
                att, size=image.shape[:2], mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()

        view_results[view_name] = {
            "risk_prob": risk_prob,
            "density_class": density_class,
            "density_probs": density_probs,
            "attention_map": attention_map,
            "image": image,
        }

    # Aggregate risk across views (max pooling)
    overall_risk = max(r["risk_prob"] for r in view_results.values())
    dominant_density = max(
        view_results.values(), key=lambda r: r["density_probs"].max()
    )["density_class"]

    category, color, recommendation = classify_risk(overall_risk)

    # Generate PDF
    risk_cmap = LinearSegmentedColormap.from_list("risk", [
        (0, 0, 0, 0), (1, 1, 0, 0.3), (1, 0.5, 0, 0.6), (1, 0, 0, 0.8)
    ])

    with PdfPages(str(output_path)) as pdf:
        # Page 1: Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        # Header
        ax.text(0.5, 0.95, "BREAST CANCER RISK ASSESSMENT REPORT",
                ha="center", va="top", fontsize=18, fontweight="bold",
                color="#1565C0")
        ax.text(0.5, 0.92, "AI-Assisted Mammographic Risk Prediction",
                ha="center", va="top", fontsize=12, color="#666666")

        # Divider
        ax.axhline(y=0.90, xmin=0.1, xmax=0.9, color="#1565C0", linewidth=2)

        # Patient info
        y = 0.86
        info_items = [
            ("Patient ID:", patient_id),
            ("Exam Date:", datetime.now().strftime("%Y-%m-%d")),
            ("Institution:", first_meta.institution),
            ("Views Analyzed:", ", ".join(views.keys())),
            ("Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for label, value in info_items:
            ax.text(0.12, y, label, fontsize=11, fontweight="bold")
            ax.text(0.40, y, str(value), fontsize=11)
            y -= 0.03

        # Risk score (large)
        y -= 0.04
        ax.text(0.5, y, "5-Year Breast Cancer Risk", ha="center",
                fontsize=16, fontweight="bold")
        y -= 0.08
        ax.text(0.5, y, f"{overall_risk * 100:.1f}%", ha="center",
                fontsize=48, fontweight="bold", color=color)
        y -= 0.04
        ax.text(0.5, y, f"Risk Category: {category}", ha="center",
                fontsize=16, fontweight="bold", color=color)

        # Density
        y -= 0.06
        ax.text(0.5, y, f"BI-RADS Density: {DENSITY_LABELS[dominant_density]}",
                ha="center", fontsize=12)

        # Recommendation
        y -= 0.06
        ax.text(0.5, y, "Clinical Recommendation", ha="center",
                fontsize=14, fontweight="bold", color="#1565C0")
        y -= 0.03
        # Wrap long text
        import textwrap
        wrapped = textwrap.fill(recommendation, width=80)
        for line in wrapped.split("\n"):
            ax.text(0.5, y, line, ha="center", fontsize=11)
            y -= 0.025

        # Disclaimer
        ax.text(0.5, 0.05,
                "DISCLAIMER: This report is generated by an AI system for research "
                "purposes only.\nIt has not been cleared by the FDA and should not "
                "be used as the sole basis for clinical decisions.",
                ha="center", va="bottom", fontsize=8, color="#999999",
                style="italic")

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # Page 2: View analysis with heatmaps
        n_views = len(view_results)
        fig, axes = plt.subplots(2, max(n_views, 1), figsize=(8.5, 11))
        if n_views == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle("Mammographic Views with Risk Attention Maps",
                     fontsize=14, fontweight="bold", y=0.98)

        for idx, (view_name, result) in enumerate(view_results.items()):
            if idx >= axes.shape[1]:
                break

            # Original image
            axes[0, idx].imshow(result["image"], cmap="gray", aspect="auto")
            axes[0, idx].set_title(f"{view_name}\nRisk: {result['risk_prob']*100:.1f}%",
                                    fontsize=10)
            axes[0, idx].axis("off")

            # Heatmap overlay
            axes[1, idx].imshow(result["image"], cmap="gray", aspect="auto")
            if result["attention_map"] is not None:
                axes[1, idx].imshow(result["attention_map"], cmap=risk_cmap, aspect="auto")
            axes[1, idx].set_title(f"Attention Map", fontsize=10)
            axes[1, idx].axis("off")

        # Hide unused axes
        for idx in range(n_views, axes.shape[1]):
            axes[0, idx].axis("off")
            axes[1, idx].axis("off")

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    logger.info("Clinical report saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clinical risk assessment PDF report",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dicom-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="risk_report.pdf")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})

    model = MammographyRiskModel(
        backbone=model_cfg.get("backbone", "efficientnet_b5"),
        num_density_classes=model_cfg.get("num_density_classes", 4),
        dropout_rate=model_cfg.get("dropout_rate", 0.3),
        use_attention=model_cfg.get("use_attention", True),
        attention_type=model_cfg.get("attention_type", "cbam"),
        pretrained=False,
        input_channels=model_cfg.get("input_channels", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load mammogram study
    views = load_mammogram_study(args.dicom_dir)

    if not views:
        logger.error("No views loaded from %s", args.dicom_dir)
        sys.exit(1)

    # Preprocess and generate report
    preprocessor = MammogramPreprocessor()
    generate_pdf_report(views, model, preprocessor, args.output, device)

    logger.info("Report generation complete: %s", args.output)


if __name__ == "__main__":
    main()
