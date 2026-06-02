#!/usr/bin/env python3
"""Main training script for mammography risk prediction.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --data-dir /data/mammo --output-dir experiments/run_001
    python scripts/train.py --config configs/default.yaml --backbone resnet50 --lr 5e-5

Environment variable overrides:
    MAMMO_RISK_LEARNING_RATE=0.001 python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import MammographyDataset, create_data_loaders
from src.models.risk_model import MammographyRiskModel
from src.training.trainer import Trainer
from src.utils.config import TrainingConfig


def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    """Configure logging to both console and file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file)),
    ]

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train mammography risk prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )

    # Data overrides
    parser.add_argument("--data-dir", type=str, help="Override data directory.")
    parser.add_argument("--labels-csv", type=str, help="Override labels CSV path.")
    parser.add_argument("--output-dir", type=str, help="Override output directory.")

    # Model overrides
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["efficientnet_b5", "efficientnet_b3", "resnet50", "resnet101"],
        help="Override backbone architecture.",
    )
    parser.add_argument("--no-attention", action="store_true", help="Disable attention.")
    parser.add_argument(
        "--attention-type",
        type=str,
        choices=["cbam", "multi_scale", "gated"],
        help="Override attention type.",
    )

    # Training overrides
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")

    # Hardware
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu).")
    parser.add_argument("--num-workers", type=int, help="Override data loader workers.")

    # Misc
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from.")

    return parser.parse_args()


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply command-line argument overrides to config."""
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.labels_csv:
        config.data.labels_csv = args.labels_csv
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.backbone:
        config.model.backbone = args.backbone
    if args.no_attention:
        config.model.use_attention = False
    if args.attention_type:
        config.model.attention_type = args.attention_type
    if args.lr:
        config.learning_rate = args.lr
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_amp:
        config.use_amp = False
    if args.num_workers:
        config.num_workers = args.num_workers
        config.data.num_workers = args.num_workers
    if args.seed:
        config.seed = args.seed

    return config


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Load configuration
    config = TrainingConfig.from_yaml(args.config)
    config = apply_overrides(config, args)

    # Setup
    output_dir = Path(config.output_dir)
    setup_logging(output_dir, args.log_level)
    logger = logging.getLogger(__name__)

    set_seed(config.seed)

    # Log configuration
    logger.info("=" * 60)
    logger.info("Mammography Risk Prediction -- Training")
    logger.info("=" * 60)
    logger.info("Config file: %s", args.config)
    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Backbone: %s", config.model.backbone)
    logger.info("Attention: %s (%s)", config.model.use_attention, config.model.attention_type)
    logger.info("Batch size: %d (effective: %d)", config.batch_size, config.batch_size * config.gradient_accumulation_steps)
    logger.info("Learning rate: %e", config.learning_rate)
    logger.info("Epochs: %d", config.num_epochs)
    logger.info("AMP: %s", config.use_amp)

    # Save config snapshot
    config.to_yaml(output_dir / "config.yaml")

    # Build model
    model = MammographyRiskModel(
        backbone=config.model.backbone,
        num_density_classes=config.model.num_density_classes,
        dropout_rate=config.model.dropout_rate,
        use_attention=config.model.use_attention,
        attention_type=config.model.attention_type,
        pretrained=config.model.pretrained,
        freeze_backbone_epochs=config.model.freeze_backbone_epochs,
        input_channels=config.model.input_channels,
    )

    logger.info(
        "Model created: %.2fM parameters (%.2fM trainable)",
        model.count_parameters(trainable_only=False) / 1e6,
        model.count_parameters(trainable_only=True) / 1e6,
    )

    # Build datasets
    train_dataset = MammographyDataset(
        data_dir=config.data.data_dir,
        labels_csv=config.data.labels_csv,
        split="train",
        augment=config.augmentation.enabled,
        target_size=config.data.target_size,
        views=config.data.views,
    )

    val_dataset = MammographyDataset(
        data_dir=config.data.data_dir,
        labels_csv=config.data.labels_csv,
        split="val",
        augment=False,
        target_size=config.data.target_size,
        views=config.data.views,
    )

    logger.info(
        "Datasets loaded: train=%d, val=%d (positive rate: %.2f%%)",
        len(train_dataset),
        len(val_dataset),
        train_dataset.positive_rate * 100,
    )

    # Device
    device = torch.device(args.device) if args.device else None

    # Build trainer
    trainer = Trainer(
        model=model,
        config=config,
        output_dir=output_dir,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer, _ = Trainer.load_from_checkpoint(
            args.resume, model, config, device
        )

    # Train
    history = trainer.fit(train_dataset, val_dataset)

    # Summary
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Best validation AUC: %.4f", trainer.best_val_auc)
    logger.info("Best model saved to: %s", output_dir / "best_model.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
