"""Configuration management for mammography risk prediction pipeline.

Uses Python dataclasses for type-safe configuration with YAML serialization.
Supports loading from files, environment variable overrides, and nested
configuration hierarchies for model, training, and data settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    data_dir: str = "data/raw"
    labels_csv: str = "data/labels.csv"
    target_size: Tuple[int, int] = (2048, 1024)
    views: List[str] = field(default_factory=lambda: ["CC", "MLO"])
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    normalize_laterality: bool = True
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class AugmentationConfig:
    """Data augmentation configuration for training."""

    enabled: bool = True
    rotation_range: float = 5.0
    scale_range: Tuple[float, float] = (0.95, 1.05)
    translate_range: float = 0.02
    intensity_shift_range: float = 0.05
    gaussian_noise_std: float = 0.01
    horizontal_flip_prob: float = 0.0
    random_crop_pad: int = 20


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    backbone: str = "efficientnet_b5"
    num_density_classes: int = 4
    dropout_rate: float = 0.3
    use_attention: bool = True
    attention_type: str = "cbam"  # "cbam", "multi_scale", "gated"
    pretrained: bool = True
    freeze_backbone_epochs: int = 3
    input_channels: int = 1


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Training hyperparameters
    num_epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    backbone_lr_factor: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Learning rate schedule
    warmup_epochs: int = 5
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau"
    lr_step_size: int = 15
    lr_gamma: float = 0.1

    # Loss
    loss_type: str = "focal"  # "focal", "bce", "calibrated_bce"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    risk_loss_weight: float = 1.0
    density_loss_weight: float = 0.3
    label_smoothing: float = 0.1
    use_uncertainty_weighting: bool = False

    # Regularization
    use_amp: bool = True
    early_stopping_patience: int = 10

    # Checkpointing
    checkpoint_interval: int = 10
    output_dir: str = "experiments/default"

    # Reproducibility
    seed: int = 42

    # Hardware
    num_workers: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load configuration from a YAML file.

        Supports nested configuration and flattens sub-configs appropriately.
        Environment variables can override any setting using the prefix
        MAMMO_RISK_ (e.g., MAMMO_RISK_LEARNING_RATE=0.001).

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Populated TrainingConfig instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Build sub-configs
        data_cfg = DataConfig(**raw.get("data", {}))
        aug_cfg = AugmentationConfig(**raw.get("augmentation", {}))
        model_cfg = ModelConfig(**raw.get("model", {}))

        # Build training config from top-level + training section
        training_raw = raw.get("training", {})
        top_level = {
            k: v for k, v in raw.items()
            if k not in ("data", "augmentation", "model", "training")
        }
        merged = {**top_level, **training_raw}

        # Filter to only valid TrainingConfig fields
        valid_fields = {f.name for f in fields(cls)} - {"data", "augmentation", "model"}
        filtered = {k: v for k, v in merged.items() if k in valid_fields}

        config = cls(
            data=data_cfg,
            augmentation=aug_cfg,
            model=model_cfg,
            **filtered,
        )

        # Apply environment variable overrides
        config = cls._apply_env_overrides(config)

        return config

    @staticmethod
    def _apply_env_overrides(config: "TrainingConfig") -> "TrainingConfig":
        """Override config values with environment variables.

        Environment variables should be prefixed with MAMMO_RISK_ and use
        uppercase with underscores. For example:
            MAMMO_RISK_LEARNING_RATE=0.001
            MAMMO_RISK_BATCH_SIZE=16
        """
        prefix = "MAMMO_RISK_"

        for f in fields(config):
            if f.name in ("data", "augmentation", "model"):
                continue

            env_key = prefix + f.name.upper()
            env_val = os.environ.get(env_key)

            if env_val is not None:
                try:
                    if f.type in (int, "int"):
                        setattr(config, f.name, int(env_val))
                    elif f.type in (float, "float"):
                        setattr(config, f.name, float(env_val))
                    elif f.type in (bool, "bool"):
                        setattr(config, f.name, env_val.lower() in ("true", "1", "yes"))
                    else:
                        setattr(config, f.name, env_val)
                except (ValueError, TypeError):
                    pass

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "data": asdict(self.data),
            "augmentation": asdict(self.augmentation),
            "model": asdict(self.model),
            "training": {
                f.name: getattr(self, f.name)
                for f in fields(self)
                if f.name not in ("data", "augmentation", "model")
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for logging."""
        return asdict(self)
