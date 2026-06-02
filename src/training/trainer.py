"""Training engine for mammography risk prediction.

Implements a production-grade training loop with mixed-precision training,
learning rate scheduling with warmup, early stopping, comprehensive metric
tracking, and checkpoint management. Designed for the unique challenges of
medical imaging -- extreme class imbalance, calibration requirements, and
the need for reproducibility in clinical research.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics
from src.training.losses import CalibratedBCELoss, FocalLoss, MultiTaskLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to terminate training when validation metric plateaus.

    Args:
        patience: Number of epochs with no improvement to wait.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'max' for metrics where higher is better (AUC), 'min' for loss.
    """

    def __init__(
        self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value to monitor.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            (value > self.best_value + self.min_delta)
            if self.mode == "max"
            else (value < self.best_value - self.min_delta)
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.patience,
                )

        return self.should_stop


class Trainer:
    """Training engine for the mammography risk prediction model.

    Handles the complete training lifecycle including:
    - Mixed precision training with automatic loss scaling
    - Cosine annealing LR schedule with linear warmup
    - Gradient accumulation for effective large batch training
    - Multi-task loss balancing (risk + density)
    - Comprehensive metric logging per epoch
    - Checkpoint saving (best + periodic)
    - Early stopping based on validation AUC

    Args:
        model: The MammographyRiskModel to train.
        config: Training configuration object.
        output_dir: Directory for checkpoints, logs, and metrics.
        device: Device to train on. Auto-detected if None.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        output_dir: str | Path = "experiments/default",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Loss function
        self.criterion = MultiTaskLoss(
            risk_loss_fn=FocalLoss(
                alpha=getattr(config, "focal_alpha", 0.75),
                gamma=getattr(config, "focal_gamma", 2.0),
            ),
            density_loss_fn=nn.CrossEntropyLoss(
                label_smoothing=getattr(config, "label_smoothing", 0.1)
            ),
            risk_weight=getattr(config, "risk_loss_weight", 1.0),
            density_weight=getattr(config, "density_loss_weight", 0.3),
        )

        # Mixed precision
        self.use_amp = getattr(config, "use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient accumulation
        self.grad_accum_steps = getattr(config, "gradient_accumulation_steps", 1)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=getattr(config, "early_stopping_patience", 10),
            min_delta=1e-4,
            mode="max",
        )

        # Tracking
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_sensitivity_at_90spec": [],
            "learning_rate": [],
        }
        self.best_val_auc = 0.0
        self.global_step = 0

        logger.info(
            "Trainer initialized: device=%s, AMP=%s, grad_accum=%d",
            self.device,
            self.use_amp,
            self.grad_accum_steps,
        )

    def _build_optimizer(self) -> AdamW:
        """Build optimizer with layer-wise learning rate decay for backbone."""
        lr = getattr(self.config, "learning_rate", 1e-4)
        weight_decay = getattr(self.config, "weight_decay", 1e-4)
        backbone_lr_factor = getattr(self.config, "backbone_lr_factor", 0.1)

        backbone_params = list(self.model.backbone.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        head_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]

        param_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_factor},
            {"params": head_params, "lr": lr},
        ]

        return AdamW(param_groups, weight_decay=weight_decay)

    def _build_scheduler(
        self, num_training_steps: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Build LR scheduler with linear warmup + cosine annealing."""
        warmup_steps = getattr(self.config, "warmup_epochs", 5)
        total_epochs = getattr(self.config, "num_epochs", 50)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_steps,
            eta_min=1e-7,
        )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.

        Returns:
            Training history dictionary.
        """
        batch_size = getattr(self.config, "batch_size", 8)
        num_workers = getattr(self.config, "num_workers", 4)
        num_epochs = getattr(self.config, "num_epochs", 50)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        scheduler = self._build_scheduler(len(train_loader) * num_epochs)

        logger.info(
            "Starting training: %d epochs, %d train batches, %d val batches",
            num_epochs,
            len(train_loader),
            len(val_loader),
        )

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Update backbone freeze schedule
            if hasattr(self.model, "set_epoch"):
                self.model.set_epoch(epoch)

            # Training
            train_loss = self._train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)

            # Validation
            val_metrics = self._validate(val_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auc"].append(val_metrics["auc"])
            self.history["val_sensitivity_at_90spec"].append(
                val_metrics.get("sensitivity_at_90_specificity", 0.0)
            )
            self.history["learning_rate"].append(
                self.optimizer.param_groups[-1]["lr"]
            )

            # Step scheduler
            scheduler.step()

            # Logging
            elapsed = time.time() - epoch_start
            logger.info(
                "Epoch %d/%d [%.1fs] -- train_loss=%.4f, val_loss=%.4f, "
                "val_auc=%.4f, val_sens@90spec=%.4f, lr=%.2e",
                epoch + 1,
                num_epochs,
                elapsed,
                train_loss,
                val_metrics["loss"],
                val_metrics["auc"],
                val_metrics.get("sensitivity_at_90_specificity", 0.0),
                self.optimizer.param_groups[-1]["lr"],
            )

            # Checkpoint
            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
                self._save_checkpoint(epoch, val_metrics, is_best=True)

            if (epoch + 1) % getattr(self.config, "checkpoint_interval", 10) == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            # Early stopping
            if self.early_stopping.step(val_metrics["auc"]):
                logger.info("Training stopped early at epoch %d.", epoch + 1)
                break

        # Save final training history
        self._save_history()

        return self.history

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one training epoch.

        Args:
            loader: Training DataLoader.
            epoch: Current epoch number.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            risk_labels = batch["risk_label"].to(self.device, non_blocking=True)
            density_labels = batch["density_label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(
                    risk_logits=outputs["risk_logit"].squeeze(-1),
                    density_logits=outputs["density_logits"],
                    risk_targets=risk_labels,
                    density_targets=density_labels,
                )
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=getattr(self.config, "max_grad_norm", 1.0),
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """Run validation and compute metrics.

        Args:
            loader: Validation DataLoader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_risk_probs = []
        all_risk_labels = []
        all_density_logits = []
        all_density_labels = []

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            risk_labels = batch["risk_label"].to(self.device, non_blocking=True)
            density_labels = batch["density_label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(
                    risk_logits=outputs["risk_logit"].squeeze(-1),
                    density_logits=outputs["density_logits"],
                    risk_targets=risk_labels,
                    density_targets=density_labels,
                )

            total_loss += loss.item()
            num_batches += 1

            all_risk_probs.append(outputs["risk_prob"].squeeze(-1).cpu().numpy())
            all_risk_labels.append(risk_labels.cpu().numpy())
            all_density_logits.append(outputs["density_logits"].cpu().numpy())
            all_density_labels.append(density_labels.cpu().numpy())

        risk_probs = np.concatenate(all_risk_probs)
        risk_labels = np.concatenate(all_risk_labels)

        # Compute comprehensive metrics
        metrics = compute_all_metrics(risk_labels, risk_probs)
        metrics["loss"] = total_loss / max(num_batches, 1)

        return metrics

    def _save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "best_val_auc": self.best_val_auc,
            "global_step": self.global_step,
            "config": vars(self.config) if hasattr(self.config, "__dict__") else {},
        }

        if is_best:
            path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, path)
            logger.info("Saved best model (AUC=%.4f) to %s", metrics["auc"], path)
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            torch.save(checkpoint, path)

    def _save_history(self) -> None:
        """Save training history to JSON file."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training history saved to %s", history_path)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: nn.Module,
        config: Any,
        device: Optional[torch.device] = None,
    ) -> Tuple["Trainer", Dict[str, Any]]:
        """Load trainer state from a checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            model: Model instance (architecture must match checkpoint).
            config: Training config object.
            device: Device to load onto.

        Returns:
            Tuple of (trainer_instance, checkpoint_metadata).
        """
        checkpoint_path = Path(checkpoint_path)
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        trainer = cls(model=model, config=config, device=device)
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        trainer.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        trainer.global_step = checkpoint.get("global_step", 0)

        logger.info(
            "Loaded checkpoint from %s (epoch %d, AUC=%.4f)",
            checkpoint_path,
            checkpoint["epoch"],
            checkpoint["metrics"].get("auc", 0.0),
        )

        return trainer, checkpoint
