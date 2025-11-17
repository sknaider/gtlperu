"""
Universal Trainer for PyTorch and TensorFlow Models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from pathlib import Path
import mlflow
from tqdm import tqdm
from loguru import logger
import json
import time
from backend.core.config import settings


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Model
    model_name: str = "resnet50"
    num_classes: int = 10

    # Data
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8

    # Training
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: Optional[str] = "cosine"  # cosine, step, plateau, None
    gradient_clip: Optional[float] = None

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    num_gpus: int = 1

    # Checkpointing
    save_dir: str = "./models/checkpoints"
    save_every: int = 1
    save_best_only: bool = True

    # Logging
    log_every: int = 10
    mlflow_tracking: bool = True
    wandb_tracking: bool = False

    # Early stopping
    early_stopping: bool = False
    patience: int = 5

    # Advanced
    resume_from: Optional[str] = None
    seed: int = 42


class Trainer:
    """Universal trainer for deep learning models"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize trainer

        Args:
            model: PyTorch model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            metrics: Dictionary of metric functions
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metrics = metrics or {}

        # Setup
        self._setup_device()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_mixed_precision()
        self._setup_logging()

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.patience_counter = 0

        # Load checkpoint if resume_from is provided
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def _setup_device(self):
        """Setup device (CPU/GPU)"""
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        if self.config.num_gpus > 1 and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        logger.info(f"Using device: {self.device}")

    def _setup_optimizer(self):
        """Setup optimizer"""
        from .optimizers import get_optimizer

        self.optimizer = get_optimizer(
            self.model.parameters(),
            self.config.optimizer,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        from .schedulers import get_scheduler

        if self.config.scheduler:
            self.scheduler = get_scheduler(
                self.optimizer,
                self.config.scheduler,
                num_epochs=self.config.epochs,
            )
        else:
            self.scheduler = None

    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        self.scaler = None
        if self.config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")

    def _setup_logging(self):
        """Setup experiment tracking"""
        if self.config.mlflow_tracking:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            mlflow.start_run()

            # Log config
            mlflow.log_params(self.config.__dict__)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                self.optimizer.step()

            # Update metrics
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self.global_step += 1

            # Update progress bar
            avg_loss = total_loss / total_samples
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Log to MLflow
            if (
                self.config.mlflow_tracking
                and self.global_step % self.config.log_every == 0
            ):
                mlflow.log_metric("train_loss", loss.item(), step=self.global_step)

        return {"loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model

        Returns:
            Dictionary of validation metrics
        """
        if not self.val_loader:
            return {}

        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_outputs = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc="Validation")

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            # Update metrics
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

        # Calculate metrics
        avg_loss = total_loss / total_samples
        metrics = {"val_loss": avg_loss}

        # Calculate additional metrics
        if self.metrics:
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)

            for metric_name, metric_fn in self.metrics.items():
                value = metric_fn(all_outputs, all_targets)
                metrics[f"val_{metric_name}"] = value

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Main training loop

        Returns:
            Training history
        """
        logger.info("Starting training...")
        history = {"train": [], "val": []}
        start_time = time.time()

        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch

                # Train
                train_metrics = self.train_epoch()
                history["train"].append(train_metrics)

                logger.info(
                    f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}"
                )

                # Validate
                val_metrics = self.validate()
                if val_metrics:
                    history["val"].append(val_metrics)
                    logger.info(f"Epoch {epoch}: Val Loss = {val_metrics['val_loss']:.4f}")

                    # Log to MLflow
                    if self.config.mlflow_tracking:
                        for metric_name, value in val_metrics.items():
                            mlflow.log_metric(metric_name, value, step=epoch)

                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get("val_loss", train_metrics["loss"]))
                    else:
                        self.scheduler.step()

                # Save checkpoint
                if (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(
                        Path(self.config.save_dir) / f"checkpoint_epoch_{epoch}.pt"
                    )

                # Save best model
                current_metric = val_metrics.get("val_loss", train_metrics["loss"])
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0

                    if self.config.save_best_only:
                        self.save_checkpoint(
                            Path(self.config.save_dir) / "best_model.pt"
                        )
                        logger.info(f"Saved best model (metric: {self.best_metric:.4f})")
                else:
                    self.patience_counter += 1

                # Early stopping
                if (
                    self.config.early_stopping
                    and self.patience_counter >= self.config.patience
                ):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            # Training complete
            training_time = time.time() - start_time
            logger.info(f"Training complete in {training_time:.2f} seconds")

            # Save final model
            self.save_checkpoint(Path(self.config.save_dir) / "final_model.pt")

            return {
                "history": history,
                "training_time": training_time,
                "best_metric": self.best_metric,
            }

        finally:
            # Cleanup
            if self.config.mlflow_tracking:
                mlflow.end_run()

    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config.__dict__,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Checkpoint loaded from epoch {self.current_epoch - 1}")
