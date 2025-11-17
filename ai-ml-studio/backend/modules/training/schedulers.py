"""
Learning Rate Scheduler Factory
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import Optional


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    num_epochs: int = 100,
    **kwargs,
) -> Optional[_LRScheduler]:
    """
    Get learning rate scheduler by name

    Args:
        optimizer: Optimizer instance
        scheduler_name: Scheduler name
        num_epochs: Total number of epochs
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=kwargs.get("eta_min", 0),
        )

    elif scheduler_name == "cosine_warmup":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", 10),
            T_mult=kwargs.get("T_mult", 2),
            eta_min=kwargs.get("eta_min", 0),
        )

    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )

    elif scheduler_name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get("milestones", [30, 60, 90]),
            gamma=kwargs.get("gamma", 0.1),
        )

    elif scheduler_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get("gamma", 0.95),
        )

    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 10),
            verbose=True,
        )

    elif scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 0.1),
            total_steps=num_epochs,
            pct_start=kwargs.get("pct_start", 0.3),
        )

    elif scheduler_name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get("start_factor", 1.0),
            end_factor=kwargs.get("end_factor", 0.0),
            total_iters=num_epochs,
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
