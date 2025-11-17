"""
Optimizer Factory for Training
"""

import torch
from torch.optim import Optimizer
from typing import Iterator
from torch.nn.parameter import Parameter


def get_optimizer(
    parameters: Iterator[Parameter],
    optimizer_name: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs,
) -> Optimizer:
    """
    Get optimizer by name

    Args:
        parameters: Model parameters
        optimizer_name: Optimizer name
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )

    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )

    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
            nesterov=kwargs.get("nesterov", False),
        )

    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            alpha=kwargs.get("alpha", 0.99),
            eps=kwargs.get("eps", 1e-8),
        )

    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            lr_decay=kwargs.get("lr_decay", 0),
        )

    elif optimizer_name == "adadelta":
        return torch.optim.Adadelta(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            rho=kwargs.get("rho", 0.9),
        )

    elif optimizer_name == "lion":
        try:
            from lion_pytorch import Lion

            return Lion(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get("betas", (0.9, 0.99)),
            )
        except ImportError:
            raise ImportError(
                "Lion optimizer requires lion-pytorch: pip install lion-pytorch"
            )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
