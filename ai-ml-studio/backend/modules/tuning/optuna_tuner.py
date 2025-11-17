"""
Hyperparameter Tuning with Optuna
Optimized for parallel trials on high-end hardware
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from typing import Dict, Any, Callable, Optional
from loguru import logger
import torch


class OptunaTuner:
    """Hyperparameter tuning with Optuna"""

    def __init__(
        self,
        objective_func: Callable,
        n_trials: int = 100,
        n_jobs: int = 4,  # Parallel trials (good for Ryzen 9 9950X)
        study_name: Optional[str] = None,
        direction: str = "minimize",
    ):
        """
        Initialize Optuna tuner

        Args:
            objective_func: Objective function to optimize
            n_trials: Number of trials
            n_jobs: Number of parallel jobs
            study_name: Name of the study
            direction: minimize or maximize
        """
        self.objective_func = objective_func
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_name = study_name or "optuna_study"
        self.direction = direction

        # Create study with advanced sampler and pruner
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=TPESampler(n_startup_trials=10),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        logger.info(f"Created Optuna study: {self.study_name}")

    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization

        Returns:
            Best parameters and value
        """
        logger.info(f"Starting optimization with {self.n_trials} trials")

        # Run optimization with parallel trials
        self.study.optimize(
            self.objective_func,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        logger.info(f"Optimization complete!")
        logger.info(f"Best value: {self.study.best_value}")
        logger.info(f"Best params: {self.study.best_params}")

        return {
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial": self.study.best_trial.number,
        }

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters"""
        return self.study.best_params

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history"""
        try:
            from optuna.visualization import plot_optimization_history

            fig = plot_optimization_history(self.study)

            if save_path:
                fig.write_html(save_path)

            return fig
        except ImportError:
            logger.warning("Install plotly for visualization")

    def plot_param_importances(self, save_path: Optional[str] = None):
        """Plot parameter importances"""
        try:
            from optuna.visualization import plot_param_importances

            fig = plot_param_importances(self.study)

            if save_path:
                fig.write_html(save_path)

            return fig
        except ImportError:
            logger.warning("Install plotly for visualization")


# Example objective function for PyTorch model
def example_objective(trial):
    """Example objective function"""

    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Train model with these hyperparameters
    # ...

    # Return validation loss/accuracy
    val_loss = 0.5  # Placeholder

    return val_loss
