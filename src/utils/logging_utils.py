"""
Logging utilities for HyGE-Mamba.

Provides logging, metrics tracking, and TensorBoard integration.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json


def get_logger(
        name: str,
        level: int = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file to write logs to.
        format_string: Optional custom format string.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    if format_string is None:
        format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """
    Simple metrics logger that tracks and logs training metrics.

    Args:
        log_dir: Directory to save metrics.
        experiment_name: Name of the experiment.
    """

    def __init__(
            self,
            log_dir: Union[str, Path] = "logs",
            experiment_name: str = "experiment",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"

        # Running averages
        self._running_metrics: Dict[str, list] = {}
        self._step = 0

    def log(
            self,
            metrics: Dict[str, float],
            step: Optional[int] = None,
            prefix: str = "",
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric values.
            step: Global step (auto-incremented if None).
            prefix: Optional prefix for metric names.
        """
        if step is None:
            step = self._step
            self._step += 1

        # Add prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Add step and timestamp
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }

        # Write to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Update running averages
        for key, value in metrics.items():
            if key not in self._running_metrics:
                self._running_metrics[key] = []
            self._running_metrics[key].append(value)

    def get_average(self, metric_name: str, window: int = 100) -> float:
        """Get running average of a metric."""
        if metric_name not in self._running_metrics:
            return 0.0
        values = self._running_metrics[metric_name][-window:]
        return sum(values) / len(values) if values else 0.0

    def reset_running(self) -> None:
        """Reset running averages."""
        self._running_metrics = {}

    def format_metrics(
            self,
            metrics: Dict[str, float],
            precision: int = 4,
    ) -> str:
        """Format metrics as a readable string."""
        parts = []
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                parts.append(f"{key}: {value:.{precision}f}")
            else:
                parts.append(f"{key}: {value}")
        return " | ".join(parts)


class TensorBoardLogger:
    """
    TensorBoard logger for training visualization.

    Args:
        log_dir: Directory for TensorBoard logs.
        experiment_name: Name of the experiment.
    """

    def __init__(
            self,
            log_dir: Union[str, Path] = "runs",
            experiment_name: Optional[str] = None,
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard not installed. Install with: pip install tensorboard"
            )

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(
            self,
            tag: str,
            value: float,
            step: int,
    ) -> None:
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
            self,
            main_tag: str,
            tag_scalar_dict: Dict[str, float],
            step: int,
    ) -> None:
        """Log multiple scalars under a main tag."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(
            self,
            tag: str,
            values,
            step: int,
    ) -> None:
        """Log a histogram of values."""
        self.writer.add_histogram(tag, values, step)

    def log_image(
            self,
            tag: str,
            img_tensor,
            step: int,
    ) -> None:
        """Log an image."""
        self.writer.add_image(tag, img_tensor, step)

    def log_text(
            self,
            tag: str,
            text: str,
            step: int,
    ) -> None:
        """Log text."""
        self.writer.add_text(tag, text, step)

    def log_hyperparams(
            self,
            hparam_dict: Dict[str, Any],
            metric_dict: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log hyperparameters."""
        if metric_dict is None:
            metric_dict = {}
        self.writer.add_hparams(hparam_dict, metric_dict)

    def log_metrics(
            self,
            metrics: Dict[str, float],
            step: int,
            prefix: str = "",
    ) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)

    def log_model_graph(
            self,
            model,
            input_tensor,
    ) -> None:
        """Log model computation graph."""
        self.writer.add_graph(model, input_tensor)

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()

    def flush(self) -> None:
        """Flush pending events to disk."""
        self.writer.flush()


class WandBLogger:
    """
    Weights & Biases logger for experiment tracking.

    Args:
        project: W&B project name.
        experiment_name: Run name.
        config: Configuration to log.
    """

    def __init__(
            self,
            project: str = "hyge-mamba",
            experiment_name: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "Weights & Biases not installed. Install with: pip install wandb"
            )

        self.run = wandb.init(
            project=project,
            name=experiment_name,
            config=config,
        )

    def log(
            self,
            metrics: Dict[str, float],
            step: Optional[int] = None,
    ) -> None:
        """Log metrics."""
        if step is not None:
            self.wandb.log(metrics, step=step)
        else:
            self.wandb.log(metrics)

    def log_artifact(
            self,
            artifact_path: Union[str, Path],
            artifact_name: str,
            artifact_type: str = "model",
    ) -> None:
        """Log an artifact (e.g., model checkpoint)."""
        artifact = self.wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(str(artifact_path))
        self.run.log_artifact(artifact)

    def watch(
            self,
            model,
            log_freq: int = 100,
    ) -> None:
        """Watch model gradients and parameters."""
        self.wandb.watch(model, log_freq=log_freq)

    def finish(self) -> None:
        """Finish the W&B run."""
        self.wandb.finish()


class ProgressBar:
    """
    Simple progress bar for training.

    Args:
        total: Total number of iterations.
        prefix: Prefix string.
        bar_length: Length of the progress bar.
    """

    def __init__(
            self,
            total: int,
            prefix: str = "Progress",
            bar_length: int = 40,
    ) -> None:
        self.total = total
        self.prefix = prefix
        self.bar_length = bar_length
        self.current = 0
        self.metrics: Dict[str, float] = {}

    def update(
            self,
            n: int = 1,
            metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update progress."""
        self.current += n
        if metrics:
            self.metrics.update(metrics)
        self._print()

    def _print(self) -> None:
        """Print progress bar."""
        percent = self.current / self.total
        filled = int(self.bar_length * percent)
        bar = "" * filled + "" * (self.bar_length - filled)

        metrics_str = ""
        if self.metrics:
            parts = [f"{k}: {v:.4f}" for k, v in self.metrics.items()]
            metrics_str = " | " + " | ".join(parts)

        print(
            f"\r{self.prefix}: [{bar}] {self.current}/{self.total} "
            f"({100 * percent:.1f}%){metrics_str}",
            end="",
        )

        if self.current >= self.total:
            print()

    def reset(self) -> None:
        """Reset progress bar."""
        self.current = 0
        self.metrics = {}