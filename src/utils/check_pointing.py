"""
Checkpointing utilities for HyGE-Mamba.

Provides utilities for saving and loading model checkpoints,
including support for distributed training and automatic cleanup.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
import shutil
from datetime import datetime


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    path: Union[str, Path] = "checkpoint.pt",
    save_config_json: bool = True,
) -> Path:
    """
    Save a training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optional optimizer state.
        scheduler: Optional scheduler state.
        epoch: Current epoch.
        step: Current global step.
        metrics: Optional metrics dictionary.
        config: Optional configuration dictionary.
        path: Path to save checkpoint.
        save_config_json: If True, also save config as JSON.

    Returns:
        Path where checkpoint was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)

    # Save config as JSON for easy inspection
    if save_config_json and config is not None:
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    return path


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Optional model to load state into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        map_location: Device mapping (e.g., 'cpu', 'cuda:0').
        strict: If True, require exact model state dict match.

    Returns:
        Checkpoint dictionary.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class CheckpointManager:
    """
    Manages multiple checkpoints with automatic cleanup.

    Keeps the best N checkpoints and most recent N checkpoints,
    automatically removing old ones.

    Args:
        save_dir: Directory to save checkpoints.
        max_to_keep: Maximum number of recent checkpoints to keep.
        best_metric: Metric name for best model selection.
        mode: 'min' or 'max' for best metric comparison.
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        max_to_keep: int = 5,
        best_metric: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.max_to_keep = max_to_keep
        self.best_metric = best_metric
        self.mode = mode

        self.checkpoints: List[Dict[str, Any]] = []
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_checkpoint_path: Optional[Path] = None

        # Load existing checkpoints info
        self._scan_existing_checkpoints()

    def _scan_existing_checkpoints(self) -> None:
        """Scan for existing checkpoints in the directory."""
        for ckpt_path in sorted(self.save_dir.glob("checkpoint_*.pt")):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                self.checkpoints.append({
                    "path": ckpt_path,
                    "epoch": ckpt.get("epoch", 0),
                    "step": ckpt.get("step", 0),
                    "metrics": ckpt.get("metrics", {}),
                })
            except Exception:
                pass

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a checkpoint and manage cleanup.

        Args:
            model: Model to save.
            optimizer: Optional optimizer.
            scheduler: Optional scheduler.
            epoch: Current epoch.
            step: Current step.
            metrics: Metrics dictionary.
            config: Configuration dictionary.

        Returns:
            Path to saved checkpoint.
        """
        # Generate filename
        filename = f"checkpoint_epoch{epoch:04d}_step{step:08d}.pt"
        path = self.save_dir / filename

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            metrics=metrics,
            config=config,
            path=path,
        )

        # Track checkpoint
        self.checkpoints.append({
            "path": path,
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        })

        # Check if this is the best
        if metrics is not None and self.best_metric in metrics:
            current_value = metrics[self.best_metric]
            is_best = (
                (self.mode == "min" and current_value < self.best_value) or
                (self.mode == "max" and current_value > self.best_value)
            )

            if is_best:
                self.best_value = current_value
                self.best_checkpoint_path = path

                # Save as best
                best_path = self.save_dir / "best_model.pt"
                shutil.copy(path, best_path)

        # Cleanup old checkpoints
        self._cleanup()

        return path

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond max_to_keep."""
        # Keep best checkpoint
        protected = set()
        if self.best_checkpoint_path:
            protected.add(self.best_checkpoint_path)

        # Sort by step (most recent first)
        self.checkpoints.sort(key=lambda x: x["step"], reverse=True)

        # Keep max_to_keep most recent
        to_remove = []
        kept = 0
        for ckpt_info in self.checkpoints:
            if ckpt_info["path"] in protected:
                continue

            if kept < self.max_to_keep:
                kept += 1
            else:
                to_remove.append(ckpt_info)

        # Remove old checkpoints
        for ckpt_info in to_remove:
            try:
                ckpt_info["path"].unlink()
                self.checkpoints.remove(ckpt_info)
            except Exception:
                pass

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.save_dir / "best_model.pt"

        if not best_path.exists():
            raise FileNotFoundError("Best model checkpoint not found")

        return load_checkpoint(
            path=best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
        )

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load the most recent checkpoint."""
        if not self.checkpoints:
            raise FileNotFoundError("No checkpoints found")

        # Sort by step and get most recent
        self.checkpoints.sort(key=lambda x: x["step"], reverse=True)
        latest_path = self.checkpoints[0]["path"]

        return load_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
        )

    @property
    def latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        if not self.checkpoints:
            return None
        self.checkpoints.sort(key=lambda x: x["step"], reverse=True)
        return self.checkpoints[0]["path"]


def save_model_for_inference(
    model: nn.Module,
    path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model for inference (without optimizer state).

    Args:
        model: Model to save.
        path: Output path.
        config: Optional configuration.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, path)


def load_model_for_inference(
    model: nn.Module,
    path: Union[str, Path],
    map_location: Optional[str] = None,
) -> nn.Module:
    """
    Load model for inference.

    Args:
        model: Model instance to load weights into.
        path: Checkpoint path.
        map_location: Device mapping.

    Returns:
        Model with loaded weights.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model