"""
Training configuration dataclass for HyGE-Mamba.

This module defines all training hyperparameters including optimizer,
scheduler, loss weights, data loading, checkpointing, and logging settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    name: str = "cosine_warmup"
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-7
    min_lr: float = 1e-6


@dataclass
class CurveWeightsConfig:
    """Per-curve loss weights."""

    lai: float = 1.0
    biomass: float = 1.0
    grain_fill: float = 1.0
    water_stress: float = 1.0


@dataclass
class TraitWeightsConfig:
    """Per-trait loss weights."""

    yield_weight: float = 2.0
    flowering_date: float = 1.0
    maturity_date: float = 1.0
    plant_height: float = 0.5
    harvest_index: float = 1.0


@dataclass
class LossWeightsConfig:
    """Loss weights for multi-task learning."""

    apsim_params: float = 1.0
    curves: float = 1.0
    scalar_traits: float = 1.0
    moe_auxiliary: float = 0.01
    curve_weights: CurveWeightsConfig = field(default_factory=CurveWeightsConfig)
    trait_weights: TraitWeightsConfig = field(default_factory=TraitWeightsConfig)


@dataclass
class DataConfig:
    """Data loading configuration."""

    data_dir: str = "data/"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_snp_length: int = 10000
    max_weather_days: int = 180
    use_augmentation: bool = True
    snp_mask_prob: float = 0.1
    weather_noise_std: float = 0.05


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    save_dir: str = "checkpoints/"
    save_frequency: int = 1
    save_best_only: bool = False
    metric_for_best: str = "val_loss"
    mode: str = "min"
    max_keep: int = 5


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str = "logs/"
    log_frequency: int = 100
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "hyge-mamba"
    log_gradients: bool = False
    log_parameters: bool = False


@dataclass
class TrainingConfig:
    """
    Complete training configuration for HyGE-Mamba.

    Aggregates all training-related sub-configurations and provides
    factory methods for common setups.
    """

    # Core training settings
    num_epochs: int = 100
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = False
    amp_dtype: str = "float16"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4

    # Sub-configurations
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Device
    device: str = "auto"
    distributed: bool = False

    @classmethod
    def fast_debug(cls) -> "TrainingConfig":
        """Quick debug configuration with minimal epochs."""
        config = cls()
        config.num_epochs = 5
        config.data.batch_size = 8
        config.data.num_workers = 0
        config.early_stopping = False
        config.checkpoint.save_frequency = 5
        config.logging.log_frequency = 1
        return config

    @classmethod
    def full_training(cls) -> "TrainingConfig":
        """Full-scale training configuration."""
        config = cls()
        config.num_epochs = 200
        config.use_amp = True
        config.data.batch_size = 64
        config.data.num_workers = 8
        config.optimizer.lr = 3e-4
        config.scheduler.warmup_epochs = 10
        config.early_stopping_patience = 30
        return config

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Create TrainingConfig from a flat or nested dictionary."""
        config = cls()

        for key in [
            "num_epochs", "gradient_clip_norm", "gradient_accumulation_steps",
            "use_amp", "amp_dtype", "seed", "deterministic",
            "early_stopping", "early_stopping_patience", "early_stopping_min_delta",
            "device", "distributed",
        ]:
            if key in d:
                setattr(config, key, d[key])

        if "optimizer" in d and isinstance(d["optimizer"], dict):
            for k, v in d["optimizer"].items():
                if hasattr(config.optimizer, k):
                    setattr(config.optimizer, k, v)

        if "scheduler" in d and isinstance(d["scheduler"], dict):
            for k, v in d["scheduler"].items():
                if hasattr(config.scheduler, k):
                    setattr(config.scheduler, k, v)

        if "data" in d and isinstance(d["data"], dict):
            for k, v in d["data"].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)

        if "checkpoint" in d and isinstance(d["checkpoint"], dict):
            for k, v in d["checkpoint"].items():
                if hasattr(config.checkpoint, k):
                    setattr(config.checkpoint, k, v)

        if "logging" in d and isinstance(d["logging"], dict):
            for k, v in d["logging"].items():
                if hasattr(config.logging, k):
                    setattr(config.logging, k, v)

        return config