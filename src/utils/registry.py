"""
Registry utilities for HyGE-Mamba.

Provides a flexible registry pattern for models, losses, and other
components to support configuration-driven instantiation.
"""

from typing import Dict, Type, Any, Callable, Optional
import functools


class Registry:
    """
    Generic registry for components.

    Allows registering classes/functions by name and instantiating
    them from configuration dictionaries.

    Example:
        >>> MODEL_REGISTRY = Registry("models")
        >>> @MODEL_REGISTRY.register("hyge_mamba")
        ... class HyGEMamba(nn.Module):
        ...     pass
        >>> model = MODEL_REGISTRY.build({"name": "hyge_mamba", "d_model": 128})
    """

    def __init__(self, name: str) -> None:
        """
        Initialize registry.

        Args:
            name: Name of this registry (for error messages).
        """
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        """
        Register a class or function.

        Args:
            name: Name to register under. If None, use class/function name.

        Returns:
            Decorator function.
        """

        def decorator(cls_or_fn):
            register_name = name or cls_or_fn.__name__

            if register_name in self._registry:
                raise ValueError(
                    f"'{register_name}' is already registered in {self._name}"
                )

            self._registry[register_name] = cls_or_fn
            return cls_or_fn

        return decorator

    def get(self, name: str) -> Type:
        """
        Get a registered component by name.

        Args:
            name: Registered name.

        Returns:
            The registered class/function.
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self._name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def build(self, config: Dict[str, Any]) -> Any:
        """
        Build a component from configuration.

        Args:
            config: Configuration dict with 'name' key and kwargs.

        Returns:
            Instantiated component.
        """
        config = dict(config)  # Copy to avoid modifying original
        name = config.pop("name")
        cls = self.get(name)
        return cls(**config)

    def list(self) -> list:
        """List all registered names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._registry.keys())})"


# Global registries
MODEL_REGISTRY = Registry("models")
LOSS_REGISTRY = Registry("losses")
OPTIMIZER_REGISTRY = Registry("optimizers")
SCHEDULER_REGISTRY = Registry("schedulers")
DATASET_REGISTRY = Registry("datasets")


def register_model(name: Optional[str] = None):
    """Decorator to register a model."""
    return MODEL_REGISTRY.register(name)


def register_loss(name: Optional[str] = None):
    """Decorator to register a loss function."""
    return LOSS_REGISTRY.register(name)


def register_optimizer(name: Optional[str] = None):
    """Decorator to register an optimizer."""
    return OPTIMIZER_REGISTRY.register(name)


def register_scheduler(name: Optional[str] = None):
    """Decorator to register a learning rate scheduler."""
    return SCHEDULER_REGISTRY.register(name)


def register_dataset(name: Optional[str] = None):
    """Decorator to register a dataset."""
    return DATASET_REGISTRY.register(name)


# Register common optimizers and schedulers
def _register_torch_components():
    """Register PyTorch's built-in optimizers and schedulers."""
    import torch.optim as optim
    from torch.optim import lr_scheduler

    # Optimizers
    for name in ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad"]:
        if hasattr(optim, name):
            OPTIMIZER_REGISTRY.register(name.lower())(getattr(optim, name))

    # Schedulers
    for name in ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR",
                 "ReduceLROnPlateau", "OneCycleLR", "CosineAnnealingWarmRestarts"]:
        if hasattr(lr_scheduler, name):
            SCHEDULER_REGISTRY.register(name.lower())(getattr(lr_scheduler, name))


# Register on import
try:
    _register_torch_components()
except ImportError:
    pass  # PyTorch not available