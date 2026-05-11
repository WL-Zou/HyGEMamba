"""
Multi-Task Loss module for HyGE-Mamba.

This module provides loss functions for training with multiple
prediction targets (yield, APSIM params, growth curves, scalar traits).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TaskWeights:
    """Weights for multi-task loss balancing."""

    yield_weight: float = 2.0  # Higher weight for primary target
    apsim_weight: float = 1.0
    curve_weight: float = 0.5
    scalar_weight: float = 1.0
    moe_aux_weight: float = 0.01

    def to_dict(self) -> Dict[str, float]:
        return {
            "yield": self.yield_weight,
            "apsim": self.apsim_weight,
            "curve": self.curve_weight,
            "scalar": self.scalar_weight,
            "moe_aux": self.moe_aux_weight,
        }


class YieldLoss(nn.Module):
    """
    Loss function for yield prediction.

    Uses MSE with optional Huber loss for robustness to outliers.

    Args:
        use_huber: If True, use Huber loss instead of MSE.
        huber_delta: Delta parameter for Huber loss.
    """

    def __init__(
            self,
            use_huber: bool = False,
            huber_delta: float = 1.0,
    ) -> None:
        super().__init__()

        self.use_huber = use_huber
        self.huber_delta = huber_delta

        if use_huber:
            self.loss_fn = nn.HuberLoss(delta=huber_delta)
        else:
            self.loss_fn = nn.MSELoss()

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute yield loss.

        Args:
            pred: Predicted yield, shape (batch,).
            target: Target yield, shape (batch,).
            mask: Optional mask for valid samples.

        Returns:
            Scalar loss value.
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        return self.loss_fn(pred, target)


class APSIMParameterLoss(nn.Module):
    """
    Loss function for APSIM parameter prediction.

    Handles bounded parameters by optionally using transformed space.

    Args:
        param_ranges: Dictionary of parameter ranges {name: (min, max)}.
        use_normalized: If True, normalize targets to [0, 1].
    """

    def __init__(
            self,
            param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
            use_normalized: bool = True,
    ) -> None:
        super().__init__()

        self.param_ranges = param_ranges
        self.use_normalized = use_normalized
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute APSIM parameter loss.

        Args:
            pred: Predicted parameters, shape (batch, num_params).
            target: Target parameters, shape (batch, num_params).
            mask: Optional mask for valid samples.

        Returns:
            Scalar loss value.
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        # Compute per-parameter loss
        losses = self.loss_fn(pred, target)

        # Average across parameters and batch
        return losses.mean()


class ScalarTraitLoss(nn.Module):
    """
    Loss function for scalar trait prediction.

    Supports different loss functions per trait.

    Args:
        num_traits: Number of scalar traits.
        trait_weights: Optional per-trait weights.
    """

    def __init__(
            self,
            num_traits: int = 5,
            trait_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_traits = num_traits

        if trait_weights is None:
            self.register_buffer("trait_weights", torch.ones(num_traits))
        else:
            self.register_buffer("trait_weights", trait_weights)

        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scalar trait loss.

        Args:
            pred: Predicted traits, shape (batch, num_traits).
            target: Target traits, shape (batch, num_traits).
            mask: Optional mask for valid samples.

        Returns:
            Scalar loss value.
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        # Per-trait loss
        losses = self.loss_fn(pred, target)  # (batch, num_traits)

        # Weighted average
        weighted = losses * self.trait_weights

        return weighted.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for HyGE-Mamba.

    Combines losses from multiple prediction heads with learned or
    fixed task weights.

    Args:
        task_weights: TaskWeights object or dict of weights.
        use_uncertainty_weighting: If True, learn task weights.
        yield_loss_type: Type of yield loss ('mse', 'huber').
        apsim_param_ranges: Parameter ranges for APSIM loss.
    """

    def __init__(
            self,
            task_weights: Optional[TaskWeights] = None,
            use_uncertainty_weighting: bool = False,
            yield_loss_type: str = "mse",
            apsim_param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__()

        if task_weights is None:
            task_weights = TaskWeights()

        if isinstance(task_weights, dict):
            self.weights = task_weights
        else:
            self.weights = task_weights.to_dict()

        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Individual loss functions
        self.yield_loss = YieldLoss(use_huber=(yield_loss_type == "huber"))
        self.apsim_loss = APSIMParameterLoss(param_ranges=apsim_param_ranges)
        self.scalar_loss = ScalarTraitLoss()

        # Curve loss will be imported
        from .curve_loss import CurveLoss
        self.curve_loss = CurveLoss()

        # Uncertainty weighting (homoscedastic uncertainty)
        if use_uncertainty_weighting:
            # log(sigma^2) for numerical stability
            self.log_vars = nn.ParameterDict({
                "yield": nn.Parameter(torch.zeros(1)),
                "apsim": nn.Parameter(torch.zeros(1)),
                "curve": nn.Parameter(torch.zeros(1)),
                "scalar": nn.Parameter(torch.zeros(1)),
            })

    def forward(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            moe_aux_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined multi-task loss.

        Args:
            predictions: Dictionary of predictions with keys:
                - 'yield': (batch,)
                - 'apsim_params': (batch, num_params)
                - 'growth_curves': (batch, num_curves, timesteps)
                - 'scalar_traits': (batch, num_traits)
            targets: Dictionary of targets with same keys.
            moe_aux_loss: Optional MoE auxiliary loss.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual task losses.
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        # Yield loss
        if "yield" in predictions and "yields" in targets:
            yield_loss = self.yield_loss(predictions["yield"], targets["yields"])
            loss_dict["yield"] = yield_loss

            if self.use_uncertainty_weighting:
                precision = torch.exp(-self.log_vars["yield"])
                total_loss = total_loss + precision * yield_loss + self.log_vars["yield"]
            else:
                total_loss = total_loss + self.weights["yield"] * yield_loss

        # APSIM parameter loss
        if "apsim_params" in predictions and "apsim_params" in targets:
            apsim_loss = self.apsim_loss(
                predictions["apsim_params"], targets["apsim_params"]
            )
            loss_dict["apsim"] = apsim_loss

            if self.use_uncertainty_weighting:
                precision = torch.exp(-self.log_vars["apsim"])
                total_loss = total_loss + precision * apsim_loss + self.log_vars["apsim"]
            else:
                total_loss = total_loss + self.weights["apsim"] * apsim_loss

        # Growth curve loss
        if "growth_curves" in predictions and "growth_curves" in targets:
            curve_loss = self.curve_loss(
                predictions["growth_curves"], targets["growth_curves"]
            )
            loss_dict["curve"] = curve_loss

            if self.use_uncertainty_weighting:
                precision = torch.exp(-self.log_vars["curve"])
                total_loss = total_loss + precision * curve_loss + self.log_vars["curve"]
            else:
                total_loss = total_loss + self.weights["curve"] * curve_loss

        # Scalar trait loss
        if "scalar_traits" in predictions and "scalar_traits" in targets:
            scalar_loss = self.scalar_loss(
                predictions["scalar_traits"], targets["scalar_traits"]
            )
            loss_dict["scalar"] = scalar_loss

            if self.use_uncertainty_weighting:
                precision = torch.exp(-self.log_vars["scalar"])
                total_loss = total_loss + precision * scalar_loss + self.log_vars["scalar"]
            else:
                total_loss = total_loss + self.weights["scalar"] * scalar_loss

        # MoE auxiliary loss
        if moe_aux_loss is not None:
            loss_dict["moe_aux"] = moe_aux_loss
            total_loss = total_loss + self.weights["moe_aux"] * moe_aux_loss

        loss_dict["total"] = total_loss

        return total_loss, loss_dict