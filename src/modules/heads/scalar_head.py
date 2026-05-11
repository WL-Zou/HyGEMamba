"""
Scalar Head module for HyGE-Mamba.

This module predicts scalar trait values from the GxE representation:
- Grain yield (kg/ha or bu/acre)
- Flowering date (days after planting)
- Maturity date (days after planting)
- Plant height (cm)
- Harvest index (grain/total biomass ratio)

These are the primary outputs for yield prediction and phenotypic
selection in crop breeding programs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class ScalarHead(nn.Module):
    """
    Single scalar output head with optional uncertainty estimation.

    Predicts a single scalar value, optionally with heteroscedastic
    uncertainty (predicted variance).

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension for MLP.
        num_layers: Number of hidden layers.
        output_range: Optional (min, max) tuple for bounded output.
        predict_uncertainty: If True, also predict variance.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_range: Optional[Tuple[float, float]] = None,
        predict_uncertainty: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.output_range = output_range
        self.predict_uncertainty = predict_uncertainty

        # Build MLP
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output head (1 for mean, optionally 1 for variance)
        num_outputs = 2 if predict_uncertainty else 1
        self.output = nn.Linear(hidden_dim, num_outputs)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict scalar value.

        Args:
            x: Input tensor, shape (batch, input_dim).

        Returns:
            If predict_uncertainty is False:
                Prediction, shape (batch,).
            If predict_uncertainty is True:
                Tuple of (prediction, variance), each shape (batch,).
        """
        hidden = self.mlp(x)
        output = self.output(hidden)

        if self.predict_uncertainty:
            mean = output[:, 0]
            log_var = output[:, 1]
            var = F.softplus(log_var) + 1e-6  # Ensure positive variance

            # Soft bound: tanh * 5 keeps predictions in [-5, +5] z-score space.
            # tanh'(0)=1 is larger than sigmoid'(0)=0.25 and helps avoid divergence.
            mean = 5.0 * torch.tanh(mean / 5.0)
            return mean, var
        else:
            mean = output.squeeze(-1)

            # Soft bound: tanh * 5 keeps predictions in [-5, +5] z-score space.
            # tanh'(0)=1 is larger than sigmoid'(0)=0.25 and helps avoid divergence.
            mean = 5.0 * torch.tanh(mean / 5.0)
            return mean


class MultiTaskScalarHead(nn.Module):
    """
    Multi-task scalar prediction head.

    Predicts multiple scalar traits with shared representations
    and task-specific output heads.

    Default traits for maize:
    - Yield: 0-20000 kg/ha
    - Flowering date: 50-120 days
    - Maturity date: 90-180 days
    - Plant height: 100-350 cm
    - Harvest index: 0.3-0.6

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        trait_configs: Dictionary mapping trait names to config dicts.
                      Each config can have 'range' and 'weight'.
        num_shared_layers: Number of shared layers.
        num_task_layers: Number of task-specific layers.
        predict_uncertainty: If True, predict variance for each trait.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        trait_configs: Optional[Dict[str, Dict]] = None,
        num_shared_layers: int = 2,
        num_task_layers: int = 1,
        predict_uncertainty: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.predict_uncertainty = predict_uncertainty

        # Default trait configurations
        if trait_configs is None:
            trait_configs = {
                "yield": {"range": (0.0, 20000.0), "weight": 2.0},
                "flowering_date": {"range": (50.0, 120.0), "weight": 1.0},
                "maturity_date": {"range": (90.0, 180.0), "weight": 1.0},
                "plant_height": {"range": (100.0, 350.0), "weight": 0.5},
                "harvest_index": {"range": (0.3, 0.6), "weight": 0.5},
            }

        self.trait_names = list(trait_configs.keys())
        self.trait_configs = trait_configs

        # Shared layers
        shared_layers = []
        current_dim = input_dim
        for _ in range(num_shared_layers):
            shared_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        self.shared = nn.Sequential(*shared_layers)

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for name, config in trait_configs.items():
            task_range = config.get("range", None)
            self.task_heads[name] = ScalarHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                num_layers=num_task_layers,
                output_range=task_range,
                predict_uncertainty=predict_uncertainty,
                dropout=dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        traits: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor] | Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict multiple scalar traits.

        Args:
            x: Input tensor, shape (batch, input_dim).
            traits: Optional list of traits to predict. If None, predict all.

        Returns:
            Dictionary mapping trait names to predictions.
            If predict_uncertainty, each value is (mean, variance) tuple.
        """
        # Shared processing
        shared_repr = self.shared(x)

        # Predict requested traits
        traits_to_predict = traits or self.trait_names

        predictions = {}
        for name in traits_to_predict:
            if name in self.task_heads:
                predictions[name] = self.task_heads[name](shared_repr)

        return predictions

    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for multi-task learning."""
        return {
            name: config.get("weight", 1.0)
            for name, config in self.trait_configs.items()
        }


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for scalar predictions.

    Combines per-trait losses with optional uncertainty weighting.

    Args:
        trait_weights: Dictionary of per-trait loss weights.
        use_uncertainty_weighting: If True, use homoscedastic uncertainty.
        reduction: Loss reduction method.
    """

    def __init__(
        self,
        trait_weights: Optional[Dict[str, float]] = None,
        use_uncertainty_weighting: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.trait_weights = trait_weights or {}
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.reduction = reduction

        # Learnable log-variance for homoscedastic uncertainty
        if use_uncertainty_weighting:
            # Will be populated when trait names are known
            self.log_vars = nn.ParameterDict()

    def _ensure_log_vars(self, trait_names: List[str], device: torch.device) -> None:
        """Ensure log variance parameters exist for all traits."""
        for name in trait_names:
            if name not in self.log_vars:
                self.log_vars[name] = nn.Parameter(torch.zeros(1, device=device))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.

        Args:
            predictions: Predicted values or (mean, var) tuples.
            targets: Target values.
            mask: Optional masks for valid targets.

        Returns:
            Tuple of (total_loss, per_trait_losses).
        """
        trait_names = list(predictions.keys())
        device = next(iter(predictions.values()))
        if isinstance(device, tuple):
            device = device[0].device
        else:
            device = device.device

        if self.use_uncertainty_weighting:
            self._ensure_log_vars(trait_names, device)

        losses = {}
        total_loss = torch.tensor(0.0, device=device)

        for name in trait_names:
            if name not in targets:
                continue

            pred = predictions[name]
            target = targets[name]

            # Handle uncertainty predictions
            if isinstance(pred, tuple):
                pred_mean, pred_var = pred
                # Negative log-likelihood with predicted variance
                loss = 0.5 * (
                    torch.log(pred_var) + (pred_mean - target) ** 2 / pred_var
                )
            else:
                loss = (pred - target) ** 2

            # Apply mask if provided
            if mask is not None and name in mask:
                loss = loss * mask[name]
                if self.reduction == "mean":
                    loss = loss.sum() / mask[name].sum().clamp(min=1)
                else:
                    loss = loss.sum()
            else:
                if self.reduction == "mean":
                    loss = loss.mean()
                else:
                    loss = loss.sum()

            losses[name] = loss

            # Apply weighting
            weight = self.trait_weights.get(name, 1.0)

            if self.use_uncertainty_weighting:
                # Homoscedastic uncertainty weighting
                log_var = self.log_vars[name]
                weighted_loss = loss / (2 * torch.exp(log_var)) + log_var / 2
            else:
                weighted_loss = weight * loss

            total_loss = total_loss + weighted_loss

        losses["total"] = total_loss

        return total_loss, losses


class YieldPredictionHead(nn.Module):
    """
    Specialized head for yield prediction.

    Combines direct prediction with derived predictions from other
    components (APSIM parameters, growth curves) for better accuracy.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        apsim_dim: Dimension of APSIM parameter features (optional).
        curve_dim: Dimension of curve summary features (optional).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        apsim_dim: Optional[int] = None,
        curve_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Total input dimension
        total_input = input_dim
        if apsim_dim:
            total_input += apsim_dim
        if curve_dim:
            total_input += curve_dim

        # Main prediction head
        self.mlp = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Yield range (kg/ha)
        self.yield_min = 0.0
        self.yield_max = 20000.0

    def forward(
        self,
        gxe_repr: torch.Tensor,
        apsim_params: Optional[torch.Tensor] = None,
        curve_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict yield.

        Args:
            gxe_repr: GxE representation, shape (batch, input_dim).
            apsim_params: Optional APSIM parameters.
            curve_summary: Optional growth curve summary.

        Returns:
            Predicted yield, shape (batch,).
        """
        # Concatenate inputs
        inputs = [gxe_repr]
        if apsim_params is not None:
            inputs.append(apsim_params)
        if curve_summary is not None:
            inputs.append(curve_summary)

        combined = torch.cat(inputs, dim=-1) if len(inputs) > 1 else gxe_repr

        # Predict
        raw_yield = self.mlp(combined).squeeze(-1)

        # Soft bound: tanh * 5 keeps predictions in [-5, +5] z-score space.
        return 5.0 * torch.tanh(raw_yield / 5.0)
