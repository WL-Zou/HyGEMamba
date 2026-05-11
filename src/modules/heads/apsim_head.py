"""
APSIM Head module for HyGE-Mamba.

This module predicts APSIM (Agricultural Production Systems sIMulator)
crop model parameters from the learned GE representation.

APSIM is a widely-used crop simulation model that requires genotype-specific
parameters to predict yield. By learning to predict these parameters,
the model bridges deep learning and mechanistic crop modeling.

Key APSIM parameters for maize:
- Thermal time from emergence to end of juvenile phase
- Thermal time from flowering to maturity
- Maximum grain number
- Grain filling rate
- Radiation use efficiency
- Transpiration efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class APSIMParameterRanges:
    """
    Default APSIM parameter ranges for maize.
    
    These ranges are based on typical values from crop modeling literature
    and can be adjusted for specific cultivar groups.
    """
    
    # Thermal time parameters (C days)
    tt_emerg_to_endjuv: Tuple[float, float] = (200.0, 400.0)
    tt_flower_to_maturity: Tuple[float, float] = (600.0, 1000.0)
    
    # Grain parameters
    max_grain_number: Tuple[float, float] = (400.0, 800.0)
    grain_fill_rate: Tuple[float, float] = (5.0, 15.0)  # mg/grain/day
    
    # Efficiency parameters
    rue: Tuple[float, float] = (1.2, 2.0)  # g/MJ radiation use efficiency
    te: Tuple[float, float] = (5.0, 12.0)  # Pa transpiration efficiency
    
    @classmethod
    def get_ranges(cls) -> Dict[str, Tuple[float, float]]:
        """Get all parameter ranges as a dictionary."""
        instance = cls()
        return {
            "tt_emerg_to_endjuv": instance.tt_emerg_to_endjuv,
            "tt_flower_to_maturity": instance.tt_flower_to_maturity,
            "max_grain_number": instance.max_grain_number,
            "grain_fill_rate": instance.grain_fill_rate,
            "rue": instance.rue,
            "te": instance.te,
        }
    
    @classmethod
    def get_names(cls) -> list:
        """Get parameter names in order."""
        return [
            "tt_emerg_to_endjuv",
            "tt_flower_to_maturity",
            "max_grain_number",
            "grain_fill_rate",
            "rue",
            "te",
        ]


class BoundedOutput(nn.Module):
    """
    Output layer with bounded range.
    
    Maps unbounded network output to a specified range using sigmoid
    or other bounded activations.
    
    Args:
        min_val: Minimum output value.
        max_val: Maximum output value.
        activation: Bounding activation ('sigmoid', 'softplus', 'tanh').
    """
    
    def __init__(
        self,
        min_val: float,
        max_val: float,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply bounded transformation.
        
        Args:
            x: Unbounded input tensor.
        
        Returns:
            Bounded output in [min_val, max_val].
        """
        if self.activation == "sigmoid":
            bounded = torch.sigmoid(x)
        elif self.activation == "tanh":
            bounded = (torch.tanh(x) + 1) / 2  # Map to [0, 1]
        elif self.activation == "softplus":
            # Softplus with scaling
            bounded = torch.sigmoid(F.softplus(x) / 5)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        return self.min_val + bounded * self.range


class APSIMHead(nn.Module):
    """
    APSIM parameter prediction head.
    
    Predicts bounded APSIM crop model parameters from the GE representation.
    Each parameter has biologically meaningful bounds.
    
    Args:
        input_dim: Input dimension from GE fusion.
        hidden_dim: Hidden dimension for MLP.
        param_ranges: Dictionary of parameter name to (min, max) tuples.
                     If None, uses default maize parameters.
        num_layers: Number of hidden layers.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        # Use default ranges if not provided
        if param_ranges is None:
            param_ranges = APSIMParameterRanges.get_ranges()
        
        self.param_names = list(param_ranges.keys())
        self.num_params = len(param_ranges)
        
        # Shared hidden layers
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Parameter-specific output heads with bounded outputs
        self.param_heads = nn.ModuleDict()
        self.bounded_outputs = nn.ModuleDict()
        
        for name, (min_val, max_val) in param_ranges.items():
            self.param_heads[name] = nn.Linear(hidden_dim, 1)
            self.bounded_outputs[name] = BoundedOutput(min_val, max_val)
    
    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        Predict APSIM parameters.
        
        Args:
            x: Input tensor from GE fusion, shape (batch, input_dim).
            return_dict: If True, return dictionary of parameters.
                        If False, return stacked tensor.
        
        Returns:
            If return_dict: Dictionary mapping parameter names to predictions.
            If not return_dict: Tensor of shape (batch, num_params).
        """
        # Shared processing
        hidden = self.shared(x)
        
        # Per-parameter predictions
        params = {}
        for name in self.param_names:
            raw_output = self.param_heads[name](hidden)
            bounded_output = self.bounded_outputs[name](raw_output)
            params[name] = bounded_output.squeeze(-1)
        
        if return_dict:
            return params
        else:
            return torch.stack([params[name] for name in self.param_names], dim=-1)
    
    def get_parameter_names(self) -> list:
        """Get list of predicted parameter names."""
        return self.param_names


class APSIMParameterLoss(nn.Module):
    """
    Loss function for APSIM parameter prediction.
    
    Combines MSE loss with optional regularization for parameter
    smoothness and biological plausibility.
    
    Args:
        param_weights: Optional dictionary of per-parameter loss weights.
        use_relative_loss: If True, use relative (percentage) error.
    """
    
    def __init__(
        self,
        param_weights: Optional[Dict[str, float]] = None,
        use_relative_loss: bool = False,
    ) -> None:
        super().__init__()
        
        self.param_weights = param_weights or {}
        self.use_relative_loss = use_relative_loss
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute APSIM parameter loss.
        
        Args:
            predictions: Predicted parameters.
            targets: Target parameters.
        
        Returns:
            Tuple of (total_loss, per_parameter_losses).
        """
        losses = {}
        total_loss = 0.0
        
        for name, pred in predictions.items():
            if name not in targets:
                continue
            
            target = targets[name]
            
            if self.use_relative_loss:
                # Relative error (percentage)
                loss = ((pred - target) / (target.abs() + 1e-6)).pow(2).mean()
            else:
                # MSE loss
                loss = F.mse_loss(pred, target)
            
            weight = self.param_weights.get(name, 1.0)
            losses[name] = loss
            total_loss = total_loss + weight * loss
        
        return total_loss, losses
