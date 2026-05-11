"""
Curve Decoder module for HyGE-Mamba.

This module decodes time-resolved growth curves from the learned
GE representation. Growth curves capture the temporal dynamics
of crop development:

- Leaf Area Index (LAI): Canopy development over time
- Biomass accumulation: Total plant dry matter
- Grain fill: Grain weight development
- Water stress: Cumulative stress over the season

These curves provide interpretable outputs that can be validated
against field observations and integrated with crop models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal positions.
    
    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length.
    """
    
    def __init__(self, d_model: int, max_len: int = 365) -> None:
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2] if d_model % 2 else div_term)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.shape[1]
        return x + self.pe[:seq_len].unsqueeze(0)


class CurveDecoder(nn.Module):
    """
    Decoder for time-resolved growth curves.
    
    Uses a transformer-style decoder to generate curves autoregressively
    or in parallel, conditioned on the GE representation.
    
    Args:
        input_dim: Dimension of GE representation.
        hidden_dim: Hidden dimension for decoder.
        num_timesteps: Number of output timesteps.
        num_curves: Number of output curves (e.g., LAI, biomass, etc.).
        num_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        curve_bounds: Optional dict mapping curve names to (min, max) bounds.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_timesteps: int = 180,
        num_curves: int = 4,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        curve_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.num_curves = num_curves
        
        # Default curve bounds
        self.curve_bounds = curve_bounds or {
            "lai": (0.0, 8.0),
            "biomass": (0.0, 30000.0),  # kg/ha
            "grain_fill": (0.0, 1.0),  # fraction
            "water_stress": (0.0, 1.0),
        }
        self.curve_names = list(self.curve_bounds.keys())[:num_curves]
        
        # Project GE representation to decoder dimension
        self.condition_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Temporal position encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, num_timesteps)
        
        # Learnable query tokens for parallel decoding
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_timesteps, hidden_dim) * 0.02
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Per-curve output heads
        self.curve_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_curves)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        condition: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode growth curves from GE representation.
        
        Args:
            condition: GE representation, shape (batch, input_dim).
            timesteps: Optional timestep indices, shape (batch, num_steps).
                      If None, use all timesteps.
        
        Returns:
            Growth curves, shape (batch, num_timesteps, num_curves).
        """
        batch_size = condition.shape[0]
        
        # Project condition to decoder dimension
        memory = self.condition_proj(condition).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Prepare query tokens
        queries = self.query_tokens.expand(batch_size, -1, -1)
        queries = self.pos_encoding(queries)
        
        # Decode
        decoded = self.decoder(queries, memory)
        decoded = self.output_norm(decoded)
        
        # Generate curves
        curves = []
        for i, (head, name) in enumerate(zip(self.curve_heads, self.curve_names)):
            raw_curve = head(decoded).squeeze(-1)  # (batch, num_timesteps)
            
            # Apply bounds
            min_val, max_val = self.curve_bounds.get(name, (0.0, 1.0))
            bounded_curve = torch.sigmoid(raw_curve) * (max_val - min_val) + min_val
            curves.append(bounded_curve)
        
        output = torch.stack(curves, dim=-1)  # (batch, num_timesteps, num_curves)
        
        return output


class GrowthCurveDecoder(nn.Module):
    """
    Specialized decoder for biologically-constrained growth curves.
    
    Incorporates biological constraints:
    - LAI follows logistic growth
    - Biomass is monotonically increasing
    - Grain fill follows sigmoid pattern
    
    Args:
        input_dim: Dimension of GE representation.
        hidden_dim: Hidden dimension.
        num_timesteps: Number of output timesteps.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_timesteps: int = 180,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # LAI curve parameters (logistic growth)
        # LAI(t) = L_max / (1 + exp(-k * (t - t_mid)))
        self.lai_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),  # L_max, k, t_mid, baseline
        )
        
        # Biomass parameters (monotonic with growth rate)
        # Biomass(t) = B_max * (1 - exp(-r * t))
        self.biomass_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),  # B_max, r, baseline
        )
        
        # Grain fill parameters (sigmoid)
        # GF(t) = 1 / (1 + exp(-k * (t - t_onset)))
        self.grain_fill_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),  # k, t_onset, max_fill
        )
        
        # Water stress (cumulative, from daily values)
        self.water_stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_timesteps),  # Daily stress factors
        )
        
        # Time tensor (normalized 0-1)
        time = torch.linspace(0, 1, num_timesteps)
        self.register_buffer("time", time)
    
    def _logistic(
        self,
        t: torch.Tensor,
        L_max: torch.Tensor,
        k: torch.Tensor,
        t_mid: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        """Logistic growth function."""
        return baseline + L_max / (1 + torch.exp(-k * (t - t_mid)))
    
    def _saturating_growth(
        self,
        t: torch.Tensor,
        B_max: torch.Tensor,
        r: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        """Saturating growth function (monotonic increasing)."""
        return baseline + B_max * (1 - torch.exp(-r * t))
    
    def forward(
        self,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode growth curves with biological constraints.
        
        Args:
            condition: GE representation, shape (batch, input_dim).
        
        Returns:
            Tuple of:
                - Growth curves, shape (batch, num_timesteps, 4)
                - Parameter dictionary for interpretability
        """
        batch_size = condition.shape[0]
        t = self.time.unsqueeze(0).expand(batch_size, -1)  # (batch, num_timesteps)
        
        # Shared features
        features = self.shared(condition)
        
        # LAI curve
        lai_params = self.lai_head(features)
        L_max = F.softplus(lai_params[:, 0]) * 6 + 2  # Range ~2-8
        k = F.softplus(lai_params[:, 1]) * 10 + 5  # Steepness
        t_mid = torch.sigmoid(lai_params[:, 2]) * 0.6 + 0.2  # 20-80% of season
        baseline = F.softplus(lai_params[:, 3]) * 0.2
        
        lai_curve = self._logistic(
            t,
            L_max.unsqueeze(-1),
            k.unsqueeze(-1),
            t_mid.unsqueeze(-1),
            baseline.unsqueeze(-1),
        )
        
        # Biomass curve
        biomass_params = self.biomass_head(features)
        B_max = F.softplus(biomass_params[:, 0]) * 25000 + 5000  # 5000-30000 kg/ha
        r = F.softplus(biomass_params[:, 1]) * 3 + 1
        b_baseline = F.softplus(biomass_params[:, 2]) * 100
        
        biomass_curve = self._saturating_growth(
            t,
            B_max.unsqueeze(-1),
            r.unsqueeze(-1),
            b_baseline.unsqueeze(-1),
        )
        
        # Grain fill curve
        gf_params = self.grain_fill_head(features)
        gf_k = F.softplus(gf_params[:, 0]) * 15 + 5
        t_onset = torch.sigmoid(gf_params[:, 1]) * 0.3 + 0.4  # 40-70% of season
        max_fill = torch.sigmoid(gf_params[:, 2])
        
        grain_fill_curve = max_fill.unsqueeze(-1) * torch.sigmoid(
            gf_k.unsqueeze(-1) * (t - t_onset.unsqueeze(-1))
        )
        
        # Water stress curve
        water_stress_logits = self.water_stress_head(features)
        water_stress_curve = torch.sigmoid(water_stress_logits)
        
        # Stack curves
        curves = torch.stack([
            lai_curve,
            biomass_curve,
            grain_fill_curve,
            water_stress_curve,
        ], dim=-1)
        
        # Parameter dictionary for interpretability
        params = {
            "lai_max": L_max,
            "lai_k": k,
            "lai_t_mid": t_mid,
            "biomass_max": B_max,
            "grain_fill_onset": t_onset,
        }
        
        return curves, params


class CurveLoss(nn.Module):
    """
    Loss function for growth curve predictions.
    
    Combines point-wise MSE with shape-aware terms:
    - Smoothness regularization
    - Monotonicity constraints (where applicable)
    - Peak alignment
    
    Args:
        smoothness_weight: Weight for smoothness term.
        monotonicity_weight: Weight for monotonicity term.
        curve_weights: Per-curve loss weights.
    """
    
    def __init__(
        self,
        smoothness_weight: float = 0.01,
        monotonicity_weight: float = 0.1,
        curve_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        
        self.smoothness_weight = smoothness_weight
        self.monotonicity_weight = monotonicity_weight
        self.curve_weights = curve_weights or {
            "lai": 1.0,
            "biomass": 1.0,
            "grain_fill": 1.0,
            "water_stress": 0.5,
        }
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute curve loss.
        
        Args:
            predictions: Predicted curves, shape (batch, timesteps, num_curves).
            targets: Target curves, shape (batch, timesteps, num_curves).
            mask: Optional mask for valid timesteps.
        
        Returns:
            Tuple of (total_loss, loss_components).
        """
        losses = {}
        
        # MSE loss
        mse = F.mse_loss(predictions, targets, reduction="none")
        if mask is not None:
            mse = mse * mask.unsqueeze(-1)
            mse_loss = mse.sum() / (mask.sum() * predictions.shape[-1])
        else:
            mse_loss = mse.mean()
        losses["mse"] = mse_loss
        
        # Smoothness: penalize large second derivatives
        if self.smoothness_weight > 0:
            diff1 = predictions[:, 1:, :] - predictions[:, :-1, :]
            diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
            smoothness_loss = (diff2 ** 2).mean()
            losses["smoothness"] = smoothness_loss
        else:
            smoothness_loss = 0.0
        
        # Monotonicity for biomass (index 1)
        if self.monotonicity_weight > 0:
            biomass_diff = predictions[:, 1:, 1] - predictions[:, :-1, 1]
            monotonicity_loss = F.relu(-biomass_diff).mean()
            losses["monotonicity"] = monotonicity_loss
        else:
            monotonicity_loss = 0.0
        
        total_loss = (
            mse_loss
            + self.smoothness_weight * smoothness_loss
            + self.monotonicity_weight * monotonicity_loss
        )
        
        losses["total"] = total_loss
        
        return total_loss, losses
