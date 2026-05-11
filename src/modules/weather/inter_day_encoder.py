"""
Inter-Day Encoder module for HyGE-Mamba.

This module encodes day-to-day weather patterns across the growing season,
capturing seasonal dynamics that affect crop development:
- Temperature accumulation (thermal time)
- Rainfall distribution and dry spells
- Seasonal radiation patterns
- Drought stress periods
- Critical growth stage weather

The encoder processes daily weather summaries (from IntraDayEncoder) and
produces a season-level representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..common.ssm_block import SSMBlock, MambaEncoder
from ..common.layer_norm import RMSNorm


class SeasonalPositionEncoding(nn.Module):
    """
    Position encoding for days within a growing season.

    Encodes day-of-season using both absolute position and seasonal
    phase information (e.g., early/mid/late season markers).

    Args:
        dim: Encoding dimension.
        max_days: Maximum number of days in growing season.
    """

    def __init__(
            self,
            dim: int,
            max_days: int = 365,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_days = max_days

        # Learnable day embeddings with sinusoidal initialization
        position = torch.arange(max_days).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        encoding = torch.zeros(max_days, dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 0:
            encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            encoding[:, 1::2] = torch.cos(position * div_term[:-1])

        self.encoding = nn.Parameter(encoding, requires_grad=True)

    def forward(self, day_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get day position encodings.

        Args:
            day_indices: Day indices, shape (batch, num_days).
                        If None, return all encodings.

        Returns:
            Position encodings.
        """
        if day_indices is None:
            return self.encoding
        return self.encoding[day_indices.clamp(0, self.max_days - 1)]


class GrowthStageEmbedding(nn.Module):
    """
    Embedding for crop growth stages.

    Encodes the developmental stage of the crop, which determines
    sensitivity to weather conditions.

    Maize growth stages:
        - VE-V6: Vegetative early
        - V6-VT: Vegetative late
        - VT-R1: Flowering
        - R1-R3: Early grain fill
        - R3-R6: Late grain fill

    Args:
        num_stages: Number of growth stages.
        dim: Embedding dimension.
    """

    def __init__(
            self,
            num_stages: int = 10,
            dim: int = 64,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_stages, dim)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def forward(self, stage_indices: torch.Tensor) -> torch.Tensor:
        """
        Get growth stage embeddings.

        Args:
            stage_indices: Growth stage indices, shape (batch, num_days).

        Returns:
            Stage embeddings, shape (batch, num_days, dim).
        """
        return self.embedding(stage_indices)


class InterDayEncoder(nn.Module):
    """
    Encoder for inter-day (daily/seasonal) weather patterns.

    Processes daily weather summaries across the growing season using
    SSM blocks to capture seasonal dynamics and long-term weather patterns.

    Args:
        input_dim: Dimension of daily weather summaries (from IntraDayEncoder).
        d_model: Model dimension.
        d_state: SSM state dimension.
        num_layers: Number of SSM layers.
        max_days: Maximum number of days in growing season.
        dropout: Dropout probability.
        use_positional_encoding: If True, add day position encoding.
        use_growth_stage: If True, include growth stage information.
        num_growth_stages: Number of growth stages (if using growth stage).
    """

    def __init__(
            self,
            input_dim: int = 64,
            d_model: int = 128,
            d_state: int = 16,
            num_layers: int = 3,
            max_days: int = 365,
            dropout: float = 0.1,
            use_positional_encoding: bool = True,
            use_growth_stage: bool = True,
            num_growth_stages: int = 10,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.use_growth_stage = use_growth_stage

        # Input projection
        proj_input_dim = input_dim
        if use_growth_stage:
            proj_input_dim += d_model // 4  # Stage embedding dimension
            self.growth_stage_embedding = GrowthStageEmbedding(
                num_stages=num_growth_stages,
                dim=d_model // 4,
            )

        self.input_proj = nn.Sequential(
            nn.Linear(proj_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Position encoding
        if use_positional_encoding:
            self.position_encoding = SeasonalPositionEncoding(d_model, max_days)

        # SSM layers for temporal modeling
        self.ssm_layers = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                dropout=dropout,
                bidirectional=True,
            )
            for _ in range(num_layers)
        ])

        # Thermal time accumulator (for GDD tracking)
        self.thermal_accumulator = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            batch_first=True,
            bidirectional=False,  # Unidirectional for accumulation
        )

        # Combine SSM and thermal outputs
        self.combine_proj = nn.Linear(d_model + d_model // 2, d_model)

        self.output_norm = RMSNorm(d_model)

    def forward(
            self,
            daily_weather: torch.Tensor,
            day_indices: Optional[torch.Tensor] = None,
            growth_stages: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode daily weather across the growing season.

        Args:
            daily_weather: Daily weather summaries, shape (batch, num_days, input_dim).
            day_indices: Day-of-year indices, shape (batch, num_days).
            growth_stages: Growth stage indices, shape (batch, num_days).
            mask: Optional mask for valid days, shape (batch, num_days).

        Returns:
            Season-encoded daily representations, shape (batch, num_days, d_model).
        """
        batch_size, num_days, _ = daily_weather.shape

        # Optionally add growth stage information
        if self.use_growth_stage and growth_stages is not None:
            stage_emb = self.growth_stage_embedding(growth_stages)
            daily_weather = torch.cat([daily_weather, stage_emb], dim=-1)
        elif self.use_growth_stage:
            # Default growth stages based on day index
            default_stages = torch.arange(num_days, device=daily_weather.device)
            default_stages = (default_stages * 10 // num_days).clamp(max=9)
            default_stages = default_stages.unsqueeze(0).expand(batch_size, -1)
            stage_emb = self.growth_stage_embedding(default_stages)
            daily_weather = torch.cat([daily_weather, stage_emb], dim=-1)

        # Project to model dimension
        x = self.input_proj(daily_weather)

        # Add position encoding
        if self.use_positional_encoding:
            if day_indices is not None:
                pos_enc = self.position_encoding(day_indices)
            else:
                pos_enc = self.position_encoding()[:num_days].unsqueeze(0)
            x = x + pos_enc

        # Process through SSM layers
        ssm_output = x
        for ssm_layer in self.ssm_layers:
            ssm_output = ssm_layer(ssm_output, mask)

        # Thermal time accumulation (forward pass)
        thermal_output, _ = self.thermal_accumulator(x)

        # Combine SSM and thermal outputs
        combined = torch.cat([ssm_output, thermal_output], dim=-1)
        output = self.combine_proj(combined)

        output = self.output_norm(output)

        return output


class SeasonalAttentionPooling(nn.Module):
    """
    Attention-based pooling for seasonal representation.

    Uses learnable queries for different aspects of the growing season:
    - Early season establishment
    - Vegetative growth
    - Flowering/pollination
    - Grain fill
    - Maturation

    Args:
        d_model: Model dimension.
        num_queries: Number of seasonal queries.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            d_model: int = 128,
            num_queries: int = 5,
            num_heads: int = 4,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_queries = num_queries

        # Learnable queries for different growth periods
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model * num_queries, d_model)
        self.norm = RMSNorm(d_model)

    def forward(
            self,
            season_repr: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool seasonal representation to fixed-size output.

        Args:
            season_repr: Daily representations, shape (batch, num_days, d_model).
            mask: Optional mask for valid days.

        Returns:
            Pooled season representation, shape (batch, d_model).
        """
        batch_size = season_repr.shape[0]

        # Expand queries for batch
        queries = self.queries.expand(batch_size, -1, -1)

        # Attend to daily representations
        attended, _ = self.attention(
            queries, season_repr, season_repr,
            key_padding_mask=~mask if mask is not None else None,
        )

        # Flatten and project
        attended_flat = attended.reshape(batch_size, -1)
        output = self.output_proj(attended_flat)
        output = self.norm(output)

        return output


class WeatherAnomalyDetector(nn.Module):
    """
    Detect weather anomalies and stress events.

    Identifies critical weather events that significantly impact crop yield:
    - Heat stress (extreme high temperatures)
    - Cold stress (frost, chilling)
    - Drought (extended dry periods)
    - Flooding (excessive precipitation)
    - Combined stresses

    Args:
        input_dim: Dimension of daily weather representation.
        hidden_dim: Hidden dimension for detection network.
        num_anomaly_types: Number of anomaly types to detect.
    """

    def __init__(
            self,
            input_dim: int = 128,
            hidden_dim: int = 64,
            num_anomaly_types: int = 5,
    ) -> None:
        super().__init__()

        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_anomaly_types),
            nn.Sigmoid(),
        )

        # Severity estimator
        self.severity = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(
            self,
            daily_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies in daily weather.

        Args:
            daily_repr: Daily weather representations, shape (batch, num_days, input_dim).

        Returns:
            Tuple of:
                - Anomaly probabilities, shape (batch, num_days, num_anomaly_types)
                - Anomaly severity, shape (batch, num_days, 1)
        """
        anomaly_probs = self.detector(daily_repr)
        severity = self.severity(daily_repr)

        return anomaly_probs, severity