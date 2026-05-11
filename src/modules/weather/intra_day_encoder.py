"""
Intra-Day Encoder module for HyGE-Mamba.

This module encodes hourly weather patterns within each day, capturing
diurnal variations that affect crop physiology:
- Temperature cycles (day/night)
- Solar radiation patterns
- Vapor pressure deficit variations
- Wind speed patterns

The encoder processes each day's hourly data and produces a daily
summary representation that captures the relevant intra-day dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..common.ssm_block import SSMBlock
from ..common.layer_norm import RMSNorm


class HourlyPositionEncoding(nn.Module):
    """
    Position encoding for hours within a day.

    Encodes the hour of day using sinusoidal functions to capture
    the cyclical nature of diurnal patterns.

    Args:
        dim: Encoding dimension.
        hours_per_day: Number of hours (default 24).
    """

    def __init__(
            self,
            dim: int,
            hours_per_day: int = 24,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.hours_per_day = hours_per_day

        # Create sinusoidal encoding for hours
        position = torch.arange(hours_per_day).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )

        encoding = torch.zeros(hours_per_day, dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[:dim // 2] if dim % 2 == 0 else div_term)

        self.register_buffer("encoding", encoding)

    def forward(self, hours: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get hour position encodings.

        Args:
            hours: Optional hour indices. If None, return all 24 encodings.

        Returns:
            Position encodings, shape (hours_per_day, dim) or (batch, seq, dim).
        """
        if hours is None:
            return self.encoding
        return self.encoding[hours]


class IntraDayEncoder(nn.Module):
    """
    Encoder for intra-day (hourly) weather patterns.

    Processes hourly weather data within each day using SSM blocks
    to capture temporal dependencies and produces a daily summary.

    Input weather features typically include:
        - Temperature (C)
        - Humidity (%)
        - Solar radiation (W/m)
        - Wind speed (m/s)
        - Precipitation (mm)
        - Vapor pressure deficit (kPa)
        - Soil temperature (C)
        - Photosynthetically active radiation (PAR)

    Args:
        input_features: Number of input weather features.
        d_model: Model dimension after projection.
        d_state: SSM state dimension.
        num_layers: Number of SSM layers.
        hours_per_day: Number of hours per day (default 24).
        dropout: Dropout probability.
        aggregation: How to aggregate hours ('pool', 'last', 'attention').
    """

    def __init__(
            self,
            input_features: int = 8,
            d_model: int = 64,
            d_state: int = 8,
            num_layers: int = 2,
            hours_per_day: int = 24,
            dropout: float = 0.1,
            aggregation: str = "attention",
    ) -> None:
        super().__init__()

        self.input_features = input_features
        self.d_model = d_model
        self.hours_per_day = hours_per_day
        self.aggregation = aggregation

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Hour position encoding
        self.hour_encoding = HourlyPositionEncoding(d_model, hours_per_day)

        # Feature normalization (per-feature standardization)
        self.feature_norm = nn.LayerNorm(input_features)

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

        # Aggregation mechanism
        if aggregation == "attention":
            self.agg_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.agg_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
        elif aggregation == "pool":
            self.pool_weights = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        self.output_norm = RMSNorm(d_model)

    def forward(
            self,
            hourly_weather: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode hourly weather data for a single day.

        Args:
            hourly_weather: Weather features, shape (batch, hours, features)
                           or (batch, days, hours, features).
            mask: Optional mask for missing hours, shape (batch, hours)
                  or (batch, days, hours).

        Returns:
            Daily summary, shape (batch, d_model) or (batch, days, d_model).
        """
        # Handle both single-day and multi-day inputs
        multi_day = hourly_weather.dim() == 4
        if multi_day:
            batch_size, num_days, hours, features = hourly_weather.shape
            hourly_weather = hourly_weather.reshape(batch_size * num_days, hours, features)
            if mask is not None:
                mask = mask.reshape(batch_size * num_days, hours)
        else:
            batch_size = hourly_weather.shape[0]

        # Normalize features
        x = self.feature_norm(hourly_weather)

        # Project to model dimension
        x = self.input_proj(x)

        # Add hour position encoding
        hour_pos = self.hour_encoding()[:hourly_weather.shape[1]]
        x = x + hour_pos.unsqueeze(0)

        # Process through SSM layers
        for ssm_layer in self.ssm_layers:
            x = ssm_layer(x, mask)

        # Aggregate to daily representation
        if self.aggregation == "attention":
            query = self.agg_query.expand(x.shape[0], -1, -1)
            daily_repr, _ = self.agg_attn(
                query, x, x,
                key_padding_mask=~mask if mask is not None else None,
            )
            daily_repr = daily_repr.squeeze(1)
        elif self.aggregation == "pool":
            weights = self.pool_weights(x).squeeze(-1)
            if mask is not None:
                weights = weights.masked_fill(~mask, float("-inf"))
            weights = F.softmax(weights, dim=-1)
            daily_repr = torch.einsum("bh,bhd->bd", weights, x)
        elif self.aggregation == "last":
            daily_repr = x[:, -1, :]
        else:
            # Mean pooling
            if mask is not None:
                x = x * mask.unsqueeze(-1).float()
                daily_repr = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                daily_repr = x.mean(dim=1)

        daily_repr = self.output_norm(daily_repr)

        # Reshape for multi-day input
        if multi_day:
            daily_repr = daily_repr.reshape(batch_size, num_days, -1)

        return daily_repr


class DiurnalFeatureExtractor(nn.Module):
    """
    Extract specific diurnal features from hourly data.

    Computes biologically relevant summary statistics:
    - Daily min/max/mean temperatures
    - Growing degree days (GDD)
    - Vapor pressure deficit accumulation
    - Radiation sum

    Args:
        input_features: Number of input weather features.
        output_dim: Output dimension after feature extraction.
    """

    def __init__(
            self,
            input_features: int = 8,
            output_dim: int = 64,
    ) -> None:
        super().__init__()

        # Assume standard weather feature ordering:
        # [temp, humidity, radiation, wind, precip, vpd, soil_temp, par]
        self.temp_idx = 0
        self.rad_idx = 2
        self.vpd_idx = 5

        # Number of derived features
        # temp: min, max, mean, range, gdd_10, gdd_30
        # rad: sum, max, mean
        # vpd: sum, max, mean
        # other: means
        num_derived = 12 + (input_features - 3)  # 12 special + rest as means

        self.projection = nn.Linear(num_derived, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
            self,
            hourly_weather: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract diurnal features.

        Args:
            hourly_weather: Weather data, shape (batch, hours, features).
            mask: Optional mask for valid hours.

        Returns:
            Derived features, shape (batch, output_dim).
        """
        if mask is not None:
            # Mask invalid values
            hourly_weather = hourly_weather * mask.unsqueeze(-1).float()
            valid_hours = mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            valid_hours = hourly_weather.shape[1]

        # Temperature features
        temp = hourly_weather[:, :, self.temp_idx]
        temp_min = temp.min(dim=1)[0]
        temp_max = temp.max(dim=1)[0]
        temp_mean = temp.sum(dim=1) / valid_hours.squeeze(-1) if mask is not None else temp.mean(dim=1)
        temp_range = temp_max - temp_min

        # Growing degree days (base 10C, cap 30C)
        temp_clamped = temp.clamp(min=10, max=30) - 10
        gdd = temp_clamped.sum(dim=1) / 24  # Daily GDD

        # Radiation features
        rad = hourly_weather[:, :, self.rad_idx]
        rad_sum = rad.sum(dim=1)
        rad_max = rad.max(dim=1)[0]
        rad_mean = rad.sum(dim=1) / valid_hours.squeeze(-1) if mask is not None else rad.mean(dim=1)

        # VPD features
        vpd = hourly_weather[:, :, self.vpd_idx]
        vpd_sum = vpd.sum(dim=1)
        vpd_max = vpd.max(dim=1)[0]
        vpd_mean = vpd.sum(dim=1) / valid_hours.squeeze(-1) if mask is not None else vpd.mean(dim=1)

        # Other features: just means
        other_features = []
        for i in range(hourly_weather.shape[2]):
            if i not in [self.temp_idx, self.rad_idx, self.vpd_idx]:
                feat = hourly_weather[:, :, i]
                feat_mean = feat.sum(dim=1) / valid_hours.squeeze(-1) if mask is not None else feat.mean(dim=1)
                other_features.append(feat_mean)

        # Concatenate all derived features
        derived = torch.stack([
                                  temp_min, temp_max, temp_mean, temp_range, gdd, gdd,  # 6 temp features
                                  rad_sum, rad_max, rad_mean,  # 3 rad features
                                  vpd_sum, vpd_max, vpd_mean,  # 3 vpd features
                              ] + other_features, dim=-1)

        # Project to output dimension
        output = self.projection(derived)
        output = self.norm(output)

        return output