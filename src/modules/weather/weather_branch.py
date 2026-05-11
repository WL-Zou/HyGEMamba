"""
Weather Branch module for HyGE-Mamba.

This module assembles all weather processing components into a complete
branch that processes hierarchical weather time series:

1. Intra-Day Encoder: Process hourly patterns within each day
2. Inter-Day Encoder: Process daily patterns across the growing season
3. Seasonal Pooling: Aggregate to season-level representation

The output is a weather representation for GE fusion with the genome.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

from .intra_day_encoder import IntraDayEncoder, DiurnalFeatureExtractor
from .inter_day_encoder import InterDayEncoder, SeasonalAttentionPooling, WeatherAnomalyDetector
from ..common.layer_norm import RMSNorm


@dataclass
class WeatherBranchOutput:
    """Output container for weather branch."""

    weather_repr: torch.Tensor  # Final weather representation
    daily_repr: torch.Tensor  # Daily-level representations
    seasonal_queries: Optional[torch.Tensor] = None  # Seasonal attention outputs
    anomaly_probs: Optional[torch.Tensor] = None  # Detected anomalies
    anomaly_severity: Optional[torch.Tensor] = None  # Anomaly severity scores


class WeatherBranch(nn.Module):
    """
    Complete weather processing branch for HyGE-Mamba.

    Processes hierarchical weather data through:
    hourly  daily  seasonal representations

    Args:
        input_features: Number of hourly weather features.
        intra_day_dim: Dimension for intra-day encoder output.
        inter_day_dim: Dimension for inter-day encoder output.
        output_dim: Final output dimension.
        intra_day_layers: Number of SSM layers in intra-day encoder.
        inter_day_layers: Number of SSM layers in inter-day encoder.
        hours_per_day: Number of hours per day (default 24).
        max_days: Maximum growing season length in days.
        dropout: Dropout probability.
        use_anomaly_detection: If True, include anomaly detection.
        use_diurnal_features: If True, extract explicit diurnal features.
    """

    def __init__(
            self,
            input_features: int = 8,
            intra_day_dim: int = 64,
            inter_day_dim: int = 128,
            output_dim: int = 128,
            intra_day_layers: int = 2,
            inter_day_layers: int = 3,
            hours_per_day: int = 24,
            max_days: int = 365,
            dropout: float = 0.1,
            use_anomaly_detection: bool = True,
            use_diurnal_features: bool = True,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.use_anomaly_detection = use_anomaly_detection
        self.use_diurnal_features = use_diurnal_features

        # 1. Intra-Day Encoder (hourly  daily)
        self.intra_day_encoder = IntraDayEncoder(
            input_features=input_features,
            d_model=intra_day_dim,
            d_state=8,
            num_layers=intra_day_layers,
            hours_per_day=hours_per_day,
            dropout=dropout,
            aggregation="attention",
        )

        # Optional diurnal feature extractor
        if use_diurnal_features:
            self.diurnal_extractor = DiurnalFeatureExtractor(
                input_features=input_features,
                output_dim=intra_day_dim // 2,
            )
            inter_day_input_dim = intra_day_dim + intra_day_dim // 2
        else:
            self.diurnal_extractor = None
            inter_day_input_dim = intra_day_dim

        # 2. Inter-Day Encoder (daily  seasonal)
        self.inter_day_encoder = InterDayEncoder(
            input_dim=inter_day_input_dim,
            d_model=inter_day_dim,
            d_state=16,
            num_layers=inter_day_layers,
            max_days=max_days,
            dropout=dropout,
            use_positional_encoding=True,
            use_growth_stage=True,
        )

        # 3. Seasonal Pooling
        self.seasonal_pooling = SeasonalAttentionPooling(
            d_model=inter_day_dim,
            num_queries=5,  # 5 growth period queries
            num_heads=4,
            dropout=dropout,
        )

        # 4. Optional Anomaly Detection
        if use_anomaly_detection:
            self.anomaly_detector = WeatherAnomalyDetector(
                input_dim=inter_day_dim,
                hidden_dim=inter_day_dim // 2,
                num_anomaly_types=5,
            )

        # 5. Output Projection
        self.output_proj = nn.Sequential(
            RMSNorm(inter_day_dim),
            nn.Linear(inter_day_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        # Global pooling for sequence representation
        self.global_pool = nn.Sequential(
            nn.Linear(inter_day_dim, inter_day_dim // 2),
            nn.GELU(),
            nn.Linear(inter_day_dim // 2, 1),
        )

        self.final_norm = RMSNorm(output_dim)

    def forward(
            self,
            hourly_weather: torch.Tensor,
            day_mask: Optional[torch.Tensor] = None,
            hour_mask: Optional[torch.Tensor] = None,
            day_indices: Optional[torch.Tensor] = None,
            growth_stages: Optional[torch.Tensor] = None,
            return_intermediates: bool = False,
    ) -> torch.Tensor | WeatherBranchOutput:
        """
        Process hierarchical weather data.

        Args:
            hourly_weather: Weather data, shape (batch, num_days, hours_per_day, features).
            day_mask: Optional mask for valid days, shape (batch, num_days).
            hour_mask: Optional mask for valid hours, shape (batch, num_days, hours_per_day).
            day_indices: Day-of-year indices, shape (batch, num_days).
            growth_stages: Growth stage indices, shape (batch, num_days).
            return_intermediates: If True, return detailed outputs.

        Returns:
            If return_intermediates is False:
                Final weather representation, shape (batch, output_dim).
            If return_intermediates is True:
                WeatherBranchOutput with all intermediate representations.
        """
        batch_size, num_days, hours_per_day, num_features = hourly_weather.shape

        # 1. Intra-day encoding: (batch, days, hours, feat)  (batch, days, intra_dim)
        daily_encoded = self.intra_day_encoder(hourly_weather, hour_mask)

        # Optional: Extract explicit diurnal features
        if self.use_diurnal_features:
            # Reshape for diurnal extraction
            hourly_flat = hourly_weather.reshape(batch_size * num_days, hours_per_day, num_features)
            hour_mask_flat = hour_mask.reshape(batch_size * num_days, hours_per_day) if hour_mask is not None else None

            diurnal_features = self.diurnal_extractor(hourly_flat, hour_mask_flat)
            diurnal_features = diurnal_features.reshape(batch_size, num_days, -1)

            # Concatenate with encoded daily representation
            daily_encoded = torch.cat([daily_encoded, diurnal_features], dim=-1)

        # 2. Inter-day encoding: (batch, days, daily_dim)  (batch, days, inter_dim)
        seasonal_encoded = self.inter_day_encoder(
            daily_encoded,
            day_indices=day_indices,
            growth_stages=growth_stages,
            mask=day_mask,
        )

        # 3. Anomaly detection (optional)
        anomaly_probs = None
        anomaly_severity = None
        if self.use_anomaly_detection:
            anomaly_probs, anomaly_severity = self.anomaly_detector(seasonal_encoded)

        # 4. Seasonal pooling
        seasonal_repr = self.seasonal_pooling(seasonal_encoded, day_mask)

        # 5. Output projection
        daily_repr = self.output_proj(seasonal_encoded)

        # 6. Global pooling for final representation
        pool_weights = F.softmax(
            self.global_pool(seasonal_encoded).squeeze(-1), dim=-1
        )
        if day_mask is not None:
            pool_weights = pool_weights * day_mask.float()
            pool_weights = pool_weights / pool_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        weather_repr = torch.einsum("bd,bdo->bo", pool_weights, daily_repr)
        weather_repr = self.final_norm(weather_repr)

        if return_intermediates:
            return WeatherBranchOutput(
                weather_repr=weather_repr,
                daily_repr=daily_repr,
                seasonal_queries=seasonal_repr,
                anomaly_probs=anomaly_probs,
                anomaly_severity=anomaly_severity,
            )

        return weather_repr


class SimpleWeatherBranch(nn.Module):
    """
    Simplified weather branch for when hourly data is not available.

    Processes daily weather summaries directly without the intra-day step.

    Args:
        input_features: Number of daily weather features.
        d_model: Model dimension.
        output_dim: Output dimension.
        num_layers: Number of SSM layers.
        max_days: Maximum growing season length.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            input_features: int = 12,  # Daily summary features
            d_model: int = 128,
            output_dim: int = 128,
            num_layers: int = 3,
            max_days: int = 365,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Inter-day encoder
        self.encoder = InterDayEncoder(
            input_dim=d_model,
            d_model=d_model,
            d_state=16,
            num_layers=num_layers,
            max_days=max_days,
            dropout=dropout,
        )

        # Pooling
        self.pooling = SeasonalAttentionPooling(
            d_model=d_model,
            num_queries=5,
            num_heads=4,
            dropout=dropout,
        )

        # Output
        self.output_proj = nn.Linear(d_model, output_dim)
        self.norm = RMSNorm(output_dim)

    def forward(
            self,
            daily_weather: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            day_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process daily weather data.

        Args:
            daily_weather: Daily weather features, shape (batch, num_days, features).
            mask: Optional mask for valid days.
            day_indices: Optional day-of-year indices.

        Returns:
            Weather representation, shape (batch, output_dim).
        """
        # Project input
        x = self.input_proj(daily_weather)

        # Encode
        encoded = self.encoder(x, day_indices=day_indices, mask=mask)

        # Pool
        pooled = self.pooling(encoded, mask)

        # Output
        output = self.output_proj(pooled)
        output = self.norm(output)

        return output


class WeatherBranchConfig:
    """Configuration helper for WeatherBranch."""

    @staticmethod
    def from_model_config(config: Any) -> Dict[str, Any]:
        """Convert ModelConfig.weather to WeatherBranch kwargs."""
        return {
            "input_features": config.intra_day.input_features,
            "intra_day_dim": config.intra_day.d_model,
            "inter_day_dim": config.inter_day.d_model,
            "output_dim": config.output_dim,
            "intra_day_layers": config.intra_day.num_layers,
            "inter_day_layers": config.inter_day.num_layers,
            "hours_per_day": config.intra_day.hours_per_day,
            "max_days": config.inter_day.max_days,
            "dropout": config.intra_day.dropout,
        }