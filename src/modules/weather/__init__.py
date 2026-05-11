"""
Weather processing branch modules for HyGE-Mamba.

This package contains modules for processing hierarchical weather time
series data, capturing both intra-day (hourly) and inter-day (daily/seasonal)
patterns that affect crop growth.
"""

from .intra_day_encoder import IntraDayEncoder, DiurnalFeatureExtractor, HourlyPositionEncoding
from .inter_day_encoder import InterDayEncoder, SeasonalPositionEncoding, GrowthStageEmbedding
from .weather_branch import WeatherBranch, SimpleWeatherBranch, WeatherBranchOutput

__all__ = [
    "IntraDayEncoder",
    "DiurnalFeatureExtractor",
    "HourlyPositionEncoding",
    "InterDayEncoder",
    "SeasonalPositionEncoding",
    "GrowthStageEmbedding",
    "WeatherBranch",
    "SimpleWeatherBranch",
    "WeatherBranchOutput",
]