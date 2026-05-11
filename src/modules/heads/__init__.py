"""
Prediction head modules for HyGE-Mamba.

This package contains output heads for:
- APSIM crop model parameters
- Time-resolved growth curves
- Scalar trait predictions
"""

from .apsim_head import APSIMHead, APSIMParameterRanges
from .curve_decoder import CurveDecoder, GrowthCurveDecoder
from .scalar_head import ScalarHead, MultiTaskScalarHead

__all__ = [
    "APSIMHead",
    "APSIMParameterRanges",
    "CurveDecoder",
    "GrowthCurveDecoder",
    "ScalarHead",
    "MultiTaskScalarHead",
]
