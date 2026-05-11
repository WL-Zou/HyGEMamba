"""
GE Fusion modules for HyGE-Mamba.

This package contains modules for fusing genetic and environmental
representations to capture genotype-by-environment interactions.
"""

from .gxe_fusion import GxEFusion, GatedCrossAttention, BilinearFusion, FiLMFusion

__all__ = [
    "GxEFusion",
    "GatedCrossAttention",
    "BilinearFusion",
    "FiLMFusion",
]