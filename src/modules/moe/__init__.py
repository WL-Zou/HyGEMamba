"""
Mixture of Experts (MoE) modules for HyGE-Mamba.

This package implements sparse MoE with:
- Top-k routing for expert selection
- Load balancing auxiliary loss
- Shared + routed expert architecture
"""

from .router import Router, TopKRouter, NoisyTopKRouter
from .experts import Expert, SharedExpert, ExpertGroup
from .sparse_moe import SparseMoE, SparseMoELayer

__all__ = [
    "Router",
    "TopKRouter",
    "NoisyTopKRouter",
    "Expert",
    "SharedExpert",
    "ExpertGroup",
    "SparseMoE",
    "SparseMoELayer",
]
