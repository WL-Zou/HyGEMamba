"""
HyGE-Mamba: Hybrid Genotype-Environment Mamba Network

A mechanism-aware deep learning architecture for hybrid crop yield prediction
using dual-parent VCF genotype data and hierarchical weather time series.

Main components:
- models: HyGEMamba main model
- modules: Genome branch, Weather branch, GE fusion, MoE, Output heads
- data: Dataset classes and data loading utilities
- losses: Multi-task loss functions
- utils: Training utilities, checkpointing, logging
"""

__version__ = "0.1.0"
__author__ = "HyGE-Mamba Team"

# Lazy imports to avoid circular dependencies
def get_model():
    from src.models.hyge_mamba import HyGEMamba, HyGEMambaConfig, build_hyge_mamba
    return HyGEMamba, HyGEMambaConfig, build_hyge_mamba