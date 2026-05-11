"""
Genome processing branch modules for HyGE-Mamba.

This package contains modules for processing dual-parent VCF genotype data,
including SNP embedding, position encoding, Mamba encoders, and hybrid
genetic fusion operations.
"""

from .snp_embedding import SNPEmbedding
from .position_encoder import PositionEncoder, ChromosomePositionEncoder
from .mamba_encoder import GenomeMambaEncoder
from .segment_router import SegmentRouter
from .hybrid_fusion import HybridFusion, AdditiveFusion, DominanceFusion, InteractionFusion
from .recombination_gate import RecombinationGate
from .epistasis_mixer import EpistasisMixer
from .genome_branch import GenomeBranch

__all__ = [
    "SNPEmbedding",
    "PositionEncoder",
    "ChromosomePositionEncoder",
    "GenomeMambaEncoder",
    "SegmentRouter",
    "HybridFusion",
    "AdditiveFusion",
    "DominanceFusion",
    "InteractionFusion",
    "RecombinationGate",
    "EpistasisMixer",
    "GenomeBranch",
]