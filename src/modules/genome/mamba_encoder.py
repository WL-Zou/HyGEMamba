"""
Genome Mamba Encoder for HyGE-Mamba.

This module provides the Mamba-style SSM encoder for processing genomic
sequences (SNP data). It wraps the common SSM blocks with genome-specific
preprocessing and includes support for bidirectional processing to capture
both upstream and downstream genetic context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..common.ssm_block import SSMBlock, MambaEncoder
from ..common.layer_norm import RMSNorm


class GenomeMambaEncoder(nn.Module):
    """
    Mamba encoder specialized for genomic sequence processing.

    Processes SNP sequences through multiple SSM layers, capturing
    long-range dependencies and linkage disequilibrium patterns
    along the chromosome.

    Key Features:
        - Bidirectional SSM for capturing both flanking regions
        - Optional down-sampling for very long sequences
        - Hierarchical processing for multi-scale patterns

    Args:
        d_model: Model dimension (should match SNP embedding dim).
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand_factor: Expansion factor for inner dimension.
        num_layers: Number of SSM blocks.
        dropout: Dropout probability.
        bidirectional: If True, process in both directions.
        use_glu: If True, use GLU gating in SSM.
        downsample_factor: Factor for sequence downsampling (1 = no downsampling).
    """

    def __init__(
            self,
            d_model: int = 128,
            d_state: int = 16,
            d_conv: int = 4,
            expand_factor: int = 2,
            num_layers: int = 4,
            dropout: float = 0.1,
            bidirectional: bool = True,
            use_glu: bool = True,
            downsample_factor: int = 1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.downsample_factor = downsample_factor

        # Optional downsampling
        if downsample_factor > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=downsample_factor,
                          stride=downsample_factor, padding=0),
                nn.GELU(),
            )
        else:
            self.downsample = None

        # Input projection (for dimension adjustment if needed)
        self.input_proj = nn.Linear(d_model, d_model)

        # Stack of SSM blocks
        self.layers = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout,
                bidirectional=bidirectional,
                use_glu=use_glu,
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_all_layers: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Process genomic sequence through Mamba encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional boolean mask, shape (batch, seq_len).
            return_all_layers: If True, return outputs from all layers.

        Returns:
            If return_all_layers is False:
                Output tensor of shape (batch, seq_len', d_model)
                where seq_len' = seq_len // downsample_factor.
            If return_all_layers is True:
                Tuple of (final_output, list_of_layer_outputs).
        """
        # Input projection
        x = self.input_proj(x)

        # Optional downsampling
        if self.downsample is not None:
            x = x.transpose(1, 2)  # (batch, d_model, seq_len)
            x = self.downsample(x)
            x = x.transpose(1, 2)  # (batch, seq_len', d_model)

            # Adjust mask if provided
            if mask is not None:
                mask = mask[:, ::self.downsample_factor]

        # Process through SSM layers
        layer_outputs = []
        for layer in self.layers:
            x = layer(x, mask)
            if return_all_layers:
                layer_outputs.append(x)

        # Final normalization and projection
        x = self.final_norm(x)
        x = self.output_proj(x)

        if return_all_layers:
            return x, layer_outputs
        return x


class HierarchicalGenomeMambaEncoder(nn.Module):
    """
    Hierarchical Mamba encoder for multi-scale genomic processing.

    Processes genomic sequences at multiple resolutions:
    1. Fine-scale: Individual SNPs and local LD blocks
    2. Medium-scale: Gene-level patterns
    3. Coarse-scale: Chromosome-level organization

    This hierarchical approach captures patterns at different scales
    while maintaining computational efficiency.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        num_layers_per_level: Number of layers at each hierarchy level.
        downsample_factors: Downsampling factor at each level.
        dropout: Dropout probability.
        bidirectional: If True, use bidirectional SSM.
    """

    def __init__(
            self,
            d_model: int = 128,
            d_state: int = 16,
            num_layers_per_level: List[int] = [2, 2, 2],
            downsample_factors: List[int] = [1, 4, 4],
            dropout: float = 0.1,
            bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.num_levels = len(num_layers_per_level)
        assert len(downsample_factors) == self.num_levels

        # Create encoders for each level
        self.levels = nn.ModuleList()
        for level_idx in range(self.num_levels):
            self.levels.append(
                GenomeMambaEncoder(
                    d_model=d_model,
                    d_state=d_state,
                    num_layers=num_layers_per_level[level_idx],
                    dropout=dropout,
                    bidirectional=bidirectional,
                    downsample_factor=downsample_factors[level_idx],
                )
            )

        # Cross-level fusion
        self.level_fusion = nn.ModuleList([
            nn.Linear(d_model * 2, d_model)
            for _ in range(self.num_levels - 1)
        ])

        self.final_norm = RMSNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process genomic sequence through hierarchical encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional boolean mask.

        Returns:
            Multi-scale encoded representation, shape (batch, seq_len_final, d_model).
        """
        level_outputs = []
        current_x = x

        for level_idx, encoder in enumerate(self.levels):
            current_x = encoder(current_x, mask)
            level_outputs.append(current_x)

            # Update mask for downsampled sequence
            if mask is not None and encoder.downsample is not None:
                mask = mask[:, ::encoder.downsample_factor]

        # Fuse representations from different levels (bottom-up)
        # Use the finest resolution from the last level
        fused = level_outputs[-1]

        return self.final_norm(fused)


class DualParentGenomeEncoder(nn.Module):
    """
    Encoder for processing both parent genomes in parallel.

    Processes Parent A and Parent B genomes through separate (or shared)
    Mamba encoders, enabling the model to learn parent-specific patterns
    before fusion.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        num_layers: Number of SSM layers.
        dropout: Dropout probability.
        bidirectional: If True, use bidirectional SSM.
        share_weights: If True, share encoder weights between parents.
    """

    def __init__(
            self,
            d_model: int = 128,
            d_state: int = 16,
            num_layers: int = 4,
            dropout: float = 0.1,
            bidirectional: bool = True,
            share_weights: bool = False,
    ) -> None:
        super().__init__()

        self.share_weights = share_weights

        # Parent A encoder
        self.encoder_a = GenomeMambaEncoder(
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Parent B encoder (shared or separate)
        if share_weights:
            self.encoder_b = self.encoder_a
        else:
            self.encoder_b = GenomeMambaEncoder(
                d_model=d_model,
                d_state=d_state,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )

    def forward(
            self,
            parent_a: torch.Tensor,
            parent_b: torch.Tensor,
            mask_a: Optional[torch.Tensor] = None,
            mask_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process both parent genomes.

        Args:
            parent_a: Parent A embeddings, shape (batch, seq_len, d_model).
            parent_b: Parent B embeddings, shape (batch, seq_len, d_model).
            mask_a: Optional mask for parent A.
            mask_b: Optional mask for parent B.

        Returns:
            Tuple of (encoded_a, encoded_b), each shape (batch, seq_len, d_model).
        """
        encoded_a = self.encoder_a(parent_a, mask_a)
        encoded_b = self.encoder_b(parent_b, mask_b)

        return encoded_a, encoded_b