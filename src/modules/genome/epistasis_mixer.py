"""
Epistasis Mixer module for HyGE-Mamba.

This module models epistasis (gene-gene interactions) where the effect
of one gene depends on the genotype at other genes. Epistasis is a
major source of genetic variation in complex traits and is particularly
important for heterosis (hybrid vigor) in crops.

The mixer uses SSM-style blocks to capture long-range epistatic
interactions across the genome, allowing genes on different chromosomes
to influence each other's effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..common.ssm_block import SSMBlock
from ..common.layer_norm import RMSNorm
from ..common.glu_ffn import SwiGLUFFN


class EpistasisMixer(nn.Module):
    """
    Epistasis mixer using SSM blocks for long-range gene interactions.

    Models gene-gene interactions by allowing information to flow
    between genomic segments, capturing epistatic effects where
    alleles at different loci interact to determine phenotype.

    Key biological concepts:
        - Digenic epistasis: Interaction between two genes
        - Higher-order epistasis: Multi-gene interactions
        - Cis-epistasis: Interactions within same chromosome
        - Trans-epistasis: Interactions across chromosomes

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        num_layers: Number of SSM layers for mixing.
        dropout: Dropout probability.
        use_residual: If True, use residual connections.
        use_cross_chromosome: If True, explicitly model cross-chromosome interactions.
    """

    def __init__(
            self,
            d_model: int = 128,
            d_state: int = 16,
            num_layers: int = 2,
            dropout: float = 0.1,
            use_residual: bool = True,
            use_cross_chromosome: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_cross_chromosome = use_cross_chromosome

        # SSM layers for sequential epistatic interactions
        self.ssm_layers = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                bidirectional=True,  # Bidirectional for symmetric interactions
            )
            for _ in range(num_layers)
        ])

        # Feed-forward layers for local transformations
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                RMSNorm(d_model),
                SwiGLUFFN(d_model, d_model * 2, d_model, dropout),
            )
            for _ in range(num_layers)
        ])

        # Optional cross-chromosome interaction layer
        if use_cross_chromosome:
            self.cross_chr_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_chr_norm = RMSNorm(d_model)

        # Output projection
        self.output_proj = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(
            self,
            x: torch.Tensor,
            chromosome_ids: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply epistasis mixing to genomic representation.

        Args:
            x: Input tensor of shape (batch, num_segments, d_model).
            chromosome_ids: Optional chromosome IDs for segments,
                           shape (batch, num_segments).
            mask: Optional boolean mask.

        Returns:
            Epistasis-mixed representation, shape (batch, num_segments, d_model).
        """
        # Process through SSM layers
        for ssm_layer, ffn_layer in zip(self.ssm_layers, self.ffn_layers):
            # SSM for long-range interactions
            if self.use_residual:
                x = x + ssm_layer(x, mask)
            else:
                x = ssm_layer(x, mask)

            # FFN for local transformation
            if self.use_residual:
                x = x + ffn_layer(x)
            else:
                x = ffn_layer(x)

        # Optional cross-chromosome attention
        if self.use_cross_chromosome and chromosome_ids is not None:
            x = self._cross_chromosome_interaction(x, chromosome_ids, mask)

        # Final projection
        output = self.output_proj(x)

        return output

    def _cross_chromosome_interaction(
            self,
            x: torch.Tensor,
            chromosome_ids: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Model explicit cross-chromosome epistatic interactions.

        Uses attention where segments can attend to segments on
        different chromosomes, modeling trans-epistasis.
        """
        # Create cross-chromosome attention mask
        # Segments can only attend to different chromosomes
        chr_match = chromosome_ids.unsqueeze(-1) == chromosome_ids.unsqueeze(-2)
        cross_chr_mask = ~chr_match  # Attend to different chromosomes

        # Combine with padding mask if provided
        if mask is not None:
            padding_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            attn_mask = cross_chr_mask & padding_mask
        else:
            attn_mask = cross_chr_mask

        # Convert to attention mask format (True = masked out)
        attn_mask = ~attn_mask

        # Apply cross-chromosome attention
        x_norm = self.cross_chr_norm(x)
        attn_out, _ = self.cross_chr_attention(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask.float().masked_fill(attn_mask, float("-inf"))[:, None, :, :].expand(-1, 4, -1,
                                                                                                    -1).reshape(-1,
                                                                                                                x.size(
                                                                                                                    1),
                                                                                                                x.size(
                                                                                                                    1)) if attn_mask.any() else None,
        )

        return x + attn_out


class PairwiseEpistasisLayer(nn.Module):
    """
    Pairwise epistasis computation layer.

    Explicitly computes pairwise interactions between genomic segments,
    modeling digenic epistasis. This is more interpretable but scales
    quadratically with sequence length.

    Args:
        d_model: Model dimension.
        hidden_dim: Hidden dimension for interaction MLP.
        num_heads: Number of attention heads for interaction weighting.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            d_model: int = 128,
            hidden_dim: int = 256,
            num_heads: int = 4,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Bilinear interaction
        self.bilinear = nn.Bilinear(d_model, d_model, hidden_dim)

        # Interaction MLP
        self.interaction_mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        # Attention for aggregating interactions
        self.interaction_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = RMSNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute pairwise epistatic interactions.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional boolean mask.

        Returns:
            Epistasis-enhanced representation, shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # Compute pairwise interactions (expensive for long sequences)
        # x_i: (batch, seq_len, 1, d_model)
        # x_j: (batch, 1, seq_len, d_model)
        x_i = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        x_j = x.unsqueeze(1).expand(-1, seq_len, -1, -1)

        # Bilinear interaction
        x_i_flat = x_i.reshape(batch_size * seq_len * seq_len, self.d_model)
        x_j_flat = x_j.reshape(batch_size * seq_len * seq_len, self.d_model)
        interactions = self.bilinear(x_i_flat, x_j_flat)
        interactions = self.interaction_mlp(interactions)
        interactions = interactions.reshape(batch_size, seq_len, seq_len, self.d_model)

        # Aggregate interactions for each position
        # Sum interactions involving each position
        interaction_sum = interactions.sum(dim=2) + interactions.sum(dim=1)
        interaction_sum = interaction_sum / (2 * seq_len)  # Normalize

        # Combine with original
        output = self.norm(x + interaction_sum)

        return output


class HierarchicalEpistasisMixer(nn.Module):
    """
    Hierarchical epistasis mixer for multi-scale interactions.

    Models epistasis at multiple scales:
    1. Local: Within-gene and nearby gene interactions
    2. Regional: Within-chromosome interactions
    3. Global: Cross-chromosome interactions

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            d_model: int = 128,
            d_state: int = 16,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Local interactions (small receptive field)
        self.local_mixer = SSMBlock(
            d_model=d_model,
            d_state=d_state // 2,
            d_conv=2,  # Small convolution
            dropout=dropout,
            bidirectional=True,
        )

        # Regional interactions (medium receptive field)
        self.regional_mixer = SSMBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            dropout=dropout,
            bidirectional=True,
        )

        # Global interactions (large receptive field)
        self.global_mixer = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Fusion of multi-scale interactions
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(3)])
        self.output_norm = RMSNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply hierarchical epistasis mixing.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional boolean mask.

        Returns:
            Multi-scale epistasis representation, shape (batch, seq_len, d_model).
        """
        # Local interactions
        local_out = self.local_mixer(self.norms[0](x), mask)

        # Regional interactions
        regional_out = self.regional_mixer(self.norms[1](x + local_out), mask)

        # Global interactions
        global_x = self.norms[2](x + local_out + regional_out)
        global_out, _ = self.global_mixer(
            global_x, global_x, global_x,
            key_padding_mask=~mask if mask is not None else None,
        )

        # Fuse multi-scale representations
        multi_scale = torch.cat([local_out, regional_out, global_out], dim=-1)
        fused = self.fusion(multi_scale)

        # Residual connection
        output = self.output_norm(x + fused)

        return output