"""
SNP Embedding module for HyGE-Mamba.

This module handles the embedding of SNP (Single Nucleotide Polymorphism)
genotype values from VCF files. The standard encoding uses 0-9 values:
    0: Homozygous reference (0/0)
    1: Heterozygous (0/1 or 1/0)
    2: Homozygous alternate (1/1)
    3-9: Multi-allelic variants or special encodings

The embedding layer learns dense representations for each genotype value,
capturing the genetic information in a continuous vector space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SNPEmbedding(nn.Module):
    """
    Embedding layer for SNP genotype values.

    Converts discrete genotype values (0-9) into dense vector representations.
    Optionally includes projection and normalization layers.

    Biological Interpretation:
        - Different genotype values (0/0, 0/1, 1/1) have distinct effects
        - The embedding learns to represent these effects in a continuous space
        - Multi-allelic variants (3-9) capture complex genetic variation

    Args:
        num_allele_values: Number of possible genotype values (default 10 for 0-9).
        embedding_dim: Dimension of the embedding vectors.
        padding_idx: Index used for padding (typically not used for SNPs).
        dropout: Dropout probability applied after embedding.
        use_layer_norm: If True, apply layer normalization after embedding.
        scale_grad_by_freq: If True, scale gradients by inverse frequency.
    """

    def __init__(
            self,
            num_allele_values: int = 10,
            embedding_dim: int = 64,
            padding_idx: Optional[int] = None,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
            scale_grad_by_freq: bool = False,
    ) -> None:
        super().__init__()

        self.num_allele_values = num_allele_values
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Main embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_allele_values,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
        )

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim) if use_layer_norm else nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding weights with truncated normal distribution."""
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        if self.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.padding_idx])

    def forward(
            self,
            genotypes: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed SNP genotype values.

        Args:
            genotypes: Tensor of genotype values, shape (batch, seq_len).
                       Values should be integers in range [0, num_allele_values).
            mask: Optional boolean mask, shape (batch, seq_len).
                  True indicates valid positions, False indicates padding.

        Returns:
            Embedded genotypes, shape (batch, seq_len, embedding_dim).
        """
        # Clamp values to valid range (defensive)
        genotypes = genotypes.clamp(0, self.num_allele_values - 1)

        # Embed
        embeddings = self.embedding(genotypes)

        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Apply mask if provided
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1).float()

        return embeddings

    def extra_repr(self) -> str:
        return (
            f"num_allele_values={self.num_allele_values}, "
            f"embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}"
        )


class DosageEmbedding(nn.Module):
    """
    Alternative embedding for continuous dosage values.

    Instead of discrete genotype values, this handles continuous dosage
    (typically 0.0 to 2.0 representing the expected allele count from
    imputation or probabilistic genotype calling).

    Args:
        embedding_dim: Output embedding dimension.
        hidden_dim: Hidden dimension for the MLP.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            embedding_dim: int = 64,
            hidden_dim: int = 32,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        # MLP to project continuous dosage to embedding space
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
            self,
            dosage: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed continuous dosage values.

        Args:
            dosage: Tensor of dosage values, shape (batch, seq_len).
                    Values typically in range [0.0, 2.0].
            mask: Optional boolean mask.

        Returns:
            Embedded dosage, shape (batch, seq_len, embedding_dim).
        """
        # Add feature dimension
        dosage = dosage.unsqueeze(-1)  # (batch, seq_len, 1)

        # Project through MLP
        embeddings = self.mlp(dosage)

        # Apply mask if provided
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1).float()

        return embeddings


class HybridSNPEmbedding(nn.Module):
    """
    Combined embedding for both discrete genotypes and continuous features.

    This is useful when SNP data includes both genotype calls and additional
    features like quality scores, allele frequencies, or functional annotations.

    Args:
        num_allele_values: Number of discrete genotype values.
        num_continuous_features: Number of continuous features per SNP.
        embedding_dim: Output embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            num_allele_values: int = 10,
            num_continuous_features: int = 0,
            embedding_dim: int = 64,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_continuous_features = num_continuous_features
        self.embedding_dim = embedding_dim

        # Discrete genotype embedding
        self.genotype_embedding = SNPEmbedding(
            num_allele_values=num_allele_values,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_layer_norm=False,  # Will apply after fusion
        )

        # Continuous feature projection (if any)
        if num_continuous_features > 0:
            self.continuous_proj = nn.Linear(num_continuous_features, embedding_dim)
            self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            genotypes: torch.Tensor,
            continuous_features: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed SNP data with optional continuous features.

        Args:
            genotypes: Discrete genotype values, shape (batch, seq_len).
            continuous_features: Optional continuous features,
                                 shape (batch, seq_len, num_continuous_features).
            mask: Optional boolean mask.

        Returns:
            Combined embeddings, shape (batch, seq_len, embedding_dim).
        """
        # Embed discrete genotypes
        embeddings = self.genotype_embedding(genotypes)

        # Fuse with continuous features if provided
        if continuous_features is not None and self.num_continuous_features > 0:
            cont_emb = self.continuous_proj(continuous_features)
            embeddings = self.fusion(torch.cat([embeddings, cont_emb], dim=-1))

        # Final normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Apply mask if provided
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1).float()

        return embeddings