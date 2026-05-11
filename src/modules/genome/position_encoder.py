"""
Position Encoder module for HyGE-Mamba.

This module encodes chromosomal positions of SNPs, combining:
1. Chromosome identity (which chromosome the SNP is on)
2. Base pair position (continuous position within the chromosome)

The encoding captures both the discrete chromosome structure and the
continuous spatial relationships between SNPs along the genome.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal position encoding for continuous positions.

    Uses sine and cosine functions at different frequencies to encode
    continuous position values, similar to the original Transformer
    but adapted for genomic coordinates.

    Args:
        dim: Encoding dimension (should be even).
        max_position: Maximum position value for normalization.
        base: Base for the exponential frequency scaling.
    """

    def __init__(
            self,
            dim: int,
            max_position: int = 300_000_000,
            base: float = 10000.0,
    ) -> None:
        super().__init__()

        assert dim % 2 == 0, f"dim must be even, got {dim}"

        self.dim = dim
        self.max_position = max_position
        self.base = base

        # Precompute frequency bands
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(base) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode positions using sinusoidal functions.

        Args:
            positions: Position values, shape (...).
                       Can be any shape, encoding applied element-wise.

        Returns:
            Position encodings, shape (..., dim).
        """
        # Normalize positions
        positions = positions.float() / self.max_position

        # Compute angles: position * frequency
        angles = positions.unsqueeze(-1) * self.freqs  # (..., half_dim)

        # Concatenate sin and cos
        encodings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return encodings


class LearnablePositionEncoding(nn.Module):
    """
    Learnable position encoding using binning and embedding.

    Divides the position range into bins and learns embeddings for each bin,
    with interpolation for positions between bin centers.

    Args:
        dim: Encoding dimension.
        max_position: Maximum position value.
        num_bins: Number of position bins.
    """

    def __init__(
            self,
            dim: int,
            max_position: int = 300_000_000,
            num_bins: int = 1000,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_position = max_position
        self.num_bins = num_bins
        self.bin_size = max_position / num_bins

        # Learnable embeddings for each bin
        self.embeddings = nn.Embedding(num_bins + 1, dim)

        # Initialize with small values
        nn.init.trunc_normal_(self.embeddings.weight, std=0.02)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode positions using learnable bin embeddings.

        Args:
            positions: Position values, shape (...).

        Returns:
            Position encodings, shape (..., dim).
        """
        # Compute bin indices
        positions = positions.float()
        bin_indices = (positions / self.bin_size).long().clamp(0, self.num_bins)

        # Look up embeddings
        encodings = self.embeddings(bin_indices)

        return encodings


class ChromosomePositionEncoder(nn.Module):
    """
    Complete position encoder combining chromosome and base pair position.

    This encoder creates a unified representation that captures:
    1. Chromosome identity (discrete, 1-10 for maize)
    2. Position within chromosome (continuous, 0 to ~300Mb)

    The two components are combined through concatenation and projection.

    Args:
        num_chromosomes: Number of chromosomes (10 for maize).
        chromosome_embedding_dim: Dimension for chromosome embeddings.
        position_encoding_dim: Dimension for position encodings.
        max_position: Maximum base pair position.
        use_sinusoidal: If True, use sinusoidal encoding for positions.
        learnable_chromosomes: If True, learn chromosome embeddings.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            num_chromosomes: int = 10,
            chromosome_embedding_dim: int = 32,
            position_encoding_dim: int = 32,
            max_position: int = 300_000_000,
            use_sinusoidal: bool = True,
            learnable_chromosomes: bool = True,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_chromosomes = num_chromosomes
        self.chromosome_embedding_dim = chromosome_embedding_dim
        self.position_encoding_dim = position_encoding_dim
        self.output_dim = chromosome_embedding_dim + position_encoding_dim

        # Chromosome embedding
        if learnable_chromosomes:
            self.chromosome_embedding = nn.Embedding(
                num_embeddings=num_chromosomes + 1,  # +1 for unknown/padding
                embedding_dim=chromosome_embedding_dim,
            )
            nn.init.trunc_normal_(self.chromosome_embedding.weight, std=0.02)
        else:
            # Use one-hot encoding
            self.register_buffer(
                "chromosome_onehot",
                torch.eye(num_chromosomes + 1)[:, :chromosome_embedding_dim],
            )
            self.chromosome_embedding = None

        # Position encoding
        if use_sinusoidal:
            self.position_encoder = SinusoidalPositionEncoding(
                dim=position_encoding_dim,
                max_position=max_position,
            )
        else:
            self.position_encoder = LearnablePositionEncoding(
                dim=position_encoding_dim,
                max_position=max_position,
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            chromosomes: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode chromosome and position information.

        Args:
            chromosomes: Chromosome indices, shape (batch, seq_len).
                         Values in range [0, num_chromosomes].
            positions: Base pair positions, shape (batch, seq_len).
                       Values in range [0, max_position].

        Returns:
            Combined encodings, shape (batch, seq_len, output_dim).
        """
        # Encode chromosomes
        if self.chromosome_embedding is not None:
            chr_enc = self.chromosome_embedding(chromosomes.clamp(0, self.num_chromosomes))
        else:
            chr_enc = F.embedding(
                chromosomes.clamp(0, self.num_chromosomes),
                self.chromosome_onehot,
            )

        # Encode positions
        pos_enc = self.position_encoder(positions)

        # Concatenate
        combined = torch.cat([chr_enc, pos_enc], dim=-1)

        return self.dropout(combined)


class PositionEncoder(nn.Module):
    """
    Full position encoder with projection to target dimension.

    Wraps ChromosomePositionEncoder and projects to the desired output
    dimension, matching the SNP embedding dimension for addition/concatenation.

    Args:
        num_chromosomes: Number of chromosomes.
        chromosome_embedding_dim: Dimension for chromosome embeddings.
        position_encoding_dim: Dimension for position encodings.
        output_dim: Final output dimension after projection.
        max_position: Maximum base pair position.
        use_sinusoidal: If True, use sinusoidal encoding for positions.
        learnable_chromosomes: If True, learn chromosome embeddings.
        dropout: Dropout probability.
        fusion_method: How to combine with SNP embeddings ('add' or 'concat').
    """

    def __init__(
            self,
            num_chromosomes: int = 10,
            chromosome_embedding_dim: int = 32,
            position_encoding_dim: int = 32,
            output_dim: int = 64,
            max_position: int = 300_000_000,
            use_sinusoidal: bool = True,
            learnable_chromosomes: bool = True,
            dropout: float = 0.1,
            fusion_method: str = "add",
    ) -> None:
        super().__init__()

        self.fusion_method = fusion_method
        self.output_dim = output_dim

        # Core encoder
        self.encoder = ChromosomePositionEncoder(
            num_chromosomes=num_chromosomes,
            chromosome_embedding_dim=chromosome_embedding_dim,
            position_encoding_dim=position_encoding_dim,
            max_position=max_position,
            use_sinusoidal=use_sinusoidal,
            learnable_chromosomes=learnable_chromosomes,
            dropout=dropout,
        )

        # Projection to output dimension
        encoder_output_dim = chromosome_embedding_dim + position_encoding_dim
        self.projection = nn.Linear(encoder_output_dim, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
            self,
            chromosomes: torch.Tensor,
            positions: torch.Tensor,
            snp_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode positions and optionally combine with SNP embeddings.

        Args:
            chromosomes: Chromosome indices, shape (batch, seq_len).
            positions: Base pair positions, shape (batch, seq_len).
            snp_embeddings: Optional SNP embeddings to combine with,
                           shape (batch, seq_len, output_dim).

        Returns:
            Position encodings (or combined embeddings),
            shape (batch, seq_len, output_dim).
        """
        # Get position encodings
        pos_enc = self.encoder(chromosomes, positions)
        pos_enc = self.projection(pos_enc)
        pos_enc = self.layer_norm(pos_enc)

        # Combine with SNP embeddings if provided
        if snp_embeddings is not None:
            if self.fusion_method == "add":
                return snp_embeddings + pos_enc
            elif self.fusion_method == "concat":
                return torch.cat([snp_embeddings, pos_enc], dim=-1)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return pos_enc


class RelativePositionEncoder(nn.Module):
    """
    Relative position encoding for SNP pairs.

    Encodes the relative distance between SNPs, useful for capturing
    linkage disequilibrium patterns. Distance can be in base pairs
    or genetic distance (cM).

    Args:
        dim: Encoding dimension.
        max_distance: Maximum distance for encoding.
        num_buckets: Number of distance buckets for log-scale encoding.
    """

    def __init__(
            self,
            dim: int,
            max_distance: int = 10_000_000,  # 10 Mb
            num_buckets: int = 64,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_distance = max_distance
        self.num_buckets = num_buckets

        # Learnable embeddings for distance buckets
        self.embeddings = nn.Embedding(num_buckets * 2 + 1, dim)  # +/- directions

        nn.init.trunc_normal_(self.embeddings.weight, std=0.02)

    def _get_bucket(self, relative_distance: torch.Tensor) -> torch.Tensor:
        """Convert relative distance to bucket index."""
        # Use logarithmic bucketing for better resolution at small distances
        sign = torch.sign(relative_distance)
        abs_dist = relative_distance.abs().float()

        # Log-scale bucketing
        log_dist = torch.log(abs_dist.clamp(min=1)) / math.log(self.max_distance)
        bucket = (log_dist * self.num_buckets).long().clamp(0, self.num_buckets - 1)

        # Add offset for direction
        bucket = bucket + self.num_buckets  # Center at num_buckets
        bucket = bucket + sign.long() * (bucket != self.num_buckets).long()

        return bucket.clamp(0, self.num_buckets * 2)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute relative position encodings.

        Args:
            positions: Absolute positions, shape (batch, seq_len).

        Returns:
            Relative position encodings, shape (batch, seq_len, seq_len, dim).
        """
        # Compute pairwise relative distances
        # positions: (batch, seq_len)
        rel_dist = positions.unsqueeze(-1) - positions.unsqueeze(-2)  # (batch, seq_len, seq_len)

        # Convert to buckets
        buckets = self._get_bucket(rel_dist)

        # Look up embeddings
        encodings = self.embeddings(buckets)

        return encodings