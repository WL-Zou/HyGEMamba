"""
GE (Genotype  Environment) Fusion module for HyGE-Mamba.

This module implements various fusion strategies for combining genetic
and environmental representations. GE interaction is crucial in crop
science as the same genotype can perform differently across environments.

Key GE concepts modeled:
- Crossover interactions: Genotype ranking changes across environments
- Non-crossover interactions: Magnitude of effects varies
- Environmental sensitivity: Some genotypes more responsive to environment
- Stability: Consistent performance across environments

Fusion methods:
1. Gated Cross-Attention: Environment modulates attention over genome
2. Bilinear Fusion: Learned bilinear interaction G^T W E
3. FiLM: Feature-wise Linear Modulation (environment modulates genome)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..common.layer_norm import RMSNorm
from ..common.glu_ffn import SwiGLUFFN


class GatedCrossAttention(nn.Module):
    """
    Gated cross-attention for GE fusion.

    Uses the environment representation to attend over genomic features,
    with gating to control the flow of information.

    The intuition is that environmental conditions determine which
    genomic regions are most relevant (e.g., drought stress activates
    genes for drought tolerance).

    Args:
        genome_dim: Dimension of genome representation.
        weather_dim: Dimension of weather representation.
        hidden_dim: Hidden dimension for attention.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        use_gate: If True, apply learned gating after attention.
    """

    def __init__(
            self,
            genome_dim: int = 128,
            weather_dim: int = 128,
            hidden_dim: int = 256,
            num_heads: int = 4,
            dropout: float = 0.1,
            use_gate: bool = True,
    ) -> None:
        super().__init__()

        self.genome_dim = genome_dim
        self.weather_dim = weather_dim
        self.hidden_dim = hidden_dim
        self.use_gate = use_gate

        # Project to common dimension
        self.genome_proj = nn.Linear(genome_dim, hidden_dim)
        self.weather_proj = nn.Linear(weather_dim, hidden_dim)

        # Cross-attention: weather queries, genome keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gating mechanism
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )

        self.norm = RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            genome: torch.Tensor,
            weather: torch.Tensor,
            genome_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply gated cross-attention fusion.

        Args:
            genome: Genome representation, shape (batch, genome_dim) or
                   (batch, num_segments, genome_dim).
            weather: Weather representation, shape (batch, weather_dim) or
                    (batch, num_days, weather_dim).
            genome_mask: Optional mask for genome segments.

        Returns:
            Fused representation, shape (batch, hidden_dim) or
            (batch, seq_len, hidden_dim).
        """
        # Handle 2D inputs (global representations)
        genome_is_2d = genome.dim() == 2
        weather_is_2d = weather.dim() == 2

        if genome_is_2d:
            genome = genome.unsqueeze(1)
        if weather_is_2d:
            weather = weather.unsqueeze(1)

        # Project to common dimension
        genome_h = self.genome_proj(genome)
        weather_h = self.weather_proj(weather)

        # Cross-attention: weather attends to genome
        # Query: weather, Key/Value: genome
        attn_out, _ = self.cross_attn(
            weather_h, genome_h, genome_h,
            key_padding_mask=~genome_mask if genome_mask is not None else None,
        )

        # Apply gating
        if self.use_gate:
            # Pool weather for gating
            weather_pool = weather_h.mean(dim=1, keepdim=True).expand_as(attn_out)
            gate_input = torch.cat([attn_out, weather_pool], dim=-1)
            gate = self.gate_net(gate_input)
            attn_out = gate * attn_out

        output = self.norm(attn_out)
        output = self.dropout(output)

        # Restore original shape
        if genome_is_2d and weather_is_2d:
            output = output.squeeze(1)

        return output


class BilinearFusion(nn.Module):
    """
    Bilinear fusion for GE interaction.

    Computes: G^T W E (with optional decomposition for efficiency)

    This directly models the multiplicative interaction between
    genetic and environmental factors.

    Args:
        genome_dim: Dimension of genome representation.
        weather_dim: Dimension of weather representation.
        output_dim: Output dimension.
        rank: Rank for low-rank bilinear decomposition. If None, use full bilinear.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            genome_dim: int = 128,
            weather_dim: int = 128,
            output_dim: int = 256,
            rank: Optional[int] = None,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.genome_dim = genome_dim
        self.weather_dim = weather_dim
        self.output_dim = output_dim

        if rank is None:
            # Full bilinear
            self.bilinear = nn.Bilinear(genome_dim, weather_dim, output_dim)
        else:
            # Low-rank factorization: G^T U V^T E
            self.genome_proj = nn.Linear(genome_dim, rank)
            self.weather_proj = nn.Linear(weather_dim, rank)
            self.output_proj = nn.Linear(rank, output_dim)
            self.rank = rank

        self.norm = RMSNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            genome: torch.Tensor,
            weather: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply bilinear fusion.

        Args:
            genome: Genome representation, shape (batch, genome_dim).
            weather: Weather representation, shape (batch, weather_dim).

        Returns:
            Fused representation, shape (batch, output_dim).
        """
        if hasattr(self, "bilinear"):
            fused = self.bilinear(genome, weather)
        else:
            # Low-rank bilinear
            g_proj = self.genome_proj(genome)
            w_proj = self.weather_proj(weather)
            interaction = g_proj * w_proj  # Element-wise product
            fused = self.output_proj(interaction)

        output = self.norm(fused)
        output = self.dropout(output)

        return output


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for GE fusion.

    Environment modulates genome features through learned scale () and
    shift () parameters:
        output = (weather) * genome + (weather)

    This allows the environment to gate and shift genetic effects,
    modeling how environmental conditions modify gene expression.

    Args:
        genome_dim: Dimension of genome representation.
        weather_dim: Dimension of weather representation.
        output_dim: Output dimension.
        hidden_dim: Hidden dimension for modulation networks.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            genome_dim: int = 128,
            weather_dim: int = 128,
            output_dim: int = 256,
            hidden_dim: int = 128,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.genome_dim = genome_dim
        self.weather_dim = weather_dim
        self.output_dim = output_dim

        # Project genome to output dimension
        self.genome_proj = nn.Linear(genome_dim, output_dim)

        # Modulation networks (weather  , )
        self.gamma_net = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.norm = RMSNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            genome: torch.Tensor,
            weather: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM fusion.

        Args:
            genome: Genome representation, shape (batch, genome_dim).
            weather: Weather representation, shape (batch, weather_dim).

        Returns:
            Fused representation, shape (batch, output_dim).
        """
        # Project genome
        genome_h = self.genome_proj(genome)

        # Compute modulation parameters
        gamma = self.gamma_net(weather) + 1.0  # Center at 1 for stability
        beta = self.beta_net(weather)

        # Apply FiLM
        fused = gamma * genome_h + beta

        output = self.norm(fused)
        output = self.dropout(output)

        return output


class GxEFusion(nn.Module):
    """
    Complete GE Fusion module combining multiple fusion strategies.

    Implements a comprehensive fusion approach that combines:
    1. Gated cross-attention for selective information flow
    2. Bilinear/FiLM for multiplicative interactions
    3. Residual connections for stable training

    Args:
        genome_dim: Dimension of genome representation.
        weather_dim: Dimension of weather representation.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        num_heads: Number of attention heads.
        fusion_type: Primary fusion type ('gated_cross_attention', 'bilinear', 'film').
        dropout: Dropout probability.
        use_residual: If True, add residual connections.
    """

    def __init__(
            self,
            genome_dim: int = 128,
            weather_dim: int = 128,
            hidden_dim: int = 256,
            output_dim: int = 256,
            num_heads: int = 4,
            fusion_type: str = "gated_cross_attention",
            dropout: float = 0.1,
            use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.genome_dim = genome_dim
        self.weather_dim = weather_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.use_residual = use_residual

        # Primary fusion module
        if fusion_type == "gated_cross_attention":
            self.fusion = GatedCrossAttention(
                genome_dim=genome_dim,
                weather_dim=weather_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            fusion_out_dim = hidden_dim
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(
                genome_dim=genome_dim,
                weather_dim=weather_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
            fusion_out_dim = hidden_dim
        elif fusion_type == "film":
            self.fusion = FiLMFusion(
                genome_dim=genome_dim,
                weather_dim=weather_dim,
                output_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                dropout=dropout,
            )
            fusion_out_dim = hidden_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Combine with original representations
        if use_residual:
            self.genome_residual = nn.Linear(genome_dim, fusion_out_dim)
            self.weather_residual = nn.Linear(weather_dim, fusion_out_dim)
            combine_dim = fusion_out_dim * 3
        else:
            combine_dim = fusion_out_dim

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(combine_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # FFN for final processing
        self.ffn = SwiGLUFFN(output_dim, output_dim * 2, output_dim, dropout)

        self.norm1 = RMSNorm(output_dim)
        self.norm2 = RMSNorm(output_dim)

    def forward(
            self,
            genome: torch.Tensor,
            weather: torch.Tensor,
            genome_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply GE fusion.

        Args:
            genome: Genome representation, shape (batch, genome_dim) or
                   (batch, num_segments, genome_dim).
            weather: Weather representation, shape (batch, weather_dim) or
                    (batch, num_days, weather_dim).
            genome_mask: Optional mask for genome segments.

        Returns:
            Fused GE representation, shape (batch, output_dim).
        """
        # Handle sequence inputs by pooling
        if genome.dim() == 3:
            genome_global = genome.mean(dim=1)
        else:
            genome_global = genome

        if weather.dim() == 3:
            weather_global = weather.mean(dim=1)
        else:
            weather_global = weather

        # Apply primary fusion
        fused = self.fusion(genome_global, weather_global)

        # Add residual connections
        if self.use_residual:
            genome_res = self.genome_residual(genome_global)
            weather_res = self.weather_residual(weather_global)
            combined = torch.cat([fused, genome_res, weather_res], dim=-1)
        else:
            combined = fused

        # Output projection
        output = self.output_proj(combined)
        output = self.norm1(output)

        # FFN
        output = output + self.ffn(output)
        output = self.norm2(output)

        return output