"""
Hybrid Genetic Fusion module for HyGE-Mamba.

This module implements biologically-motivated fusion operations for
combining dual-parent genotype representations:

1. Additive Effects: (A + B) / 2
   - Represents the average contribution from both parents
   - Models quantitative trait inheritance

2. Dominance Effects: |A - B|
   - Captures deviation from purely additive inheritance
   - Important for heterosis (hybrid vigor)

3. Interaction Effects: A  B (element-wise product)
   - Models non-linear gene-gene interactions
   - Captures epistatic effects between parental alleles

These operations are inspired by classical quantitative genetics theory
and are combined through learned weights to predict hybrid phenotypes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class AdditiveFusion(nn.Module):
    """
    Additive genetic effect fusion.

    Computes the mid-parent value: (Parent_A + Parent_B) / 2

    This represents the expected hybrid value under purely additive
    inheritance, where each parent contributes equally.

    Args:
        input_dim: Dimension of parent representations.
        project: If True, apply learned projection after fusion.
    """

    def __init__(
            self,
            input_dim: int,
            project: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.project = project

        if project:
            self.projection = nn.Linear(input_dim, input_dim)
        else:
            self.projection = nn.Identity()

    def forward(
            self,
            parent_a: torch.Tensor,
            parent_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute additive fusion.

        Args:
            parent_a: Parent A representation, shape (..., input_dim).
            parent_b: Parent B representation, shape (..., input_dim).

        Returns:
            Additive fusion result, shape (..., input_dim).
        """
        additive = (parent_a + parent_b) / 2
        return self.projection(additive)


class DominanceFusion(nn.Module):
    """
    Dominance genetic effect fusion.

    Computes the absolute difference: |Parent_A - Parent_B|

    This captures deviations from additive inheritance, which can indicate:
    - Complete dominance (one allele masks the other)
    - Partial dominance
    - Over-dominance (heterozygote advantage / heterosis)

    Args:
        input_dim: Dimension of parent representations.
        use_squared: If True, use squared difference instead of absolute.
        project: If True, apply learned projection after fusion.
    """

    def __init__(
            self,
            input_dim: int,
            use_squared: bool = False,
            project: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.use_squared = use_squared
        self.project = project

        if project:
            self.projection = nn.Linear(input_dim, input_dim)
        else:
            self.projection = nn.Identity()

    def forward(
            self,
            parent_a: torch.Tensor,
            parent_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dominance fusion.

        Args:
            parent_a: Parent A representation, shape (..., input_dim).
            parent_b: Parent B representation, shape (..., input_dim).

        Returns:
            Dominance fusion result, shape (..., input_dim).
        """
        diff = parent_a - parent_b

        if self.use_squared:
            dominance = diff ** 2
        else:
            dominance = torch.abs(diff)

        return self.projection(dominance)


class InteractionFusion(nn.Module):
    """
    Gene-gene interaction (epistasis) fusion.

    Computes element-wise product: Parent_A  Parent_B

    This models non-linear interactions between parental alleles,
    capturing epistatic effects where the effect of one gene depends
    on the genotype at another gene.

    Args:
        input_dim: Dimension of parent representations.
        use_bilinear: If True, use learned bilinear interaction.
        project: If True, apply learned projection after fusion.
    """

    def __init__(
            self,
            input_dim: int,
            use_bilinear: bool = False,
            project: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.use_bilinear = use_bilinear

        if use_bilinear:
            # Learned bilinear interaction: x^T W y
            self.bilinear = nn.Bilinear(input_dim, input_dim, input_dim, bias=False)

        if project:
            self.projection = nn.Linear(input_dim, input_dim)
        else:
            self.projection = nn.Identity()

    def forward(
            self,
            parent_a: torch.Tensor,
            parent_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute interaction fusion.

        Args:
            parent_a: Parent A representation, shape (..., input_dim).
            parent_b: Parent B representation, shape (..., input_dim).

        Returns:
            Interaction fusion result, shape (..., input_dim).
        """
        if self.use_bilinear:
            interaction = self.bilinear(parent_a, parent_b)
        else:
            interaction = parent_a * parent_b

        return self.projection(interaction)


class HybridFusion(nn.Module):
    """
    Complete hybrid genetic fusion module.

    Combines additive, dominance, and interaction effects through
    learnable weights to produce a unified hybrid representation.

    The fusion equation is:
        H = W_a * Additive + W_d * Dominance + W_i * Interaction

    where W_a, W_d, W_i are learned fusion weights.

    Args:
        input_dim: Dimension of parent representations.
        hidden_dim: Hidden dimension for fusion MLP.
        output_dim: Output dimension (default: same as input_dim).
        use_additive: Include additive effects.
        use_dominance: Include dominance effects.
        use_interaction: Include interaction effects.
        fusion_method: How to combine effects ('concat_project', 'gated', 'attention').
        dropout: Dropout probability.
    """

    def __init__(
            self,
            input_dim: int = 128,
            hidden_dim: int = 256,
            output_dim: Optional[int] = None,
            use_additive: bool = True,
            use_dominance: bool = True,
            use_interaction: bool = True,
            fusion_method: str = "concat_project",
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.use_additive = use_additive
        self.use_dominance = use_dominance
        self.use_interaction = use_interaction
        self.fusion_method = fusion_method

        # Individual fusion modules
        self.fusions = nn.ModuleDict()
        num_active_fusions = 0

        if use_additive:
            self.fusions["additive"] = AdditiveFusion(input_dim)
            num_active_fusions += 1

        if use_dominance:
            self.fusions["dominance"] = DominanceFusion(input_dim)
            num_active_fusions += 1

        if use_interaction:
            self.fusions["interaction"] = InteractionFusion(input_dim)
            num_active_fusions += 1

        assert num_active_fusions > 0, "At least one fusion type must be enabled"

        # Fusion combination
        if fusion_method == "concat_project":
            self.combiner = nn.Sequential(
                nn.Linear(input_dim * num_active_fusions, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim),
            )
        elif fusion_method == "gated":
            self.combiner = GatedFusionCombiner(
                input_dim=input_dim,
                num_sources=num_active_fusions,
                output_dim=self.output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif fusion_method == "attention":
            self.combiner = AttentionFusionCombiner(
                input_dim=input_dim,
                num_sources=num_active_fusions,
                output_dim=self.output_dim,
                num_heads=4,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(
            self,
            parent_a: torch.Tensor,
            parent_b: torch.Tensor,
            return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hybrid fusion of parent representations.

        Args:
            parent_a: Parent A representation, shape (batch, ..., input_dim).
            parent_b: Parent B representation, shape (batch, ..., input_dim).
            return_components: If True, also return individual fusion components.

        Returns:
            If return_components is False:
                Fused hybrid representation, shape (batch, ..., output_dim).
            If return_components is True:
                Tuple of (fused_repr, dict of component representations).
        """
        # Compute individual fusion components
        components = {}
        component_tensors = []

        for name, fusion_module in self.fusions.items():
            component = fusion_module(parent_a, parent_b)
            components[name] = component
            component_tensors.append(component)

        # Combine components
        if self.fusion_method == "concat_project":
            combined = torch.cat(component_tensors, dim=-1)
            output = self.combiner(combined)
        else:
            # Stack for gated/attention combiner
            stacked = torch.stack(component_tensors, dim=-2)  # (..., num_sources, input_dim)
            output = self.combiner(stacked)

        output = self.layer_norm(output)

        if return_components:
            return output, components
        return output


class GatedFusionCombiner(nn.Module):
    """
    Gated combination of multiple fusion sources.

    Uses learned gates to weight the contribution of each fusion type
    (additive, dominance, interaction) to the final representation.

    Args:
        input_dim: Dimension of each fusion source.
        num_sources: Number of fusion sources to combine.
        output_dim: Output dimension.
        hidden_dim: Hidden dimension for gate computation.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            input_dim: int,
            num_sources: int,
            output_dim: int,
            hidden_dim: int = 128,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_sources = num_sources

        # Gate computation (context-dependent)
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim * num_sources, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sources),
        )

        # Value projection
        self.value_proj = nn.Linear(input_dim, output_dim)

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        """
        Combine sources with learned gates.

        Args:
            sources: Stacked fusion sources, shape (..., num_sources, input_dim).

        Returns:
            Combined representation, shape (..., output_dim).
        """
        # Compute gates based on all sources
        *batch_dims, num_sources, input_dim = sources.shape
        sources_flat = sources.reshape(*batch_dims, num_sources * input_dim)

        gates = self.gate_net(sources_flat)  # (..., num_sources)
        gates = F.softmax(gates, dim=-1)

        # Weighted combination
        gates = gates.unsqueeze(-1)  # (..., num_sources, 1)
        combined = (sources * gates).sum(dim=-2)  # (..., input_dim)

        # Project to output dimension
        output = self.value_proj(combined)

        return output


class AttentionFusionCombiner(nn.Module):
    """
    Attention-based combination of multiple fusion sources.

    Uses self-attention to let fusion sources interact and determine
    their relative importance.

    Args:
        input_dim: Dimension of each fusion source.
        num_sources: Number of fusion sources.
        output_dim: Output dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            input_dim: int,
            num_sources: int,
            output_dim: int,
            num_heads: int = 4,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_sources = num_sources

        # Self-attention over fusion sources
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable query for aggregation
        self.agg_query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        # Output projection
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        """
        Combine sources with attention.

        Args:
            sources: Stacked fusion sources, shape (..., num_sources, input_dim).

        Returns:
            Combined representation, shape (..., output_dim).
        """
        # Handle arbitrary batch dimensions
        original_shape = sources.shape[:-2]
        sources = sources.reshape(-1, self.num_sources, sources.shape[-1])
        batch_size = sources.shape[0]

        # Self-attention among sources
        attended, _ = self.self_attn(sources, sources, sources)

        # Aggregate using learned query
        query = self.agg_query.expand(batch_size, -1, -1)
        aggregated, _ = self.self_attn(query, attended, attended)
        aggregated = aggregated.squeeze(1)  # (batch, input_dim)

        # Project to output
        output = self.output_proj(aggregated)

        # Restore original batch dimensions
        output = output.reshape(*original_shape, -1)

        return output