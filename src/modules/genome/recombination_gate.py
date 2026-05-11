"""
Recombination Gate module for HyGE-Mamba.

This module models genetic recombination (crossover) events that occur
during hybrid formation. In maize and other crops:
- Recombination shuffles genetic material between parental chromosomes
- Crossover points are roughly Poisson-distributed along chromosomes
- The resulting hybrid inherits mosaic haplotype blocks from each parent

The recombination gate learns to model which genomic regions are inherited
from each parent, capturing haplotype block structure and the effects of
genetic linkage on trait inheritance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RecombinationGate(nn.Module):
    """
    Recombination gate for modeling crossover in hybrids.

    This module learns position-dependent gates that model the probability
    of inheriting genetic material from Parent A vs Parent B at each
    genomic position. The gates are designed to produce block-like
    patterns reflecting linkage disequilibrium and haplotype structure.

    Key features:
        - Temperature-controlled smoothness of crossover boundaries
        - Optional position-dependent recombination rates
        - Learnable number of expected crossover points

    Args:
        input_dim: Dimension of parent representations.
        hidden_dim: Hidden dimension for gate computation.
        num_crossover_points: Expected number of crossover events per chromosome.
        temperature: Softmax temperature (lower = sharper boundaries).
        learnable_temperature: If True, make temperature learnable.
        use_position_bias: If True, use position-dependent recombination rates.
    """

    def __init__(
            self,
            input_dim: int = 128,
            hidden_dim: int = 64,
            num_crossover_points: int = 10,
            temperature: float = 1.0,
            learnable_temperature: bool = True,
            use_position_bias: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_crossover_points = num_crossover_points
        self.use_position_bias = use_position_bias

        # Temperature parameter
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temperature", torch.tensor(math.log(temperature)))

        # Gate computation network
        # Takes both parent representations to compute context-aware gates
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Optional position-dependent bias
        if use_position_bias:
            # Learn a baseline recombination pattern
            self.position_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        # Crossover point predictor (for interpretability)
        self.crossover_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        return torch.exp(self.log_temperature)

    def _compute_cumulative_gates(
            self,
            gate_logits: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cumulative gates for block-like inheritance patterns.

        Uses cumulative sum of sigmoid gates to create block patterns
        where consecutive positions tend to come from the same parent.

        Args:
            gate_logits: Raw gate logits, shape (batch, seq_len).
            mask: Optional boolean mask.

        Returns:
            Normalized gates indicating Parent A probability,
            shape (batch, seq_len).
        """
        # Apply temperature
        gate_logits = gate_logits / self.temperature

        # Convert to probabilities of switching parent at each position
        switch_probs = torch.sigmoid(gate_logits)

        # Compute cumulative switches to create block patterns
        # This encourages consecutive positions to have same parent
        cumsum = torch.cumsum(switch_probs, dim=-1)

        # Modular arithmetic to create alternating blocks
        parent_a_prob = 0.5 + 0.5 * torch.cos(math.pi * cumsum)

        if mask is not None:
            parent_a_prob = parent_a_prob * mask.float()

        return parent_a_prob

    def forward(
            self,
            parent_a: torch.Tensor,
            parent_b: torch.Tensor,
            positions: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            return_gates: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply recombination gating to parent representations.

        Args:
            parent_a: Parent A representation, shape (batch, seq_len, input_dim).
            parent_b: Parent B representation, shape (batch, seq_len, input_dim).
            positions: Optional genomic positions, shape (batch, seq_len).
            mask: Optional boolean mask.
            return_gates: If True, also return the computed gates.

        Returns:
            If return_gates is False:
                Recombined representation, shape (batch, seq_len, input_dim).
            If return_gates is True:
                Tuple of (recombined_repr, gates).
        """
        batch_size, seq_len, _ = parent_a.shape

        # Compute context-aware gates
        combined = torch.cat([parent_a, parent_b], dim=-1)
        gate_logits = self.gate_net(combined).squeeze(-1)  # (batch, seq_len)

        # Add position-dependent bias if enabled
        if self.use_position_bias and positions is not None:
            # Normalize positions to [0, 1]
            pos_normalized = positions.float() / (positions.max() + 1)
            pos_bias = self.position_encoder(pos_normalized.unsqueeze(-1)).squeeze(-1)
            gate_logits = gate_logits + pos_bias

        # Compute probability of inheriting from Parent A
        parent_a_gates = torch.sigmoid(gate_logits / self.temperature)
        parent_b_gates = 1 - parent_a_gates

        # Recombine representations
        parent_a_gates = parent_a_gates.unsqueeze(-1)
        parent_b_gates = parent_b_gates.unsqueeze(-1)

        recombined = parent_a_gates * parent_a + parent_b_gates * parent_b

        if return_gates:
            return recombined, parent_a_gates.squeeze(-1)
        return recombined


class BlockRecombinationGate(nn.Module):
    """
    Block-based recombination gate with explicit crossover points.

    Models recombination as discrete crossover events that create
    haplotype blocks. More biologically interpretable than the
    standard gate approach.

    Args:
        input_dim: Dimension of parent representations.
        max_crossovers: Maximum number of crossover points to predict.
        hidden_dim: Hidden dimension for networks.
    """

    def __init__(
            self,
            input_dim: int = 128,
            max_crossovers: int = 10,
            hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.max_crossovers = max_crossovers

        # Network to predict crossover probability at each position
        self.crossover_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initial parent selector (which parent starts first)
        self.initial_parent = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
            self,
            parent_a: torch.Tensor,
            parent_b: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_crossovers: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply block-based recombination.

        Args:
            parent_a: Parent A representation, shape (batch, seq_len, input_dim).
            parent_b: Parent B representation, shape (batch, seq_len, input_dim).
            mask: Optional boolean mask.
            return_crossovers: If True, return crossover probabilities.

        Returns:
            Recombined representation (and optionally crossover probs).
        """
        batch_size, seq_len, _ = parent_a.shape

        # Compute crossover probabilities
        combined = torch.cat([parent_a, parent_b], dim=-1)
        crossover_logits = self.crossover_net(combined).squeeze(-1)
        crossover_probs = torch.sigmoid(crossover_logits)

        # Determine initial parent
        # Use average of first few positions for stability
        initial_combined = combined[:, :min(10, seq_len), :].mean(dim=1)
        initial_logit = self.initial_parent(initial_combined)
        initial_parent_a = torch.sigmoid(initial_logit).squeeze(-1)  # (batch,)

        # Compute cumulative crossovers to determine parent at each position
        # Each crossover switches the parent
        cumulative_crossovers = torch.cumsum(crossover_probs, dim=-1)

        # Odd number of crossovers = different parent, even = same
        # Use continuous approximation for differentiability
        parent_a_prob = initial_parent_a.unsqueeze(-1) * torch.cos(math.pi * cumulative_crossovers) * 0.5 + 0.5

        # Recombine
        parent_a_prob = parent_a_prob.unsqueeze(-1)
        recombined = parent_a_prob * parent_a + (1 - parent_a_prob) * parent_b

        if return_crossovers:
            return recombined, crossover_probs
        return recombined


class DifferentiableRecombination(nn.Module):
    """
    Fully differentiable recombination with Gumbel-Softmax.

    Uses Gumbel-Softmax to enable gradient flow through discrete
    parent selection decisions.

    Args:
        input_dim: Dimension of parent representations.
        hidden_dim: Hidden dimension.
        temperature: Gumbel-Softmax temperature.
        hard: If True, use straight-through estimator for hard selection.
    """

    def __init__(
            self,
            input_dim: int = 128,
            hidden_dim: int = 64,
            temperature: float = 1.0,
            hard: bool = False,
    ) -> None:
        super().__init__()

        self.temperature = temperature
        self.hard = hard

        # Selection network
        self.selector = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # 2 choices: Parent A or B
        )

        # Context aggregator for global recombination pattern
        self.context_agg = nn.GRU(
            input_size=input_dim * 2,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
            self,
            parent_a: torch.Tensor,
            parent_b: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply differentiable recombination.

        Args:
            parent_a: Parent A representation, shape (batch, seq_len, input_dim).
            parent_b: Parent B representation, shape (batch, seq_len, input_dim).
            mask: Optional boolean mask.

        Returns:
            Recombined representation, shape (batch, seq_len, input_dim).
        """
        # Combine parents for context
        combined = torch.cat([parent_a, parent_b], dim=-1)

        # Get global context
        context, _ = self.context_agg(combined)
        context = self.context_proj(context)

        # Compute selection logits
        selection_logits = self.selector(combined)  # (batch, seq_len, 2)

        # Gumbel-Softmax for differentiable selection
        if self.training:
            selection = F.gumbel_softmax(
                selection_logits,
                tau=self.temperature,
                hard=self.hard,
            )
        else:
            # During inference, use argmax
            selection = F.one_hot(selection_logits.argmax(dim=-1), num_classes=2).float()

        # Apply selection
        parent_a_weight = selection[:, :, 0:1]
        parent_b_weight = selection[:, :, 1:2]

        recombined = parent_a_weight * parent_a + parent_b_weight * parent_b

        return recombined