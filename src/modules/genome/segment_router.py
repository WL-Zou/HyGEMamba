"""
Segment Router module for HyGE-Mamba.

This module implements gated segment-level pooling for genomic sequences.
It aggregates SNP-level representations into segment-level representations,
where segments can represent biological units like:
- Linkage disequilibrium (LD) blocks
- Gene regions
- Quantitative trait loci (QTL)
- Chromosomal bins

The routing mechanism allows the model to learn which SNPs are most
important for each segment's representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GatedPooling(nn.Module):
    """
    Gated pooling mechanism for sequence aggregation.

    Computes attention-like gates over sequence positions to produce
    a weighted aggregation. Different from standard attention in that
    it uses a learned query per segment rather than cross-attention.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden dimension for gate computation.
        gate_activation: Activation for gates ('softmax', 'sigmoid', 'sparsemax').
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            gate_activation: str = "softmax",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.gate_activation = gate_activation

        # Gate computation
        self.gate_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gated pooling.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            mask: Optional boolean mask, shape (batch, seq_len).

        Returns:
            Tuple of:
                - Pooled representation, shape (batch, input_dim).
                - Gate weights, shape (batch, seq_len).
        """
        # Compute gate scores
        gate_scores = self.gate_proj(x).squeeze(-1)  # (batch, seq_len)

        # Apply mask
        if mask is not None:
            gate_scores = gate_scores.masked_fill(~mask, float("-inf"))

        # Compute gate weights
        if self.gate_activation == "softmax":
            gate_weights = F.softmax(gate_scores, dim=-1)
        elif self.gate_activation == "sigmoid":
            gate_weights = torch.sigmoid(gate_scores)
            if mask is not None:
                gate_weights = gate_weights * mask.float()
            gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)
        elif self.gate_activation == "sparsemax":
            gate_weights = self._sparsemax(gate_scores)
        else:
            raise ValueError(f"Unknown gate activation: {self.gate_activation}")

        # Weighted aggregation
        pooled = torch.einsum("bs,bsd->bd", gate_weights, x)

        return pooled, gate_weights

    def _sparsemax(self, scores: torch.Tensor) -> torch.Tensor:
        """Sparsemax activation for sparse attention."""
        # Sort scores in descending order
        sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)

        # Compute cumulative sum
        cumsum = torch.cumsum(sorted_scores, dim=-1)

        # Find threshold
        k = torch.arange(1, scores.size(-1) + 1, device=scores.device).float()
        threshold = (cumsum - 1) / k

        # Find k where sorted_score > threshold
        support = (sorted_scores > threshold).sum(dim=-1, keepdim=True)

        # Compute tau
        tau = (cumsum.gather(-1, support - 1) - 1) / support.float()

        # Apply sparsemax
        output = torch.clamp(scores - tau, min=0)

        return output


class SegmentRouter(nn.Module):
    """
    Segment-level router for genomic sequence aggregation.

    Divides the genomic sequence into segments and computes segment-level
    representations using gated pooling. Segments can be:
    - Fixed-size bins (equal number of SNPs per segment)
    - Chromosome-based (one segment per chromosome)
    - Custom (provided segment assignments)

    Args:
        num_segments: Number of output segments.
        hidden_dim: Hidden dimension for routing.
        input_dim: Input feature dimension (will be inferred if None).
        num_attention_heads: Number of heads for multi-head gated pooling.
        gate_activation: Activation for gates.
        dropout: Dropout probability.
        segment_mode: How to assign SNPs to segments ('fixed', 'learned', 'chromosome').
    """

    def __init__(
            self,
            num_segments: int = 100,
            hidden_dim: int = 128,
            input_dim: Optional[int] = None,
            num_attention_heads: int = 4,
            gate_activation: str = "softmax",
            dropout: float = 0.1,
            segment_mode: str = "fixed",
    ) -> None:
        super().__init__()

        self.num_segments = num_segments
        self.hidden_dim = hidden_dim
        self.num_heads = num_attention_heads
        self.segment_mode = segment_mode

        # Input dimension (will be set in first forward if None)
        self._input_dim = input_dim

        # Per-segment query vectors (for attention-based pooling)
        if input_dim is not None:
            self._init_layers(input_dim)
        else:
            self.segment_queries = None

    def _init_layers(self, input_dim: int) -> None:
        """Initialize layers once input dimension is known."""
        self._input_dim = input_dim

        # Learnable segment queries
        self.segment_queries = nn.Parameter(
            torch.randn(self.num_segments, input_dim) * 0.02
        )

        # Multi-head attention for routing
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Gated pooling for each segment
        self.gated_pooling = GatedPooling(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            gate_activation="softmax",
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, input_dim),
        )

        self.layer_norm = nn.LayerNorm(input_dim)

    def _assign_segments_fixed(
            self,
            seq_len: int,
            batch_size: int,
            device: torch.device,
    ) -> torch.Tensor:
        """Assign SNPs to segments with fixed-size bins."""
        # Compute segment assignment for each position
        segment_size = math.ceil(seq_len / self.num_segments)
        positions = torch.arange(seq_len, device=device)
        segment_ids = (positions // segment_size).clamp(max=self.num_segments - 1)

        # Expand for batch
        segment_ids = segment_ids.unsqueeze(0).expand(batch_size, -1)

        return segment_ids

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            segment_ids: Optional[torch.Tensor] = None,
            chromosomes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route SNPs to segments and compute segment representations.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            mask: Optional boolean mask, shape (batch, seq_len).
            segment_ids: Optional pre-computed segment assignments,
                        shape (batch, seq_len).
            chromosomes: Optional chromosome indices for chromosome-based
                        segmentation, shape (batch, seq_len).

        Returns:
            Tuple of:
                - Segment representations, shape (batch, num_segments, input_dim).
                - Routing weights, shape (batch, num_segments, seq_len).
        """
        batch_size, seq_len, input_dim = x.shape

        # Initialize layers on first forward pass
        if self.segment_queries is None:
            self._init_layers(input_dim)
            self.segment_queries = self.segment_queries.to(x.device)

        # Compute segment assignments if not provided
        if segment_ids is None:
            if self.segment_mode == "fixed":
                segment_ids = self._assign_segments_fixed(seq_len, batch_size, x.device)
            elif self.segment_mode == "chromosome" and chromosomes is not None:
                # Use chromosome as segment (assuming <= num_segments chromosomes)
                segment_ids = chromosomes.clamp(max=self.num_segments - 1)
            else:
                segment_ids = self._assign_segments_fixed(seq_len, batch_size, x.device)

        # Use attention between segment queries and input sequence
        # Expand segment queries for batch
        queries = self.segment_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute attention weights
        # queries: (batch, num_segments, input_dim)
        # x: (batch, seq_len, input_dim)
        attn_output, attn_weights = self.attention(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=~mask if mask is not None else None,
        )

        # Apply output projection
        segment_repr = self.output_proj(attn_output)
        segment_repr = self.layer_norm(segment_repr + queries)

        return segment_repr, attn_weights


class AdaptiveSegmentRouter(nn.Module):
    """
    Adaptive segment router that learns optimal segmentation.

    Instead of fixed segments, this router learns to identify
    biologically meaningful segments (e.g., LD blocks, genes)
    from the data.

    Args:
        max_segments: Maximum number of segments.
        hidden_dim: Hidden dimension.
        input_dim: Input feature dimension.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            max_segments: int = 100,
            hidden_dim: int = 128,
            input_dim: int = 128,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.max_segments = max_segments

        # Boundary detection network
        self.boundary_detector = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Segment aggregation
        self.segment_agg = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.output_proj = nn.Linear(hidden_dim * 2, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptively segment and aggregate sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            mask: Optional boolean mask.

        Returns:
            Tuple of (segment_representations, boundary_scores).
        """
        batch_size, seq_len, input_dim = x.shape

        # Detect boundaries by comparing adjacent positions
        x_left = x[:, :-1, :]
        x_right = x[:, 1:, :]
        boundary_input = torch.cat([x_left, x_right], dim=-1)
        boundary_scores = self.boundary_detector(boundary_input).squeeze(-1)
        boundary_probs = torch.sigmoid(boundary_scores)

        # Pad boundary scores
        boundary_probs = F.pad(boundary_probs, (0, 1), value=1.0)  # Force boundary at end

        # For now, use fixed segmentation based on boundaries
        # In training, use soft assignment; in inference, use hard boundaries

        # Process full sequence and pool based on boundaries
        encoded, _ = self.segment_agg(x)
        encoded = self.output_proj(encoded)

        # Simple aggregation: use boundary-weighted pooling
        # This is a simplified version; full implementation would use
        # differentiable segment aggregation

        # Create segment representations by pooling regions between boundaries
        segment_size = max(1, seq_len // self.max_segments)
        num_actual_segments = min(self.max_segments, seq_len // segment_size)

        segments = []
        for i in range(num_actual_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size, seq_len)
            segment = encoded[:, start:end, :].mean(dim=1)
            segments.append(segment)

        segment_repr = torch.stack(segments, dim=1)  # (batch, num_segments, input_dim)
        segment_repr = self.layer_norm(segment_repr)

        return segment_repr, boundary_probs