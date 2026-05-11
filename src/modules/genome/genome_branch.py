"""
Genome Branch module for HyGE-Mamba.

This module assembles all genome processing components into a complete
branch that processes dual-parent VCF genotype data:

1. SNP Embedding: Convert genotype values (0-9) to dense vectors
2. Position Encoding: Add chromosomal location information
3. **CNN Compression**: Two Conv1d (stride=8 each)  64 compression
   8571  1072  134, so Mamba only runs on 134 tokens
4. Mamba Encoders: Process each parent's *compressed* genome sequence
5. Segment Router: Aggregate to segment-level representations
6. Hybrid Fusion: Combine parents with additive/dominance/interaction
7. Recombination Gate: Model crossover and haplotype inheritance
8. Epistasis Mixer: Capture gene-gene interactions
9. **Bypass Branch**: Mean-pool raw embeddings (B, L, 64)  (B, 64)
   Provides a gradient highway that bypasses CNN+Mamba entirely
10. **Gated Fusion**: Learned gate balances Mamba path vs bypass

The output is a unified genome representation for GE fusion.

Design Rationale (solving Mamba gradient issues on long SNP sequences):
  - CNN compression shrinks the sequence 64 so Mamba's recurrent
    backprop only chains ~134 steps instead of 8571  no gradient decay.
  - The bypass branch (emb  mean  head) gives loss a direct gradient
    path to the embedding layer from epoch 1.
  - The gated fusion lets the model automatically shift from bypass-
    dominant (early training) to Mamba-dominant (late training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

from .snp_embedding import SNPEmbedding
from .position_encoder import PositionEncoder
from .mamba_encoder import GenomeMambaEncoder, DualParentGenomeEncoder
from .segment_router import SegmentRouter
from .hybrid_fusion import HybridFusion
from .recombination_gate import RecombinationGate
from .epistasis_mixer import EpistasisMixer
from ..common.layer_norm import RMSNorm


@dataclass
class GenomeBranchOutput:
    """Output container for genome branch."""

    genome_repr: torch.Tensor  # Final genome representation
    segment_repr: torch.Tensor  # Segment-level representation
    parent_a_encoded: torch.Tensor  # Encoded parent A
    parent_b_encoded: torch.Tensor  # Encoded parent B
    routing_weights: Optional[torch.Tensor] = None  # Segment routing weights
    recombination_gates: Optional[torch.Tensor] = None  # Recombination gates
    fusion_components: Optional[Dict[str, torch.Tensor]] = None  # Fusion breakdown
    gate_value: Optional[torch.Tensor] = None  # Bypass gate value for monitoring


class GenomeBranch(nn.Module):
    """
    Complete genome processing branch for HyGE-Mamba.

    Processes dual-parent genotype data through the full pipeline:
    embedding  CNN compress  Mamba encode  routing  fusion 
    recombination  epistasis  gated merge with bypass

    Args:
        snp_embedding_dim: Dimension for SNP embeddings.
        num_allele_values: Number of possible genotype values (default 10).
        num_chromosomes: Number of chromosomes (10 for maize).
        position_encoding_dim: Dimension for position encoding.
        mamba_d_model: Model dimension for Mamba encoders.
        mamba_d_state: SSM state dimension.
        mamba_num_layers: Number of Mamba layers.
        num_segments: Number of genomic segments.
        hidden_dim: Hidden dimension for various MLPs.
        output_dim: Final output dimension.
        dropout: Dropout probability.
        use_recombination: If True, include recombination gating.
        use_epistasis: If True, include epistasis mixing.
        share_parent_encoders: If True, share weights between parent encoders.
        cnn_stride: Stride for each CNN compression layer (default 8).
        cnn_num_layers: Number of CNN compression layers (default 2).
    """

    def __init__(
        self,
        snp_embedding_dim: int = 64,
        num_allele_values: int = 10,
        num_chromosomes: int = 10,
        position_encoding_dim: int = 32,
        mamba_d_model: int = 128,
        mamba_d_state: int = 16,
        mamba_num_layers: int = 4,
        num_segments: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        use_recombination: bool = True,
        use_epistasis: bool = True,
        share_parent_encoders: bool = False,
        cnn_stride: int = 8,
        cnn_num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.use_recombination = use_recombination
        self.use_epistasis = use_epistasis
        self.cnn_stride = cnn_stride
        self.cnn_num_layers = cnn_num_layers
        self.snp_embedding_dim = snp_embedding_dim
        self.mamba_d_model = mamba_d_model

        # ================================================================
        # 1. SNP Embedding
        # ================================================================
        self.snp_embedding = SNPEmbedding(
            num_allele_values=num_allele_values,
            embedding_dim=snp_embedding_dim,
            dropout=dropout,
        )

        # ================================================================
        # 2. Position Encoding
        # ================================================================
        self.position_encoder = PositionEncoder(
            num_chromosomes=num_chromosomes,
            chromosome_embedding_dim=position_encoding_dim,
            position_encoding_dim=position_encoding_dim,
            output_dim=snp_embedding_dim,  # Match SNP embedding for addition
            dropout=dropout,
            fusion_method="add",
        )

        # Input projection to Mamba dimension
        self.input_proj = nn.Linear(snp_embedding_dim, mamba_d_model)

        # ================================================================
        # 3. CNN Compression  (NEW  solves long-sequence gradient decay)
        #    Two Conv1d layers, each stride=8  total 64 compression
        #    8571  1072  134  (Mamba only sees ~134 tokens)
        #    BatchNorm stabilises each layer's output distribution.
        # ================================================================
        cnn_layers = []
        for i in range(cnn_num_layers):
            cnn_layers.extend([
                nn.Conv1d(
                    mamba_d_model, mamba_d_model,
                    kernel_size=cnn_stride, stride=cnn_stride, padding=0,
                ),
                nn.BatchNorm1d(mamba_d_model),
                nn.GELU(),
            ])
        self.cnn_compress = nn.Sequential(*cnn_layers)

        # ================================================================
        # 4. Mamba Encoders for each parent (on compressed sequences)
        # ================================================================
        self.parent_encoders = DualParentGenomeEncoder(
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            num_layers=mamba_num_layers,
            dropout=dropout,
            bidirectional=True,
            share_weights=share_parent_encoders,
        )

        # ================================================================
        # 5. Segment Router
        # ================================================================
        self.segment_router_a = SegmentRouter(
            num_segments=num_segments,
            hidden_dim=hidden_dim,
            input_dim=mamba_d_model,
            dropout=dropout,
        )
        self.segment_router_b = SegmentRouter(
            num_segments=num_segments,
            hidden_dim=hidden_dim,
            input_dim=mamba_d_model,
            dropout=dropout,
        )

        # ================================================================
        # 6. Hybrid Fusion
        # ================================================================
        self.hybrid_fusion = HybridFusion(
            input_dim=mamba_d_model,
            hidden_dim=hidden_dim,
            output_dim=mamba_d_model,
            use_additive=True,
            use_dominance=True,
            use_interaction=True,
            fusion_method="concat_project",
            dropout=dropout,
        )

        # 7. Recombination Gate (optional)
        if use_recombination:
            self.recombination_gate = RecombinationGate(
                input_dim=mamba_d_model,
                hidden_dim=hidden_dim // 2,
            )

        # 8. Epistasis Mixer (optional)
        if use_epistasis:
            self.epistasis_mixer = EpistasisMixer(
                d_model=mamba_d_model,
                d_state=mamba_d_state,
                num_layers=2,
                dropout=dropout,
            )

        # Output projection (Mamba path)
        self.output_proj = nn.Sequential(
            RMSNorm(mamba_d_model),
            nn.Linear(mamba_d_model, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        # Global pooling for final representation (Mamba path)
        self.global_pool = nn.Sequential(
            nn.Linear(mamba_d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.final_norm = RMSNorm(output_dim)

        # ================================================================
        # 9. Bypass Branch  (NEW  gradient highway)
        #    Mean-pool raw embeddings *before* input_proj / CNN / Mamba.
        #    Path:  emb_avg  mean(dim=1)  bypass_proj  (B, output_dim)
        #    Loss gradient flows directly to embedding  no Mamba in path.
        # ================================================================
        self.bypass_proj = nn.Sequential(
            nn.Linear(snp_embedding_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.bypass_norm = RMSNorm(output_dim)

        # ================================================================
        # 10. Gated Fusion  (NEW  auto-balance Mamba vs bypass)
        #     gate = ( W  [mamba_out  bypass_out] + b )
        #     output = gate * mamba_out + (1 - gate) * bypass_out
        #
        #     Early training: Mamba untrained  gate  0  rely on bypass
        #     Late  training: Mamba effective  gate  1  leverage sequence
        # ================================================================
        self.gate_net = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid(),
        )
        # Initialise gate bias so early gate  0.5 (balanced start)
        # The Linear in gate_net is at index 0
        nn.init.zeros_(self.gate_net[0].weight)
        nn.init.zeros_(self.gate_net[0].bias)

    def forward(
        self,
        genotypes_a: torch.Tensor,
        genotypes_b: torch.Tensor,
        chromosomes: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | GenomeBranchOutput:
        """
        Process dual-parent genotype data.

        Args:
            genotypes_a: Parent A genotype values, shape (batch, seq_len).
            genotypes_b: Parent B genotype values, shape (batch, seq_len).
            chromosomes: Chromosome indices, shape (batch, seq_len).
            positions: Base pair positions, shape (batch, seq_len).
            mask: Optional boolean mask, shape (batch, seq_len).
            return_intermediates: If True, return detailed outputs.

        Returns:
            If return_intermediates is False:
                Final genome representation, shape (batch, output_dim).
            If return_intermediates is True:
                GenomeBranchOutput with all intermediate representations.
        """
        # ==============================================================
        # 1. Embed SNPs    (B, L, snp_embedding_dim)
        # ==============================================================
        emb_a = self.snp_embedding(genotypes_a, mask)
        emb_b = self.snp_embedding(genotypes_b, mask)

        # ==============================================================
        # 9. BYPASS BRANCH  computed BEFORE any heavy processing
        #    Average both parents' raw embeddings, then mean-pool over
        #    sequence length  (B, snp_embedding_dim)  project
        #    This path: emb  mean  bypass_proj  head  has NO Mamba
        #    in it, so gradients flow unimpeded to the embedding layer.
        # ==============================================================
        bypass_emb = (emb_a + emb_b) * 0.5  # (B, L, snp_embedding_dim)
        if mask is not None:
            # Masked mean: only pool over valid positions
            mask_float = mask.unsqueeze(-1).float()  # (B, L, 1)
            bypass_pooled = (
                (bypass_emb * mask_float).sum(dim=1)
                / mask_float.sum(dim=1).clamp(min=1)
            )
        else:
            bypass_pooled = bypass_emb.mean(dim=1)  # (B, snp_embedding_dim)
        bypass_out = self.bypass_norm(
            self.bypass_proj(bypass_pooled)
        )  # (B, output_dim)

        # ==============================================================
        # 2. Add position encoding
        # ==============================================================
        emb_a = self.position_encoder(chromosomes, positions, emb_a)
        emb_b = self.position_encoder(chromosomes, positions, emb_b)

        # ==============================================================
        # 3. Project to Mamba dimension    (B, L, mamba_d_model)
        # ==============================================================
        emb_a = self.input_proj(emb_a)
        emb_b = self.input_proj(emb_b)

        # ==============================================================
        # 3b. CNN Compression    (B, L', mamba_d_model)
        #     L' = L // (stride ^ cnn_num_layers)
        #     e.g. (B, 8571, 128)  (B, 1072, 128)  (B, 134, 128)
        # ==============================================================
        # Conv1d expects (B, C, L), so transpose, compress, transpose back
        emb_a = self.cnn_compress(emb_a.transpose(1, 2)).transpose(1, 2)
        emb_b = self.cnn_compress(emb_b.transpose(1, 2)).transpose(1, 2)
        # After CNN compression, the original SNP-level mask is no longer
        # spatially aligned. We set mask=None for downstream modules.
        compressed_mask = None

        # ==============================================================
        # 4. Mamba Encode (on *compressed* sequences  ~134 tokens)
        # ==============================================================
        encoded_a, encoded_b = self.parent_encoders(
            emb_a, emb_b, compressed_mask, compressed_mask
        )

        # ==============================================================
        # 5. Route to segments
        # ==============================================================
        # Segment router uses learned queries; works on any seq length.
        # We don't pass chromosomes since the compressed positions no
        # longer correspond 1:1 to original SNP chromosomes.
        segment_a, routing_a = self.segment_router_a(
            encoded_a, compressed_mask, chromosomes=None
        )
        segment_b, routing_b = self.segment_router_b(
            encoded_b, compressed_mask, chromosomes=None
        )

        # ==============================================================
        # 6. Hybrid fusion
        # ==============================================================
        fused, fusion_components = self.hybrid_fusion(
            segment_a, segment_b, return_components=True
        )

        # ==============================================================
        # 7. Recombination gating (optional)
        # ==============================================================
        recomb_gates = None
        if self.use_recombination:
            fused, recomb_gates = self.recombination_gate(
                segment_a, segment_b, return_gates=True
            )
            # Combine with fusion output
            fused = fused + self.hybrid_fusion(segment_a, segment_b)

        # ==============================================================
        # 8. Epistasis mixing (optional)
        # ==============================================================
        if self.use_epistasis:
            batch_size, num_segments, _ = fused.shape
            segment_chr_ids = (
                torch.arange(num_segments, device=fused.device) % 10
            )
            segment_chr_ids = segment_chr_ids.unsqueeze(0).expand(batch_size, -1)
            fused = self.epistasis_mixer(fused, segment_chr_ids)

        # ==============================================================
        # Mamba-path: output projection + global pooling  (B, output_dim)
        # ==============================================================
        segment_repr = self.output_proj(fused)

        pool_weights = F.softmax(
            self.global_pool(fused).squeeze(-1), dim=-1
        )
        mamba_out = torch.einsum("bs,bsd->bd", pool_weights, segment_repr)
        mamba_out = self.final_norm(mamba_out)  # (B, output_dim)

        # ==============================================================
        # 10. GATED FUSION    final genome_repr
        #     gate  (0, 1)^output_dim  (element-wise)
        #     genome_repr = gate  mamba_out + (1  gate)  bypass_out
        # ==============================================================
        gate = self.gate_net(
            torch.cat([mamba_out, bypass_out], dim=-1)
        )  # (B, output_dim)
        genome_repr = gate * mamba_out + (1.0 - gate) * bypass_out

        if return_intermediates:
            return GenomeBranchOutput(
                genome_repr=genome_repr,
                segment_repr=segment_repr,
                parent_a_encoded=encoded_a,
                parent_b_encoded=encoded_b,
                routing_weights=routing_a,
                recombination_gates=recomb_gates,
                fusion_components=fusion_components,
                gate_value=gate.detach().mean(),
            )

        return genome_repr


class GenomeBranchConfig:
    """Configuration helper for GenomeBranch."""

    @staticmethod
    def from_model_config(config: Any) -> Dict[str, Any]:
        """Convert ModelConfig.genome to GenomeBranch kwargs."""
        return {
            "snp_embedding_dim": config.snp_embedding.embedding_dim,
            "num_allele_values": config.snp_embedding.num_allele_values,
            "num_chromosomes": config.position_encoder.num_chromosomes,
            "position_encoding_dim": config.position_encoder.position_encoding_dim,
            "mamba_d_model": config.mamba_encoder.d_model,
            "mamba_d_state": config.mamba_encoder.d_state,
            "mamba_num_layers": config.mamba_encoder.num_layers,
            "num_segments": config.segment_router.num_segments,
            "hidden_dim": config.hybrid_fusion.hidden_dim,
            "output_dim": config.output_dim,
            "dropout": config.mamba_encoder.dropout,
        }
