"""
HyGE-Mamba (Hybrid Genotype-Environment Mamba Network) Model.

This is the main model class that assembles all components for
hybrid maize yield prediction:

1. HGIE: Processes dual-parent VCF genotypes into genome_repr
2. PPH:  Predicts crop-model parameters from genome_repr only
3. HEE:  Processes hierarchical weather time series into weather_repr
4. GE Fusion:    Combines genome and weather representations  gxe_repr
5. Sparse MoE:    Mixture of experts for specialized processing
6. Downstream:    Curves & Scalars conditioned on [gxe_repr  apsim_params]

Key design: APSIM parameters are genotype-intrinsic (thermal time, RUE,
grain fill rate, ) and therefore predicted from genome_repr alone. They
are then concatenated with the GE representation to condition the
downstream prediction heads (growth curves and scalar traits), providing
biologically meaningful crop-model priors.

The model supports multi-task learning with biologically-interpretable
intermediate outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any, List
from dataclasses import dataclass, field

from ..modules.genome import GenomeBranch
from ..modules.weather import WeatherBranch, SimpleWeatherBranch
from ..modules.fusion import GxEFusion
from ..modules.moe import SparseMoE, SparseMoELayer
from ..modules.heads import APSIMHead, CurveDecoder, GrowthCurveDecoder, MultiTaskScalarHead
from ..modules.common.layer_norm import RMSNorm


@dataclass
class HyGEMambaConfig:
    """Configuration for HyGE-Mamba model."""

    # HGIE
    snp_embedding_dim: int = 64
    num_allele_values: int = 10
    num_chromosomes: int = 10
    mamba_d_model: int = 128
    mamba_d_state: int = 16
    mamba_num_layers: int = 4
    num_segments: int = 100
    cnn_stride: int = 8          # stride per CNN compression layer
    cnn_num_layers: int = 2      # number of CNN compression layers (total compression = stride^num_layers)

    # HEE
    weather_input_features: int = 8
    intra_day_dim: int = 64
    inter_day_dim: int = 128
    hours_per_day: int = 24
    max_days: int = 180
    use_hourly_weather: bool = True

    # Fusion
    fusion_hidden_dim: int = 256
    fusion_output_dim: int = 256
    fusion_type: str = "gated_cross_attention"

    # MoE
    moe_hidden_dim: int = 512
    num_experts: int = 8
    top_k: int = 2
    num_shared_experts: int = 2
    use_moe: bool = True

    # Output heads
    use_apsim_head: bool = True
    use_curve_decoder: bool = True
    num_timesteps: int = 180
    num_curves: int = 4
    num_apsim_params: int = 6  # APSIM parameter count (conditions downstream heads)

    # Scalar traits
    trait_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "yield": {"range": (0.0, 20000.0), "weight": 2.0},
        "flowering_date": {"range": (50.0, 120.0), "weight": 1.0},
        "maturity_date": {"range": (90.0, 180.0), "weight": 1.0},
        "plant_height": {"range": (100.0, 350.0), "weight": 0.5},
        "harvest_index": {"range": (0.3, 0.6), "weight": 0.5},
    })

    # Modality flags (for M4/M5 ablation experiments)
    use_genotype: bool = True       # False  M4 env-only
    use_environment: bool = True    # False  M5 geno-only

    # General
    dropout: float = 0.1
    predict_uncertainty: bool = False

    @classmethod
    def small(cls) -> "HyGEMambaConfig":
        """Small configuration for testing."""
        return cls(
            mamba_d_model=64,
            mamba_num_layers=2,
            num_segments=50,
            inter_day_dim=64,
            fusion_hidden_dim=128,
            fusion_output_dim=128,
            moe_hidden_dim=256,
            num_experts=4,
        )

    @classmethod
    def large(cls) -> "HyGEMambaConfig":
        """Large configuration for production."""
        return cls(
            mamba_d_model=256,
            mamba_num_layers=6,
            num_segments=200,
            inter_day_dim=256,
            fusion_hidden_dim=512,
            fusion_output_dim=512,
            moe_hidden_dim=1024,
            num_experts=16,
            top_k=4,
            num_shared_experts=4,
        )


@dataclass
class HyGEMambaOutput:
    """Container for HyGE-Mamba model outputs."""

    # Primary outputs
    yield_pred: torch.Tensor
    traits: Dict[str, torch.Tensor]

    # Optional outputs
    apsim_params: Optional[Dict[str, torch.Tensor]] = None
    growth_curves: Optional[torch.Tensor] = None
    curve_params: Optional[Dict[str, torch.Tensor]] = None

    # Intermediate representations
    genome_repr: Optional[torch.Tensor] = None
    weather_repr: Optional[torch.Tensor] = None
    gxe_repr: Optional[torch.Tensor] = None

    # Auxiliary losses
    moe_auxiliary_loss: Optional[torch.Tensor] = None


class HyGEMamba(nn.Module):
    """
    HyGE-Mamba: Hybrid Genotype-Environment Mamba Network.

    End-to-end model for maize hybrid yield prediction using:
    - Dual-parent VCF genotype data
    - Hierarchical weather time series
    - Multi-task learning with interpretable outputs

    Args:
        config: HyGEMambaConfig with all hyperparameters.
    """

    def __init__(self, config: HyGEMambaConfig) -> None:
        super().__init__()

        self.config = config

        # ----------------------------------------------------------------
        # 1. HGIE -> genome_repr (batch, fusion_output_dim)
        # ----------------------------------------------------------------
        self.HGIE = GenomeBranch(
            snp_embedding_dim=config.snp_embedding_dim,
            num_allele_values=config.num_allele_values,
            num_chromosomes=config.num_chromosomes,
            mamba_d_model=config.mamba_d_model,
            mamba_d_state=config.mamba_d_state,
            mamba_num_layers=config.mamba_num_layers,
            num_segments=config.num_segments,
            output_dim=config.fusion_output_dim,
            dropout=config.dropout,
            cnn_stride=config.cnn_stride,
            cnn_num_layers=config.cnn_num_layers,
        )

        # ----------------------------------------------------------------
        # 2. PPH from genome_repr only (genotype-intrinsic params)
        #    Predicted BEFORE GE fusion so they capture pure genetic
        #    crop-model parameters (thermal time, RUE, TE, ).
        # ----------------------------------------------------------------
        if config.use_apsim_head:
            self.PPH = APSIMHead(
                input_dim=config.fusion_output_dim,
                hidden_dim=config.fusion_hidden_dim // 2,
                dropout=config.dropout,
            )
            apsim_cond_dim = config.num_apsim_params
        else:
            self.PPH = None
            apsim_cond_dim = 0

        # ----------------------------------------------------------------
        # 3. HEE -> weather_repr (batch, fusion_output_dim)
        # ----------------------------------------------------------------
        if config.use_hourly_weather:
            self.HEE = WeatherBranch(
                input_features=config.weather_input_features,
                intra_day_dim=config.intra_day_dim,
                inter_day_dim=config.inter_day_dim,
                output_dim=config.fusion_output_dim,
                hours_per_day=config.hours_per_day,
                max_days=config.max_days,
                dropout=config.dropout,
            )
        else:
            self.HEE = SimpleWeatherBranch(
                input_features=config.weather_input_features,
                d_model=config.inter_day_dim,
                output_dim=config.fusion_output_dim,
                max_days=config.max_days,
                dropout=config.dropout,
            )

        # ----------------------------------------------------------------
        # 4. GE Fusion  (genome_repr  weather_repr)
        # ----------------------------------------------------------------
        self.gxe_fusion = GxEFusion(
            genome_dim=config.fusion_output_dim,
            weather_dim=config.fusion_output_dim,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.fusion_output_dim,
            fusion_type=config.fusion_type,
            dropout=config.dropout,
        )

        # ----------------------------------------------------------------
        # 4b. Single-modality bypass projections (M4 / M5)
        #     When one modality is absent, skip GE fusion entirely and
        #     project the available representation through a dedicated MLP
        #     to match the fusion output dimension.
        # ----------------------------------------------------------------
        self.env_only_proj = nn.Sequential(
            nn.Linear(config.fusion_output_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_output_dim),
        )
        self.geno_only_proj = nn.Sequential(
            nn.Linear(config.fusion_output_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_output_dim),
        )

        # ----------------------------------------------------------------
        # M4 override: disable APSIM head when genotype is unavailable
        # (APSIM params are genotype-intrinsic  meaningless without genome)
        # ----------------------------------------------------------------
        if not config.use_genotype and self.PPH is not None:
            self.PPH = None
            apsim_cond_dim = 0

        # ----------------------------------------------------------------
        # 5. Sparse MoE  (optional, refines gxe_repr)
        # ----------------------------------------------------------------
        if config.use_moe:
            self.moe = SparseMoELayer(
                input_dim=config.fusion_output_dim,
                hidden_dim=config.moe_hidden_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                num_shared_experts=config.num_shared_experts,
                dropout=config.dropout,
            )
        else:
            self.moe = None

        self.final_norm = RMSNorm(config.fusion_output_dim)

        # ----------------------------------------------------------------
        # Downstream input dim = gxe_repr + APSIM parameter vector
        # The APSIM params act as genotype-intrinsic conditioning that
        # informs Curves and Scalars about crop-model priors.
        # ----------------------------------------------------------------
        downstream_dim = config.fusion_output_dim + apsim_cond_dim

        # ----------------------------------------------------------------
        # 6. Growth Curve Decoder   (gxe_repr  apsim_params)
        # ----------------------------------------------------------------
        if config.use_curve_decoder:
            self.curve_decoder = GrowthCurveDecoder(
                input_dim=downstream_dim,
                hidden_dim=config.fusion_hidden_dim // 2,
                num_timesteps=config.num_timesteps,
                dropout=config.dropout,
            )
        else:
            self.curve_decoder = None

        # ----------------------------------------------------------------
        # 7. Scalar Trait Heads   (gxe_repr  apsim_params)
        # ----------------------------------------------------------------
        self.scalar_head = MultiTaskScalarHead(
            input_dim=downstream_dim,
            hidden_dim=config.fusion_hidden_dim // 2,
            trait_configs=config.trait_configs,
            predict_uncertainty=config.predict_uncertainty,
            dropout=config.dropout,
        )

    def forward(
            self,
            genotypes_a: torch.Tensor,
            genotypes_b: torch.Tensor,
            chromosomes: torch.Tensor,
            positions: torch.Tensor,
            weather: torch.Tensor,
            snp_mask: Optional[torch.Tensor] = None,
            day_mask: Optional[torch.Tensor] = None,
            hour_mask: Optional[torch.Tensor] = None,
            return_intermediates: bool = False,
    ) -> HyGEMambaOutput:
        """
        Forward pass through HyGE-Mamba.

        Data flow (v2  APSIM-conditioned downstream heads):
            1. HGIE            genome_repr
            2. PPH             apsim_params   (genome_repr ONLY, genotype-intrinsic)
            3. HEE             weather_repr
            4. GE Fusion      gxe_repr       (genome_repr  weather_repr)
            5. Sparse MoE      gxe_repr       (refined)
            6. Concat          downstream = [gxe_repr ; apsim_params]
            7. Curve Decoder   growth_curves  (conditioned on GE + APSIM)
            8. Scalar Heads    traits         (conditioned on GE + APSIM)

        The key insight: APSIM parameters (thermal time, RUE, grain fill
        rate, ) are genotype-intrinsic crop-model priors. By predicting
        them from genome_repr alone and then conditioning the downstream
        heads, we:
          - enforce biological interpretability (APSIM params don't depend
            on weather, only on genotype);
          - provide crop-model priors that help Curves and Scalars.

        Args:
            genotypes_a: Parent A genotypes, shape (batch, num_snps).
            genotypes_b: Parent B genotypes, shape (batch, num_snps).
            chromosomes: Chromosome indices, shape (batch, num_snps).
            positions: Base pair positions, shape (batch, num_snps).
            weather: Weather data, shape (batch, days, hours, features) or
                    (batch, days, features) for daily data.
            snp_mask: Optional mask for valid SNPs.
            day_mask: Optional mask for valid days.
            hour_mask: Optional mask for valid hours.
            return_intermediates: If True, include intermediate representations.

        Returns:
            HyGEMambaOutput with predictions and optional intermediates.
        """
        # ==============================================================
        # ROUTING: single-modality bypass or full GE fusion
        # ==============================================================
        genome_repr = None
        weather_repr = None
        apsim_params = None
        apsim_params_vec = None

        if not self.config.use_genotype:
            #  M4: Environment-only 
            # Skip genome branch entirely; project weather_repr through
            # a dedicated MLP instead of the GE fusion module.
            if self.config.use_hourly_weather:
                weather_repr = self.HEE(
                    weather, day_mask=day_mask, hour_mask=hour_mask,
                )
            else:
                weather_repr = self.HEE(weather, mask=day_mask)

            gxe_repr = self.env_only_proj(weather_repr)

        elif not self.config.use_environment:
            #  M5: Genotype-only 
            # Skip weather branch entirely; project genome_repr through
            # a dedicated MLP instead of the GE fusion module.
            genome_repr = self.HGIE(
                genotypes_a, genotypes_b,
                chromosomes, positions, mask=snp_mask,
            )

            # APSIM is genome-intrinsic  still meaningful in M5
            if self.PPH is not None:
                apsim_params = self.PPH(genome_repr)
                apsim_params_vec = torch.stack(
                    list(apsim_params.values()), dim=-1
                )

            gxe_repr = self.geno_only_proj(genome_repr)

        else:
            #  Full GE (M1 / M2 / M3) 
            # ==== 1. HGIE ====
            genome_repr = self.HGIE(
                genotypes_a, genotypes_b,
                chromosomes, positions, mask=snp_mask,
            )

            # ==== 2. PPH from genome_repr only ====
            if self.PPH is not None:
                apsim_params = self.PPH(genome_repr)
                apsim_params_vec = torch.stack(
                    list(apsim_params.values()), dim=-1
                )

            # ==== 3. HEE ====
            if self.config.use_hourly_weather:
                weather_repr = self.HEE(
                    weather, day_mask=day_mask, hour_mask=hour_mask,
                )
            else:
                weather_repr = self.HEE(weather, mask=day_mask)

            # ==== 4. GE Fusion ====
            gxe_repr = self.gxe_fusion(genome_repr, weather_repr)

        # ==============================================================
        # Shared downstream pipeline (all modalities rejoin here)
        # ==============================================================

        # ==== 5. Sparse MoE (optional) ====
        moe_aux_loss = None
        if self.moe is not None:
            gxe_repr, moe_aux_loss = self.moe(gxe_repr)

        gxe_repr = self.final_norm(gxe_repr)

        # ==== 6. Build downstream input: gxe_repr  apsim_params ====
        if apsim_params_vec is not None:
            downstream_repr = torch.cat(
                [gxe_repr, apsim_params_vec], dim=-1
            )  # (batch, fusion_output_dim + num_apsim_params)
        else:
            downstream_repr = gxe_repr

        # ==== 7. Growth Curves (conditioned on GE + APSIM) ====
        growth_curves = None
        curve_params = None
        if self.curve_decoder is not None:
            growth_curves, curve_params = self.curve_decoder(downstream_repr)

        # ==== 8. Scalar Traits (conditioned on GE + APSIM) ====
        traits = self.scalar_head(downstream_repr)

        # Extract yield for convenience
        if isinstance(traits.get("yield"), tuple):
            yield_pred = traits["yield"][0]  # Mean if uncertainty
        else:
            yield_pred = traits.get("yield", torch.zeros(genotypes_a.shape[0], device=genotypes_a.device))

        output = HyGEMambaOutput(
            yield_pred=yield_pred,
            traits=traits,
            apsim_params=apsim_params,
            growth_curves=growth_curves,
            curve_params=curve_params,
            moe_auxiliary_loss=moe_aux_loss,
        )

        if return_intermediates:
            output.genome_repr = genome_repr
            output.weather_repr = weather_repr
            output.gxe_repr = gxe_repr

        return output

    def predict_yield(
            self,
            genotypes_a: torch.Tensor,
            genotypes_b: torch.Tensor,
            chromosomes: torch.Tensor,
            positions: torch.Tensor,
            weather: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        """
        Convenience method for yield prediction only.

        Returns:
            Predicted yield, shape (batch,).
        """
        output = self.forward(
            genotypes_a, genotypes_b,
            chromosomes, positions, weather,
            return_intermediates=False,
            **kwargs,
        )
        return output.yield_pred

    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts by module."""
        counts = {}

        counts["HGIE"] = sum(p.numel() for p in self.HGIE.parameters())
        counts["HEE"] = sum(p.numel() for p in self.HEE.parameters())
        counts["gxe_fusion"] = sum(p.numel() for p in self.gxe_fusion.parameters())

        if self.moe is not None:
            counts["moe"] = sum(p.numel() for p in self.moe.parameters())

        if self.PPH is not None:
            counts["PPH"] = sum(p.numel() for p in self.PPH.parameters())

        if self.curve_decoder is not None:
            counts["curve_decoder"] = sum(p.numel() for p in self.curve_decoder.parameters())

        counts["scalar_head"] = sum(p.numel() for p in self.scalar_head.parameters())

        counts["total"] = sum(p.numel() for p in self.parameters())

        return counts


def build_hyge_mamba(
        config: Optional[HyGEMambaConfig] = None,
        size: str = "base",
) -> HyGEMamba:
    """
    Factory function to build HyGE-Mamba model.

    Args:
        config: Optional configuration. If None, use default based on size.
        size: Model size ('small', 'base', 'large').

    Returns:
        Initialized HyGEMamba model.
    """
    if config is None:
        if size == "small":
            config = HyGEMambaConfig.small()
        elif size == "large":
            config = HyGEMambaConfig.large()
        else:
            config = HyGEMambaConfig()

    model = HyGEMamba(config)

    return model
