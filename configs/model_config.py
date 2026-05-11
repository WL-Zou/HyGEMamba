"""
Model configuration dataclass for HyGE-Mamba.

This module defines all architectural hyperparameters for the HyGE-Mamba model,
organized by component (genome branch, weather branch, fusion, MoE, heads).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SNPEmbeddingConfig:
    """Configuration for SNP embedding layer."""
    
    num_allele_values: int = 10  # 0-9 encoding for genotypes
    embedding_dim: int = 64
    dropout: float = 0.1


@dataclass
class PositionEncoderConfig:
    """Configuration for chromosomal position encoding."""
    
    num_chromosomes: int = 10  # Number of chromosomes (maize: 10)
    chromosome_embedding_dim: int = 32
    position_encoding_dim: int = 32
    max_position: int = 300_000_000  # Max bp position (~300Mb for maize chr)
    use_sinusoidal: bool = True
    learnable_chromosomes: bool = True


@dataclass
class MambaEncoderConfig:
    """Configuration for Mamba-style SSM encoder."""
    
    d_model: int = 128
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4  # Local convolution width
    expand_factor: int = 2  # Expansion factor for inner dimension
    num_layers: int = 4
    dropout: float = 0.1
    bidirectional: bool = True
    use_glu: bool = True


@dataclass
class SegmentRouterConfig:
    """Configuration for segment-level gated pooling."""
    
    num_segments: int = 100  # Number of genomic segments
    hidden_dim: int = 128
    num_attention_heads: int = 4
    gate_activation: str = "softmax"  # softmax, sigmoid, or sparsemax
    dropout: float = 0.1


@dataclass 
class HybridFusionConfig:
    """Configuration for hybrid genetic fusion."""
    
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    use_additive: bool = True
    use_dominance: bool = True
    use_interaction: bool = True
    fusion_method: str = "concat_project"  # concat_project, gated, or attention
    dropout: float = 0.1


@dataclass
class RecombinationGateConfig:
    """Configuration for recombination modeling gate."""
    
    input_dim: int = 128
    hidden_dim: int = 64
    num_crossover_points: int = 10  # Approximate number of crossover events
    temperature: float = 1.0
    learnable_temperature: bool = True


@dataclass
class EpistasisMixerConfig:
    """Configuration for epistasis (gene-gene interaction) mixer."""
    
    d_model: int = 128
    d_state: int = 16
    num_layers: int = 2
    dropout: float = 0.1
    use_residual: bool = True


@dataclass
class GenomeBranchConfig:
    """Full configuration for genome processing branch."""
    
    snp_embedding: SNPEmbeddingConfig = field(default_factory=SNPEmbeddingConfig)
    position_encoder: PositionEncoderConfig = field(default_factory=PositionEncoderConfig)
    mamba_encoder: MambaEncoderConfig = field(default_factory=MambaEncoderConfig)
    segment_router: SegmentRouterConfig = field(default_factory=SegmentRouterConfig)
    hybrid_fusion: HybridFusionConfig = field(default_factory=HybridFusionConfig)
    recombination_gate: RecombinationGateConfig = field(default_factory=RecombinationGateConfig)
    epistasis_mixer: EpistasisMixerConfig = field(default_factory=EpistasisMixerConfig)
    
    output_dim: int = 128
    cnn_stride: int = 8       # stride per CNN compression layer
    cnn_num_layers: int = 2   # number of CNN compression layers (total = stride^n)


@dataclass
class IntraDayEncoderConfig:
    """Configuration for intra-day (hourly) weather encoder."""
    
    input_features: int = 8  # temp, humidity, radiation, wind, etc.
    d_model: int = 64
    d_state: int = 8
    num_layers: int = 2
    hours_per_day: int = 24
    dropout: float = 0.1


@dataclass
class InterDayEncoderConfig:
    """Configuration for inter-day (daily/seasonal) weather encoder."""
    
    input_dim: int = 64  # From intra-day encoder
    d_model: int = 128
    d_state: int = 16
    num_layers: int = 3
    max_days: int = 365  # Maximum growing season length
    dropout: float = 0.1
    use_positional_encoding: bool = True


@dataclass
class WeatherBranchConfig:
    """Full configuration for weather processing branch."""
    
    intra_day: IntraDayEncoderConfig = field(default_factory=IntraDayEncoderConfig)
    inter_day: InterDayEncoderConfig = field(default_factory=InterDayEncoderConfig)
    
    output_dim: int = 128


@dataclass
class GxEFusionConfig:
    """Configuration for Genotype  Environment fusion."""
    
    genome_dim: int = 128
    weather_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 256
    num_heads: int = 4
    fusion_type: str = "gated_cross_attention"  # gated_cross_attention, bilinear, or film
    dropout: float = 0.1
    use_residual: bool = True


@dataclass
class RouterConfig:
    """Configuration for MoE router."""
    
    input_dim: int = 256
    num_experts: int = 8
    top_k: int = 2
    noise_std: float = 0.1  # For load balancing during training
    capacity_factor: float = 1.25
    use_aux_loss: bool = True
    aux_loss_weight: float = 0.01


@dataclass
class ExpertConfig:
    """Configuration for individual expert networks."""
    
    input_dim: int = 256
    hidden_dim: int = 512
    output_dim: int = 256
    activation: str = "swiglu"  # swiglu, glu, or gelu
    dropout: float = 0.1


@dataclass
class SparseMoEConfig:
    """Full configuration for Sparse Mixture of Experts."""
    
    router: RouterConfig = field(default_factory=RouterConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    
    num_shared_experts: int = 2
    num_routed_experts: int = 6
    share_expert_weights: bool = False


@dataclass
class APSIMHeadConfig:
    """Configuration for APSIM parameter prediction head."""
    
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    num_parameters: int = 6
    # Parameter ranges for APSIM (typical maize values)
    parameter_names: List[str] = field(default_factory=lambda: [
        "tt_emerg_to_endjuv",  # Thermal time emergence to end juvenile
        "tt_flower_to_maturity",  # Thermal time flowering to maturity
        "tt_flag_to_flower",  # Thermal time flag leaf to flowering
        "head_grain_no_max",  # Maximum grain number per head
        "grain_gth_rate",  # Grain growth rate
        "transp_eff_cf",  # Transpiration efficiency coefficient
    ])
    parameter_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (200.0, 400.0),   # tt_emerg_to_endjuv
        (600.0, 1000.0),  # tt_flower_to_maturity
        (50.0, 150.0),    # tt_flag_to_flower
        (500.0, 1000.0),  # head_grain_no_max
        (5.0, 15.0),      # grain_gth_rate (mg/grain/day)
        (0.006, 0.012),   # transp_eff_cf
    ])
    dropout: float = 0.1


@dataclass
class CurveDecoderConfig:
    """Configuration for time-resolved growth curve decoder."""
    
    input_dim: int = 256
    hidden_dim: int = 128
    num_time_steps: int = 180  # Days in growing season
    num_curves: int = 4  # LAI, biomass, grain_fill, water_stress
    curve_names: List[str] = field(default_factory=lambda: [
        "lai",  # Leaf Area Index
        "biomass",  # Above-ground biomass (kg/ha)
        "grain_fill",  # Grain filling (0-1)
        "water_stress",  # Water stress factor (0-1)
    ])
    use_attention: bool = True
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class ScalarHeadConfig:
    """Configuration for scalar trait prediction heads."""
    
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    num_traits: int = 5
    trait_names: List[str] = field(default_factory=lambda: [
        "yield",  # Final grain yield (kg/ha)
        "flowering_date",  # Days to flowering
        "maturity_date",  # Days to maturity
        "plant_height",  # Final plant height (cm)
        "harvest_index",  # Harvest index (grain/total biomass)
    ])
    dropout: float = 0.1


@dataclass
class ModelConfig:
    """
    Complete model configuration for HyGE-Mamba.
    
    This dataclass aggregates all component configurations and provides
    factory methods for common configurations.
    """
    
    # Component configurations
    genome: GenomeBranchConfig = field(default_factory=GenomeBranchConfig)
    weather: WeatherBranchConfig = field(default_factory=WeatherBranchConfig)
    gxe_fusion: GxEFusionConfig = field(default_factory=GxEFusionConfig)
    sparse_moe: SparseMoEConfig = field(default_factory=SparseMoEConfig)
    apsim_head: APSIMHeadConfig = field(default_factory=APSIMHeadConfig)
    curve_decoder: CurveDecoderConfig = field(default_factory=CurveDecoderConfig)
    scalar_head: ScalarHeadConfig = field(default_factory=ScalarHeadConfig)
    
    # Global settings
    use_fp16: bool = False
    gradient_checkpointing: bool = False
    
    @classmethod
    def small(cls) -> "ModelConfig":
        """Create a small configuration for testing/debugging."""
        config = cls()
        # Reduce dimensions
        config.genome.snp_embedding.embedding_dim = 32
        config.genome.mamba_encoder.d_model = 64
        config.genome.mamba_encoder.num_layers = 2
        config.genome.segment_router.num_segments = 50
        config.genome.hybrid_fusion.hidden_dim = 128
        config.genome.output_dim = 64
        
        config.weather.intra_day.d_model = 32
        config.weather.inter_day.d_model = 64
        config.weather.output_dim = 64
        
        config.gxe_fusion.genome_dim = 64
        config.gxe_fusion.weather_dim = 64
        config.gxe_fusion.hidden_dim = 128
        config.gxe_fusion.output_dim = 128
        
        config.sparse_moe.router.input_dim = 128
        config.sparse_moe.router.num_experts = 4
        config.sparse_moe.num_shared_experts = 1
        config.sparse_moe.num_routed_experts = 3
        config.sparse_moe.expert.input_dim = 128
        config.sparse_moe.expert.hidden_dim = 256
        config.sparse_moe.expert.output_dim = 128
        
        config.apsim_head.input_dim = 128
        config.curve_decoder.input_dim = 128
        config.scalar_head.input_dim = 128
        
        return config
    
    @classmethod
    def large(cls) -> "ModelConfig":
        """Create a large configuration for full-scale training."""
        config = cls()
        # Increase dimensions
        config.genome.snp_embedding.embedding_dim = 128
        config.genome.mamba_encoder.d_model = 256
        config.genome.mamba_encoder.num_layers = 6
        config.genome.segment_router.num_segments = 200
        config.genome.hybrid_fusion.hidden_dim = 512
        config.genome.output_dim = 256
        
        config.weather.intra_day.d_model = 128
        config.weather.inter_day.d_model = 256
        config.weather.inter_day.num_layers = 4
        config.weather.output_dim = 256
        
        config.gxe_fusion.genome_dim = 256
        config.gxe_fusion.weather_dim = 256
        config.gxe_fusion.hidden_dim = 512
        config.gxe_fusion.output_dim = 512
        
        config.sparse_moe.router.input_dim = 512
        config.sparse_moe.router.num_experts = 16
        config.sparse_moe.num_shared_experts = 4
        config.sparse_moe.num_routed_experts = 12
        config.sparse_moe.expert.input_dim = 512
        config.sparse_moe.expert.hidden_dim = 1024
        config.sparse_moe.expert.output_dim = 512
        
        config.apsim_head.input_dim = 512
        config.curve_decoder.input_dim = 512
        config.curve_decoder.hidden_dim = 256
        config.scalar_head.input_dim = 512
        
        return config
