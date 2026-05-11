"""
Tests for HyGE-Mamba model components.

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hyge_mamba import HyGEMamba, HyGEMambaConfig


class TestHyGEMambaConfig:
    """Test configuration classes."""

    def test_default_config(self):
        """Test default configuration."""
        config = HyGEMambaConfig()
        assert config.mamba_d_model == 128
        assert config.num_experts == 8
        assert config.top_k == 2

    def test_small_config(self):
        """Test small configuration."""
        config = HyGEMambaConfig.small()
        assert config.mamba_d_model == 64
        assert config.mamba_num_layers == 2

    def test_large_config(self):
        """Test large configuration."""
        config = HyGEMambaConfig.large()
        assert config.mamba_d_model == 256
        assert config.num_experts == 16


class TestHyGEMamba:
    """Test HyGE-Mamba model."""

    @pytest.fixture
    def small_config(self):
        """Small config for fast testing."""
        return HyGEMambaConfig(
            snp_embedding_dim=32,
            mamba_d_model=32,
            mamba_d_state=8,
            mamba_num_layers=1,
            num_segments=10,
            intra_day_dim=32,
            inter_day_dim=32,
            fusion_hidden_dim=64,
            fusion_output_dim=64,
            moe_hidden_dim=128,
            num_experts=4,
            top_k=2,
            num_shared_experts=1,
            use_moe=True,
            use_apsim_head=True,
            use_curve_decoder=True,
            num_timesteps=90,
            num_curves=4,
            dropout=0.0,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        batch_size = 4
        num_snps = 1024
        num_days = 30
        hours_per_day = 24
        num_features = 8

        return {
            "genotypes_a": torch.randint(0, 3, (batch_size, num_snps)),
            "genotypes_b": torch.randint(0, 3, (batch_size, num_snps)),
            "chromosomes": torch.randint(0, 10, (batch_size, num_snps)),
            "positions": torch.arange(num_snps).unsqueeze(0).expand(batch_size, -1),
            "hourly_weather": torch.randn(batch_size, num_days, hours_per_day, num_features),
            "snp_mask": torch.ones(batch_size, num_snps, dtype=torch.bool),
            "day_mask": torch.ones(batch_size, num_days, dtype=torch.bool),
            "hour_mask": torch.ones(batch_size, num_days, hours_per_day, dtype=torch.bool),
        }

    def test_model_creation(self, small_config):
        """Test model instantiation."""
        model = HyGEMamba(small_config)
        assert model is not None

        # Check components exist
        assert hasattr(model, "HGIE")
        assert hasattr(model, "HEE")
        assert hasattr(model, "PPH")
        assert hasattr(model, "gxe_fusion")
        assert hasattr(model, "scalar_head")

    def test_forward_pass(self, small_config, sample_batch):
        """Test forward pass."""
        model = HyGEMamba(small_config)
        model.eval()

        with torch.no_grad():
            output = model(
                genotypes_a=sample_batch["genotypes_a"],
                genotypes_b=sample_batch["genotypes_b"],
                chromosomes=sample_batch["chromosomes"],
                positions=sample_batch["positions"],
                weather=sample_batch["hourly_weather"],
                snp_mask=sample_batch["snp_mask"],
                day_mask=sample_batch["day_mask"],
                hour_mask=sample_batch["hour_mask"],
            )

        # Check output structure
        assert output.yield_pred is not None
        assert output.yield_pred.shape == (4,)
        assert output.traits is not None

    def test_output_shapes(self, small_config, sample_batch):
        """Test output shapes."""
        model = HyGEMamba(small_config)
        model.eval()

        with torch.no_grad():
            output = model(
                genotypes_a=sample_batch["genotypes_a"],
                genotypes_b=sample_batch["genotypes_b"],
                chromosomes=sample_batch["chromosomes"],
                positions=sample_batch["positions"],
                weather=sample_batch["hourly_weather"],
            )

        batch_size = sample_batch["genotypes_a"].shape[0]

        # Yield
        assert output.yield_pred.shape == (batch_size,)

        # APSIM parameters
        if output.apsim_params is not None:
            for param_value in output.apsim_params.values():
                assert param_value.shape == (batch_size,)

        # Growth curves - shape is (batch, timesteps, num_curves)
        if output.growth_curves is not None:
            assert output.growth_curves.shape[0] == batch_size
            assert output.growth_curves.shape[1] == small_config.num_timesteps
            assert output.growth_curves.shape[2] == small_config.num_curves

    def test_gradient_flow(self, small_config, sample_batch):
        """Test that gradients flow through the model."""
        model = HyGEMamba(small_config)
        model.train()

        output = model(
            genotypes_a=sample_batch["genotypes_a"],
            genotypes_b=sample_batch["genotypes_b"],
            chromosomes=sample_batch["chromosomes"],
            positions=sample_batch["positions"],
            weather=sample_batch["hourly_weather"],
        )

        # Compute dummy loss
        loss = output.yield_pred.mean()
        loss.backward()

        # Check gradients exist
        has_gradient = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradient = True
                break

        assert has_gradient, "No gradients found in model parameters"

    def test_deterministic_inference(self, small_config, sample_batch):
        """Test that inference is deterministic in eval mode."""
        model = HyGEMamba(small_config)
        model.eval()

        with torch.no_grad():
            output1 = model(
                genotypes_a=sample_batch["genotypes_a"],
                genotypes_b=sample_batch["genotypes_b"],
                chromosomes=sample_batch["chromosomes"],
                positions=sample_batch["positions"],
                weather=sample_batch["hourly_weather"],
            )

            output2 = model(
                genotypes_a=sample_batch["genotypes_a"],
                genotypes_b=sample_batch["genotypes_b"],
                chromosomes=sample_batch["chromosomes"],
                positions=sample_batch["positions"],
                weather=sample_batch["hourly_weather"],
            )

        torch.testing.assert_close(output1.yield_pred, output2.yield_pred)

    def test_parameter_count(self, small_config):
        """Test parameter counting."""
        model = HyGEMamba(small_config)

        counts = model.get_num_parameters()

        assert "total" in counts
        assert counts["total"] > 0
        assert counts["HGIE"] > 0
        assert counts["HEE"] > 0
        assert counts["PPH"] > 0

    def test_predict_yield_method(self, small_config, sample_batch):
        """Test convenience yield prediction method."""
        model = HyGEMamba(small_config)
        model.eval()

        with torch.no_grad():
            yield_pred = model.predict_yield(
                genotypes_a=sample_batch["genotypes_a"],
                genotypes_b=sample_batch["genotypes_b"],
                chromosomes=sample_batch["chromosomes"],
                positions=sample_batch["positions"],
                weather=sample_batch["hourly_weather"],
            )

        assert yield_pred.shape == (sample_batch["genotypes_a"].shape[0],)

    def test_intermediates(self, small_config, sample_batch):
        """Test intermediate representation output."""
        model = HyGEMamba(small_config)
        model.eval()

        with torch.no_grad():
            output = model(
                genotypes_a=sample_batch["genotypes_a"],
                genotypes_b=sample_batch["genotypes_b"],
                chromosomes=sample_batch["chromosomes"],
                positions=sample_batch["positions"],
                weather=sample_batch["hourly_weather"],
                return_intermediates=True,
            )

        assert output.genome_repr is not None
        assert output.weather_repr is not None
        assert output.gxe_repr is not None

    def test_no_moe(self, sample_batch):
        """Test model without MoE."""
        config = HyGEMambaConfig(
            mamba_d_model=32,
            mamba_num_layers=1,
            num_segments=10,
            fusion_hidden_dim=64,
            fusion_output_dim=64,
            use_moe=False,
        )

        model = HyGEMamba(config)
        model.eval()

        with torch.no_grad():
            output = model(
                genotypes_a=sample_batch["genotypes_a"],
                genotypes_b=sample_batch["genotypes_b"],
                chromosomes=sample_batch["chromosomes"],
                positions=sample_batch["positions"],
                weather=sample_batch["hourly_weather"],
            )

        assert output.moe_auxiliary_loss is None


class TestGenomeBranch:
    """Test genome branch components."""

    def test_snp_embedding(self):
        """Test SNP embedding."""
        from src.modules.genome import SNPEmbedding

        embed = SNPEmbedding(
            num_allele_values=10,
            embedding_dim=32,
        )

        genotypes = torch.randint(0, 3, (4, 100))
        output = embed(genotypes)

        assert output.shape == (4, 100, 32)

    def test_position_encoder(self):
        """Test position encoding."""
        from src.modules.genome import ChromosomePositionEncoder

        encoder = ChromosomePositionEncoder(
            num_chromosomes=10,
            chromosome_embedding_dim=16,
            position_encoding_dim=16,
        )

        chromosomes = torch.randint(0, 10, (4, 100))
        positions = torch.arange(100).unsqueeze(0).expand(4, -1)

        output = encoder(chromosomes, positions)

        assert output.shape == (4, 100, 32)  # 16 + 16 = 32


class TestWeatherBranch:
    """Test weather branch components."""

    def test_intra_day_encoder(self):
        """Test intra-day encoding."""
        from src.modules.weather import IntraDayEncoder

        encoder = IntraDayEncoder(
            input_features=8,
            d_model=32,
        )

        hourly = torch.randn(4, 30, 24, 8)
        output = encoder(hourly)

        assert output.shape == (4, 30, 32)

    def test_inter_day_encoder(self):
        """Test inter-day encoding."""
        from src.modules.weather import InterDayEncoder

        encoder = InterDayEncoder(
            input_dim=32,
            d_model=64,
        )

        daily = torch.randn(4, 30, 32)
        output = encoder(daily)

        # InterDayEncoder preserves sequence dimension: (batch, days, d_model)
        assert output.shape == (4, 30, 64)


class TestFusion:
    """Test fusion module."""

    def test_gxe_fusion(self):
        """Test GE fusion."""
        from src.modules.fusion import GxEFusion

        fusion = GxEFusion(
            genome_dim=64,
            weather_dim=64,
            hidden_dim=128,
            output_dim=64,
        )

        genome = torch.randn(4, 64)
        weather = torch.randn(4, 64)

        output = fusion(genome, weather)

        assert output.shape == (4, 64)


class TestMoE:
    """Test MoE components."""

    def test_sparse_moe(self):
        """Test sparse MoE layer."""
        from src.modules.moe import SparseMoE

        moe = SparseMoE(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_experts=4,
            top_k=2,
        )

        x = torch.randn(4, 64)
        output = moe(x)

        assert output.output.shape == (4, 64)

    def test_router(self):
        """Test top-k router."""
        from src.modules.moe import TopKRouter

        router = TopKRouter(
            input_dim=64,
            num_experts=8,
            top_k=2,
        )

        x = torch.randn(4, 64)
        weights, indices, logits = router(x)

        assert weights.shape == (4, 2)
        assert indices.shape == (4, 2)
        assert logits.shape == (4, 8)


class TestHeads:
    """Test output heads."""

    def test_PPH(self):
        """Test APSIM parameter head."""
        from src.modules.heads import APSIMHead

        head = APSIMHead(input_dim=64, hidden_dim=32)

        x = torch.randn(4, 64)
        output = head(x)

        assert isinstance(output, dict)
        assert len(output) == 6  # 6 APSIM parameters

    def test_curve_decoder(self):
        """Test growth curve decoder."""
        from src.modules.heads import GrowthCurveDecoder

        decoder = GrowthCurveDecoder(
            input_dim=64,
            hidden_dim=32,
            num_timesteps=90,
        )

        x = torch.randn(4, 64)
        curves, params = decoder(x)

        assert curves.shape[0] == 4
        assert curves.shape[1] == 90  # (batch, timesteps, num_curves)

    def test_scalar_head(self):
        """Test scalar trait head."""
        from src.modules.heads import MultiTaskScalarHead

        head = MultiTaskScalarHead(
            input_dim=64,
            hidden_dim=32,
            trait_configs={
                "yield": {"range": (0, 20000)},
                "height": {"range": (100, 350)},
            },
        )

        x = torch.randn(4, 64)
        output = head(x)

        assert "yield" in output
        assert "height" in output


class TestLosses:
    """Test loss functions."""

    def test_multi_task_loss(self):
        """Test multi-task loss."""
        from src.losses import MultiTaskLoss

        loss_fn = MultiTaskLoss()

        predictions = {
            "yield": torch.randn(4),
        }
        targets = {
            "yields": torch.randn(4),
        }

        loss, loss_dict = loss_fn(predictions, targets)

        assert loss.item() >= 0
        assert "total" in loss_dict

    def test_curve_loss(self):
        """Test curve loss."""
        from src.losses import CurveLoss

        loss_fn = CurveLoss()

        pred = torch.randn(4, 4, 90)
        target = torch.randn(4, 4, 90)

        loss = loss_fn(pred, target)

        assert loss.item() >= 0


class TestData:
    """Test data loading."""

    def test_synthetic_generator(self):
        """Test synthetic data generation."""
        from src.data import SyntheticDataGenerator, SyntheticDataConfig

        config = SyntheticDataConfig(
            num_samples=10,
            num_snps=100,
            num_days=30,
        )

        generator = SyntheticDataGenerator(config)
        data = generator.generate()

        assert data["genotypes_a"].shape == (10, 100)
        assert data["genotypes_b"].shape == (10, 100)
        assert data["hourly_weather"].shape == (10, 30, 24, 8)
        assert data["yields"].shape == (10,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
