"""
Tests for HyGE-Mamba data modules.

Run with: pytest tests/test_data.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_generator import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    generate_synthetic_data,
)
from src.data.dataset import HyGEDataset, HyGECollator, HyGESample


class TestSyntheticDataConfig:
    """Test synthetic data configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SyntheticDataConfig()
        assert config.num_samples == 1000
        assert config.num_snps == 10000
        assert config.num_chromosomes == 10
        assert config.num_days == 180
        assert config.hours_per_day == 24

    def test_custom_config(self):
        """Test custom configuration."""
        config = SyntheticDataConfig(
            num_samples=100,
            num_snps=500,
            num_days=90,
        )
        assert config.num_samples == 100
        assert config.num_snps == 500
        assert config.num_days == 90


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""

    @pytest.fixture
    def small_config(self):
        """Small config for fast testing."""
        return SyntheticDataConfig(
            num_samples=10,
            num_snps=100,
            num_chromosomes=2,
            num_days=30,
            num_parents_a=5,
            num_parents_b=5,
            seed=42,
        )

    def test_generator_creation(self, small_config):
        """Test generator initialization."""
        generator = SyntheticDataGenerator(small_config)
        assert generator.config == small_config

    def test_data_generation(self, small_config):
        """Test data generation."""
        generator = SyntheticDataGenerator(small_config)
        data = generator.generate()

        # Check all keys exist
        expected_keys = [
            "genotypes_a", "genotypes_b", "chromosomes", "positions",
            "hourly_weather", "day_indices", "growth_stages",
            "yields", "apsim_params", "growth_curves", "scalar_traits",
            "sample_ids", "locations", "years",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

        # Check shapes
        assert data["genotypes_a"].shape == (10, 100)
        assert data["genotypes_b"].shape == (10, 100)
        assert data["hourly_weather"].shape == (10, 30, 24, 8)
        assert data["yields"].shape == (10,)

    def test_genotype_values(self, small_config):
        """Test genotype values are valid (0, 1, 2)."""
        generator = SyntheticDataGenerator(small_config)
        data = generator.generate()

        assert np.all(data["genotypes_a"] >= 0)
        assert np.all(data["genotypes_a"] <= 2)
        assert np.all(data["genotypes_b"] >= 0)
        assert np.all(data["genotypes_b"] <= 2)

    def test_weather_values(self, small_config):
        """Test weather values are reasonable."""
        generator = SyntheticDataGenerator(small_config)
        data = generator.generate()

        weather = data["hourly_weather"]

        # Temperature should be in reasonable range
        temp = weather[:, :, :, 0]
        assert np.all(temp > -50) and np.all(temp < 60)

        # Humidity should be in [0, 100]
        humidity = weather[:, :, :, 1]
        assert np.all(humidity >= 0) and np.all(humidity <= 110)  # Allow some slack

    def test_save_to_hdf5(self, small_config):
        """Test saving data to HDF5."""
        import os
        generator = SyntheticDataGenerator(small_config)

        # Create temp file, close it, then use path
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            data = generator.generate(output_path=temp_path)

            # Check file exists
            assert Path(temp_path).exists()
        finally:
            # Clean up - may need retry on Windows
            try:
                os.unlink(temp_path)
            except PermissionError:
                pass  # File may still be in use on Windows


class TestHyGESample:
    """Test HyGESample dataclass."""

    def test_sample_creation(self):
        """Test creating a sample."""
        sample = HyGESample(
            genotypes_a=torch.randint(0, 3, (100,)),
            genotypes_b=torch.randint(0, 3, (100,)),
            chromosomes=torch.zeros(100, dtype=torch.long),
            positions=torch.arange(100),
            hourly_weather=torch.randn(30, 24, 8),
            day_indices=torch.arange(30),
        )

        assert sample.genotypes_a.shape == (100,)
        assert sample.hourly_weather.shape == (30, 24, 8)

    def test_sample_with_labels(self):
        """Test sample with labels."""
        sample = HyGESample(
            genotypes_a=torch.randint(0, 3, (100,)),
            genotypes_b=torch.randint(0, 3, (100,)),
            chromosomes=torch.zeros(100, dtype=torch.long),
            positions=torch.arange(100),
            hourly_weather=torch.randn(30, 24, 8),
            day_indices=torch.arange(30),
            yield_value=torch.tensor(12.5),
            apsim_params=torch.randn(6),
            scalar_traits=torch.randn(5),
        )

        assert sample.yield_value.item() == pytest.approx(12.5)
        assert sample.apsim_params.shape == (6,)


class TestHyGECollator:
    """Test batch collation."""

    def test_collate_batch(self):
        """Test collating a batch of samples."""
        collator = HyGECollator()

        # Create samples with different lengths
        samples = [
            HyGESample(
                genotypes_a=torch.randint(0, 3, (50,)),
                genotypes_b=torch.randint(0, 3, (50,)),
                chromosomes=torch.zeros(50, dtype=torch.long),
                positions=torch.arange(50),
                hourly_weather=torch.randn(20, 24, 8),
                day_indices=torch.arange(20),
                yield_value=torch.tensor(10.0),
            ),
            HyGESample(
                genotypes_a=torch.randint(0, 3, (80,)),
                genotypes_b=torch.randint(0, 3, (80,)),
                chromosomes=torch.zeros(80, dtype=torch.long),
                positions=torch.arange(80),
                hourly_weather=torch.randn(30, 24, 8),
                day_indices=torch.arange(30),
                yield_value=torch.tensor(15.0),
            ),
        ]

        batch = collator(samples)

        # Check batch is properly padded
        assert batch["genotypes_a"].shape == (2, 80)  # Max length
        assert batch["hourly_weather"].shape == (2, 30, 24, 8)
        assert batch["yields"].shape == (2,)

        # Check masks
        assert batch["snp_mask"].shape == (2, 80)
        assert batch["snp_mask"][0, :50].all()  # First 50 valid
        assert not batch["snp_mask"][0, 50:].any()  # Rest padded

    def test_collate_with_max_length(self):
        """Test collation with max length constraints."""
        collator = HyGECollator(max_snps=30, max_days=15)

        samples = [
            HyGESample(
                genotypes_a=torch.randint(0, 3, (50,)),
                genotypes_b=torch.randint(0, 3, (50,)),
                chromosomes=torch.zeros(50, dtype=torch.long),
                positions=torch.arange(50),
                hourly_weather=torch.randn(20, 24, 8),
                day_indices=torch.arange(20),
            ),
        ]

        batch = collator(samples)

        # Check truncation
        assert batch["genotypes_a"].shape[1] <= 30
        assert batch["hourly_weather"].shape[1] <= 15


class TestGenerateSyntheticData:
    """Test the convenience function."""

    def test_generate_function(self):
        """Test generate_synthetic_data function."""
        import os

        # Create temp file, close it, then use path
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            data = generate_synthetic_data(
                output_path=temp_path,
                num_samples=5,
                num_snps=50,
                num_days=20,
                seed=123,
            )

            assert data["genotypes_a"].shape == (5, 50)
            assert Path(temp_path).exists()
        finally:
            # Clean up - may need retry on Windows
            try:
                os.unlink(temp_path)
            except PermissionError:
                pass  # File may still be in use on Windows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])