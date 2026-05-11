"""
Synthetic Data Generator for HyGE-Mamba.

This module generates realistic synthetic data for testing and development.
The generated data mimics real hybrid crop data with:
- Dual-parent genotypes with realistic LD patterns
- Hierarchical weather time series
- Correlated phenotypes

This allows testing the full pipeline without requiring real data.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""

    # Sample counts
    num_samples: int = 1000
    num_locations: int = 10
    num_years: int = 3

    # Genome parameters
    num_snps: int = 10000
    num_chromosomes: int = 10
    num_parents_a: int = 50  # Number of unique parent A lines
    num_parents_b: int = 50  # Number of unique parent B lines
    maf_min: float = 0.05  # Minimum minor allele frequency
    maf_max: float = 0.5   # Maximum minor allele frequency
    ld_decay_rate: float = 0.001  # LD decay with distance

    # Weather parameters
    num_days: int = 180  # Growing season length
    hours_per_day: int = 24
    num_weather_features: int = 8

    # Phenotype parameters
    yield_mean: float = 12.0  # t/ha
    yield_std: float = 2.5
    heritability: float = 0.6  # Proportion of variance due to genetics
    gxe_variance: float = 0.15  # GE interaction variance

    # APSIM parameter ranges
    apsim_ranges: Dict[str, Tuple[float, float]] = None

    # Curve parameters
    num_curve_timesteps: int = 180
    num_curves: int = 4

    # Scalar trait parameters
    num_scalar_traits: int = 5

    # Random seed
    seed: int = 42

    def __post_init__(self):
        if self.apsim_ranges is None:
            self.apsim_ranges = {
                "tt_emerg_to_endjuv": (200, 400),
                "tt_flower_to_maturity": (600, 1000),
                "max_grain_size": (250, 400),
                "radiation_use_efficiency": (1.5, 2.2),
                "transpiration_efficiency": (0.008, 0.012),
                "root_depth_rate": (2.0, 4.0),
            }


class SyntheticDataGenerator:
    """
    Generate synthetic hybrid crop data.

    Creates realistic synthetic data including:
    - Genotypes with linkage disequilibrium
    - Weather patterns with seasonal and diurnal variation
    - Phenotypes with genetic, environmental, and GE effects
    """

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Pre-generate parent genotypes for reuse
        self._parent_a_genotypes = None
        self._parent_b_genotypes = None
        self._snp_info = None
        self._location_effects = None
        self._year_effects = None

    def generate(self, output_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Generate complete synthetic dataset.

        Args:
            output_path: If provided, save to HDF5 file.

        Returns:
            Dictionary of generated arrays.
        """
        print("Generating synthetic HyGE-Mamba data...")

        # Generate parent genotypes
        print("  Generating parent genotypes...")
        self._generate_parent_genotypes()

        # Generate SNP information
        print("  Generating SNP positions...")
        self._generate_snp_info()

        # Generate environment effects
        print("  Generating environmental effects...")
        self._generate_environment_effects()

        # Generate samples
        print("  Generating samples...")
        data = self._generate_samples()

        # Save if output path provided
        if output_path is not None:
            print(f"  Saving to {output_path}...")
            self._save_hdf5(data, output_path)

        print("Done!")
        return data

    def _generate_parent_genotypes(self) -> None:
        """Generate parent genotypes with LD structure."""
        cfg = self.config

        # Generate allele frequencies for each SNP
        self._allele_freqs = self.rng.uniform(
            cfg.maf_min, cfg.maf_max, size=cfg.num_snps
        )

        # Generate parent A genotypes with LD
        self._parent_a_genotypes = self._generate_population_genotypes(
            cfg.num_parents_a
        )

        # Generate parent B genotypes with LD
        self._parent_b_genotypes = self._generate_population_genotypes(
            cfg.num_parents_b
        )

    def _generate_population_genotypes(self, num_individuals: int) -> np.ndarray:
        """Generate genotypes for a population with LD."""
        cfg = self.config

        # Start with random genotypes based on allele frequencies
        genotypes = np.zeros((num_individuals, cfg.num_snps), dtype=np.int8)

        for snp in range(cfg.num_snps):
            freq = self._allele_freqs[snp]
            # Sample diploid genotypes (0, 1, 2)
            alleles_1 = (self.rng.random(num_individuals) < freq).astype(np.int8)
            alleles_2 = (self.rng.random(num_individuals) < freq).astype(np.int8)
            genotypes[:, snp] = alleles_1 + alleles_2

        # Add LD by correlating nearby SNPs
        # This is a simplified model - real LD is more complex
        snps_per_chr = cfg.num_snps // cfg.num_chromosomes

        for chr_idx in range(cfg.num_chromosomes):
            start_snp = chr_idx * snps_per_chr
            end_snp = start_snp + snps_per_chr

            # Apply moving average to create LD blocks
            window = 5
            for i in range(start_snp + window, end_snp):
                # Mix current genotype with nearby average
                ld_weight = self.rng.uniform(0.1, 0.4)
                nearby_avg = genotypes[:, i-window:i].mean(axis=1)
                genotypes[:, i] = np.round(
                    (1 - ld_weight) * genotypes[:, i] + ld_weight * nearby_avg
                ).clip(0, 2).astype(np.int8)

        return genotypes

    def _generate_snp_info(self) -> None:
        """Generate SNP position information."""
        cfg = self.config

        snps_per_chr = cfg.num_snps // cfg.num_chromosomes
        chromosomes = []
        positions = []

        for chr_idx in range(cfg.num_chromosomes):
            n_snps = snps_per_chr if chr_idx < cfg.num_chromosomes - 1 else (
                cfg.num_snps - chr_idx * snps_per_chr
            )

            # Chromosome indices (1-based)
            chromosomes.extend([chr_idx] * n_snps)

            # Positions (roughly uniform across chromosome)
            chr_length = 300_000_000  # ~300 Mb
            pos = np.sort(self.rng.integers(0, chr_length, size=n_snps))
            positions.extend(pos.tolist())

        self._snp_info = {
            "chromosomes": np.array(chromosomes, dtype=np.int8),
            "positions": np.array(positions, dtype=np.int64),
        }

    def _generate_environment_effects(self) -> None:
        """Generate location and year effects."""
        cfg = self.config

        # Location effects (random environmental conditions)
        self._location_effects = self.rng.normal(0, 1, size=cfg.num_locations)

        # Year effects (random weather years)
        self._year_effects = self.rng.normal(0, 0.5, size=cfg.num_years)

    def _generate_samples(self) -> Dict[str, np.ndarray]:
        """Generate all samples."""
        cfg = self.config

        # Initialize arrays
        genotypes_a = np.zeros((cfg.num_samples, cfg.num_snps), dtype=np.int8)
        genotypes_b = np.zeros((cfg.num_samples, cfg.num_snps), dtype=np.int8)
        hourly_weather = np.zeros(
            (cfg.num_samples, cfg.num_days, cfg.hours_per_day, cfg.num_weather_features),
            dtype=np.float32,
        )
        day_indices = np.zeros((cfg.num_samples, cfg.num_days), dtype=np.int16)
        growth_stages = np.zeros((cfg.num_samples, cfg.num_days), dtype=np.int8)

        yields = np.zeros(cfg.num_samples, dtype=np.float32)
        apsim_params = np.zeros((cfg.num_samples, len(cfg.apsim_ranges)), dtype=np.float32)
        growth_curves = np.zeros(
            (cfg.num_samples, cfg.num_curves, cfg.num_curve_timesteps),
            dtype=np.float32,
        )
        scalar_traits = np.zeros((cfg.num_samples, cfg.num_scalar_traits), dtype=np.float32)

        locations = []
        years = []
        sample_ids = []

        # Generate each sample
        for i in range(cfg.num_samples):
            # Select parents
            parent_a_idx = self.rng.integers(0, cfg.num_parents_a)
            parent_b_idx = self.rng.integers(0, cfg.num_parents_b)

            genotypes_a[i] = self._parent_a_genotypes[parent_a_idx]
            genotypes_b[i] = self._parent_b_genotypes[parent_b_idx]

            # Select location and year
            loc_idx = self.rng.integers(0, cfg.num_locations)
            year_idx = self.rng.integers(0, cfg.num_years)

            locations.append(f"LOC_{loc_idx:02d}")
            years.append(2020 + year_idx)
            sample_ids.append(f"HYB_{i:05d}")

            # Generate weather for this sample
            hourly_weather[i], day_indices[i] = self._generate_weather(loc_idx, year_idx)

            # Generate growth stages
            growth_stages[i] = self._generate_growth_stages()

            # Generate phenotypes
            genetic_value = self._compute_genetic_value(
                genotypes_a[i], genotypes_b[i]
            )
            env_effect = (
                self._location_effects[loc_idx] +
                self._year_effects[year_idx]
            )

            # GE interaction
            gxe_effect = genetic_value * env_effect * self.rng.normal(0, cfg.gxe_variance)

            # Final yield
            yields[i] = cfg.yield_mean + genetic_value + env_effect + gxe_effect
            yields[i] += self.rng.normal(0, cfg.yield_std * (1 - cfg.heritability) ** 0.5)

            # APSIM parameters
            apsim_params[i] = self._generate_apsim_params(genetic_value)

            # Growth curves
            growth_curves[i] = self._generate_growth_curves(genetic_value, env_effect)

            # Scalar traits
            scalar_traits[i] = self._generate_scalar_traits(genetic_value, yields[i])

        return {
            "genotypes_a": genotypes_a,
            "genotypes_b": genotypes_b,
            # Store chromosomes and positions as shared 1D arrays.
            "chromosomes": self._snp_info["chromosomes"],
            "positions": self._snp_info["positions"],
            "hourly_weather": hourly_weather,
            "day_indices": day_indices,
            "growth_stages": growth_stages,
            "yields": yields,
            "apsim_params": apsim_params,
            "growth_curves": growth_curves,
            "scalar_traits": scalar_traits,
            "sample_ids": np.array(sample_ids, dtype="S20"),
            "locations": np.array(locations, dtype="S20"),
            "years": np.array(years, dtype=np.int16),
        }

    def _generate_weather(
        self, loc_idx: int, year_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate hourly weather for a growing season."""
        cfg = self.config

        weather = np.zeros(
            (cfg.num_days, cfg.hours_per_day, cfg.num_weather_features),
            dtype=np.float32,
        )

        # Day indices (planting day offset)
        planting_doy = 100 + self.rng.integers(-10, 10)
        day_indices = np.arange(planting_doy, planting_doy + cfg.num_days, dtype=np.int16)

        # Location-specific base conditions
        base_temp = 15 + loc_idx * 0.5  # Warmer locations have higher indices

        for day in range(cfg.num_days):
            doy = day_indices[day]

            # Seasonal temperature pattern
            seasonal_factor = np.sin(2 * np.pi * (doy - 80) / 365)
            day_temp = base_temp + 15 * seasonal_factor

            for hour in range(cfg.hours_per_day):
                # Diurnal temperature pattern
                diurnal_factor = np.cos(2 * np.pi * (hour - 14) / 24)
                temp = day_temp + 5 * diurnal_factor + self.rng.normal(0, 1)

                # Weather features: [temp, humidity, radiation, wind, precip, vpd, soil_temp, par]
                weather[day, hour, 0] = temp  # Temperature (C)
                weather[day, hour, 1] = 60 + 20 * self.rng.random() - 0.5 * temp  # Humidity (%)

                # Solar radiation (follows sun angle)
                if 6 <= hour <= 18:
                    sun_angle = np.sin(np.pi * (hour - 6) / 12)
                    weather[day, hour, 2] = 800 * sun_angle * (0.5 + 0.5 * self.rng.random())
                else:
                    weather[day, hour, 2] = 0

                weather[day, hour, 3] = 2 + 3 * self.rng.random()  # Wind speed (m/s)

                # Precipitation (sporadic)
                if self.rng.random() < 0.1:
                    weather[day, hour, 4] = self.rng.exponential(2)  # mm
                else:
                    weather[day, hour, 4] = 0

                # VPD (vapor pressure deficit)
                es = 0.611 * np.exp(17.27 * temp / (temp + 237.3))
                weather[day, hour, 5] = es * (1 - weather[day, hour, 1] / 100)

                weather[day, hour, 6] = temp - 2 + self.rng.normal(0, 0.5)  # Soil temp
                weather[day, hour, 7] = weather[day, hour, 2] * 0.5  # PAR

        return weather, day_indices

    def _generate_growth_stages(self) -> np.ndarray:
        """Generate growth stage progression."""
        cfg = self.config

        stages = np.zeros(cfg.num_days, dtype=np.int8)

        # Simplified growth stage progression
        # 0-1: VE-V6 (early vegetative)
        # 2-3: V6-VT (late vegetative)
        # 4: VT-R1 (flowering)
        # 5-6: R1-R3 (early grain fill)
        # 7-8: R3-R5 (late grain fill)
        # 9: R5-R6 (maturity)

        stage_lengths = [30, 25, 15, 10, 20, 20, 20, 20, 20]
        current_day = 0

        for stage, length in enumerate(stage_lengths):
            if current_day >= cfg.num_days:
                break
            end_day = min(current_day + length, cfg.num_days)
            stages[current_day:end_day] = min(stage, 9)
            current_day = end_day

        return stages

    def _compute_genetic_value(
        self, genotype_a: np.ndarray, genotype_b: np.ndarray
    ) -> float:
        """Compute genetic breeding value from genotypes."""
        cfg = self.config

        # Select QTL SNPs (5% of SNPs have effects)
        n_qtl = int(cfg.num_snps * 0.05)
        qtl_indices = self.rng.choice(cfg.num_snps, n_qtl, replace=False)

        # QTL effects
        qtl_effects = self.rng.normal(0, 1, size=n_qtl)

        # Additive value
        hybrid_geno = (genotype_a[qtl_indices] + genotype_b[qtl_indices]) / 2
        additive_value = np.sum(hybrid_geno * qtl_effects) / n_qtl ** 0.5

        # Dominance (heterosis effect)
        heterozygosity = np.abs(genotype_a[qtl_indices] - genotype_b[qtl_indices])
        dominance_effects = self.rng.normal(0, 0.3, size=n_qtl)
        dominance_value = np.sum(heterozygosity * dominance_effects) / n_qtl ** 0.5

        genetic_value = additive_value * cfg.yield_std * cfg.heritability ** 0.5
        genetic_value += dominance_value * 0.5  # Heterosis contribution

        return genetic_value

    def _generate_apsim_params(self, genetic_value: float) -> np.ndarray:
        """Generate APSIM parameters correlated with genetic value."""
        cfg = self.config

        params = []
        for name, (min_val, max_val) in cfg.apsim_ranges.items():
            # Base value in middle of range
            base = (min_val + max_val) / 2
            range_size = max_val - min_val

            # Genetic effect on parameter
            genetic_effect = genetic_value * 0.1 * range_size
            noise = self.rng.normal(0, 0.1 * range_size)

            value = np.clip(base + genetic_effect + noise, min_val, max_val)
            params.append(value)

        return np.array(params, dtype=np.float32)

    def _generate_growth_curves(
        self, genetic_value: float, env_effect: float
    ) -> np.ndarray:
        """Generate growth curves (LAI, biomass, grain fill, water stress)."""
        cfg = self.config

        t = np.linspace(0, 1, cfg.num_curve_timesteps)
        curves = np.zeros((cfg.num_curves, cfg.num_curve_timesteps), dtype=np.float32)

        # LAI curve (logistic growth then senescence)
        max_lai = 4.0 + genetic_value * 0.3 + env_effect * 0.2
        curves[0] = max_lai * np.exp(-((t - 0.5) ** 2) / 0.1)

        # Biomass accumulation (sigmoid)
        max_biomass = 20 + genetic_value * 2 + env_effect * 1.5
        curves[1] = max_biomass / (1 + np.exp(-10 * (t - 0.4)))

        # Grain fill (delayed sigmoid)
        max_grain = 0.5 * max_biomass
        curves[2] = max_grain / (1 + np.exp(-15 * (t - 0.6)))

        # Water stress (random events)
        stress = np.zeros(cfg.num_curve_timesteps)
        n_stress_events = self.rng.integers(0, 3)
        for _ in range(n_stress_events):
            center = self.rng.uniform(0.3, 0.8)
            width = self.rng.uniform(0.05, 0.15)
            intensity = self.rng.uniform(0.3, 0.8)
            stress += intensity * np.exp(-((t - center) ** 2) / (2 * width ** 2))
        curves[3] = np.clip(stress, 0, 1)

        # Add noise
        for i in range(cfg.num_curves):
            curves[i] += self.rng.normal(0, 0.02 * curves[i].max(), size=cfg.num_curve_timesteps)
            curves[i] = np.clip(curves[i], 0, None)

        return curves

    def _generate_scalar_traits(
        self, genetic_value: float, yield_value: float
    ) -> np.ndarray:
        """Generate scalar traits correlated with genetics and yield."""
        cfg = self.config

        # [yield, flowering_date, maturity_date, plant_height, harvest_index]
        traits = np.zeros(cfg.num_scalar_traits, dtype=np.float32)

        traits[0] = yield_value  # Yield (already computed)
        traits[1] = 70 + genetic_value * 2 + self.rng.normal(0, 3)  # Flowering date (days)
        traits[2] = 130 + genetic_value * 3 + self.rng.normal(0, 5)  # Maturity date
        traits[3] = 250 + genetic_value * 10 + self.rng.normal(0, 15)  # Plant height (cm)
        traits[4] = 0.45 + genetic_value * 0.02 + self.rng.normal(0, 0.03)  # Harvest index

        return traits

    def _save_hdf5(self, data: Dict[str, np.ndarray], path: str) -> None:
        """Save data to HDF5 file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value, compression="gzip")
            f.attrs['is_shared_genomic_coords'] = True


def generate_synthetic_data(
    output_path: str = "data/synthetic_hyge.h5",
    num_samples: int = 1000,
    num_snps: int = 10000,
    num_days: int = 180,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to generate synthetic data.

    Args:
        output_path: Path to save HDF5 file.
        num_samples: Number of samples to generate.
        num_snps: Number of SNPs per sample.
        num_days: Number of days in growing season.
        seed: Random seed.

    Returns:
        Dictionary of generated data.
    """
    config = SyntheticDataConfig(
        num_samples=num_samples,
        num_snps=num_snps,
        num_days=num_days,
        seed=seed,
    )

    generator = SyntheticDataGenerator(config)
    return generator.generate(output_path)


if __name__ == "__main__":
    # Generate sample data when run directly
    generate_synthetic_data(
        output_path="data/synthetic_hyge.h5",
        num_samples=100,
        num_snps=1000,
        num_days=180,
    )
