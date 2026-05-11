"""
Dataset module for HyGE-Mamba.

This module provides PyTorch Dataset classes for loading and processing
hybrid crop data including:
- Dual-parent genotype data (VCF format)
- Hierarchical weather time series
- Phenotype/yield data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any, Union
from dataclasses import dataclass
import json


@dataclass
class HyGESample:
    """Container for a single HyGE-Mamba sample."""

    # Genome data
    genotypes_a: torch.Tensor  # Parent A genotypes, shape (num_snps,)
    genotypes_b: torch.Tensor  # Parent B genotypes, shape (num_snps,)
    chromosomes: torch.Tensor  # Chromosome indices, shape (num_snps,)
    positions: torch.Tensor  # Base pair positions, shape (num_snps,)

    # Weather data
    hourly_weather: torch.Tensor  # Shape (num_days, 24, num_features)
    day_indices: torch.Tensor  # Day-of-year indices, shape (num_days,)
    growth_stages: Optional[torch.Tensor] = None  # Growth stage indices

    # Labels
    yield_value: Optional[torch.Tensor] = None
    apsim_params: Optional[torch.Tensor] = None
    growth_curves: Optional[torch.Tensor] = None
    scalar_traits: Optional[torch.Tensor] = None

    # Masks
    snp_mask: Optional[torch.Tensor] = None
    day_mask: Optional[torch.Tensor] = None
    hour_mask: Optional[torch.Tensor] = None

    # Metadata
    hybrid_id: Optional[str] = None
    location_id: Optional[str] = None
    year: Optional[int] = None


class HyGEDataset(Dataset):
    """
    PyTorch Dataset for HyGE-Mamba training data.

    Supports loading from:
    - HDF5 files (recommended for large datasets)
    - NumPy arrays (for smaller datasets)
    - Synthetic data generation

    Args:
        data_path: Path to data file (HDF5) or directory.
        genome_path: Optional separate path for genome data.
        weather_path: Optional separate path for weather data.
        phenotype_path: Optional separate path for phenotype data.
        max_snps: Maximum number of SNPs to load (for memory management).
        max_days: Maximum number of days in growing season.
        hours_per_day: Number of hours per day (default 24).
        transform: Optional transform to apply to samples.
        cache_genome: If True, cache genome data in memory.
        preload: If True, preload all data into memory.
    """

    def __init__(
            self,
            data_path: Optional[Union[str, Path]] = None,
            genome_path: Optional[Union[str, Path]] = None,
            weather_path: Optional[Union[str, Path]] = None,
            phenotype_path: Optional[Union[str, Path]] = None,
            max_snps: int = 100000,
            max_days: int = 180,
            hours_per_day: int = 24,
            transform: Optional[Any] = None,
            cache_genome: bool = True,
            preload: bool = False,
    ) -> None:
        super().__init__()

        self.max_snps = max_snps
        self.max_days = max_days
        self.hours_per_day = hours_per_day
        self.transform = transform
        self.cache_genome = cache_genome

        # Storage for data
        self.samples: List[Dict[str, Any]] = []
        self.genome_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Load data if paths provided
        if data_path is not None:
            self._load_data(Path(data_path))
        elif genome_path is not None and weather_path is not None:
            self._load_separate_data(
                Path(genome_path) if genome_path else None,
                Path(weather_path) if weather_path else None,
                Path(phenotype_path) if phenotype_path else None,
            )

        if preload:
            self._preload_all()

    def _load_data(self, data_path: Path) -> None:
        """Load data from HDF5 file."""
        if data_path.suffix in [".h5", ".hdf5"]:
            self._load_hdf5(data_path)
        elif data_path.is_dir():
            self._load_directory(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

    def _load_hdf5(self, path: Path) -> None:
        """Load data from HDF5 file.

        Optimizations:
        - Load shared 1D chromosome/position arrays once, while keeping
          compatibility with legacy 2D per-sample arrays.
        - Keep the HDF5 handle open to avoid repeated open/close calls in
          __getitem__.
        """
        self._h5_path = str(path)
        self._h5_file = None  # Lazy-opened per DataLoader worker.

        with h5py.File(path, "r") as f:
            # Get sample IDs
            if "sample_ids" in f:
                sample_ids = [s.decode() if isinstance(s, bytes) else str(s)
                              for s in f["sample_ids"][:]]
            else:
                sample_ids = [str(i) for i in range(len(f["genotypes_a"]))]

            # Load shared chromosome and position coordinates.
            is_shared = f.attrs.get('is_shared_genomic_coords', False)
            if "chromosomes" in f:
                if is_shared or f["chromosomes"].ndim == 1:
                    self._shared_chromosomes = torch.tensor(
                        f["chromosomes"][:self.max_snps], dtype=torch.long)
                else:
                    # Legacy 2D format: use the first sample row.
                    self._shared_chromosomes = torch.tensor(
                        f["chromosomes"][0][:self.max_snps], dtype=torch.long)
            else:
                self._shared_chromosomes = None

            if "positions" in f:
                if is_shared or f["positions"].ndim == 1:
                    self._shared_positions = torch.tensor(
                        f["positions"][:self.max_snps], dtype=torch.long)
                else:
                    self._shared_positions = torch.tensor(
                        f["positions"][0][:self.max_snps], dtype=torch.long)
            else:
                self._shared_positions = None

            for i, sample_id in enumerate(sample_ids):
                sample_info = {
                    "index": i,
                    "hybrid_id": sample_id,
                    "data_file": str(path),
                }

                # Store metadata if available
                if "locations" in f:
                    sample_info["location_id"] = f["locations"][i].decode() if isinstance(f["locations"][i],
                                                                                          bytes) else str(
                        f["locations"][i])
                if "years" in f:
                    sample_info["year"] = int(f["years"][i])

                self.samples.append(sample_info)

    @property
    def _h5(self):
        """Return the worker-local HDF5 file handle."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self._h5_path, "r", swmr=True)
        return self._h5_file

    def __del__(self):
        if hasattr(self, '_h5_file') and self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass

    def _load_directory(self, path: Path) -> None:
        """Load data from directory structure."""
        # Look for manifest file
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            for sample in manifest["samples"]:
                self.samples.append({
                    "hybrid_id": sample["hybrid_id"],
                    "genome_file": path / sample.get("genome_file", f"genomes/{sample['hybrid_id']}.npz"),
                    "weather_file": path / sample.get("weather_file", f"weather/{sample['hybrid_id']}.npz"),
                    "phenotype_file": path / sample.get("phenotype_file", f"phenotypes/{sample['hybrid_id']}.npz"),
                })

    def _load_separate_data(
            self,
            genome_path: Optional[Path],
            weather_path: Optional[Path],
            phenotype_path: Optional[Path],
    ) -> None:
        """Load data from separate files."""
        # This would handle separate genome/weather/phenotype files
        # Implementation depends on specific data format
        pass

    def _preload_all(self) -> None:
        """Preload all data into memory."""
        for i in range(len(self.samples)):
            self._load_sample(i)

    def _load_sample(self, idx: int) -> HyGESample:
        """Load a single sample from storage."""
        sample_info = self.samples[idx]

        if "data_file" in sample_info:
            return self._load_sample_hdf5(sample_info)
        else:
            return self._load_sample_files(sample_info)

    def _load_sample_hdf5(self, sample_info: Dict[str, Any]) -> HyGESample:
        """Load sample from HDF5 file.

        Uses cached shared chromosomes/positions and persistent file handle.
        """
        idx = sample_info["index"]
        f = self._h5

        # Load genome data (only genotypes need per-sample read)
        genotypes_a = torch.tensor(f["genotypes_a"][idx][:self.max_snps], dtype=torch.long)
        genotypes_b = torch.tensor(f["genotypes_b"][idx][:self.max_snps], dtype=torch.long)

        # Use shared chromosomes/positions (already cached in memory)
        num_snps = len(genotypes_a)
        if self._shared_chromosomes is not None:
            chromosomes = self._shared_chromosomes[:num_snps]
        else:
            chromosomes = torch.zeros(num_snps, dtype=torch.long)

        if self._shared_positions is not None:
            positions = self._shared_positions[:num_snps]
        else:
            positions = torch.arange(num_snps, dtype=torch.long)

        # Load weather data
        hourly_weather = torch.tensor(
            f["hourly_weather"][idx][:self.max_days], dtype=torch.float32
        )

        if "day_indices" in f:
            day_indices = torch.tensor(f["day_indices"][idx][:self.max_days], dtype=torch.long)
        else:
            day_indices = torch.arange(hourly_weather.shape[0], dtype=torch.long)

        growth_stages = None
        if "growth_stages" in f:
            growth_stages = torch.tensor(f["growth_stages"][idx][:self.max_days], dtype=torch.long)

        # Load labels
        yield_value = None
        if "yields" in f:
            yield_value = torch.tensor(f["yields"][idx], dtype=torch.float32)

        apsim_params = None
        if "apsim_params" in f:
            apsim_params = torch.tensor(f["apsim_params"][idx], dtype=torch.float32)

        growth_curves = None
        if "growth_curves" in f:
            growth_curves = torch.tensor(f["growth_curves"][idx], dtype=torch.float32)

        scalar_traits = None
        if "scalar_traits" in f:
            scalar_traits = torch.tensor(f["scalar_traits"][idx], dtype=torch.float32)

        # Create masks
        snp_mask = genotypes_a >= 0  # Valid SNPs
        day_mask = day_indices >= 0  # Valid days
        hour_mask = hourly_weather.sum(dim=-1) != 0  # Valid hours

        return HyGESample(
            genotypes_a=genotypes_a,
            genotypes_b=genotypes_b,
            chromosomes=chromosomes,
            positions=positions,
            hourly_weather=hourly_weather,
            day_indices=day_indices,
            growth_stages=growth_stages,
            yield_value=yield_value,
            apsim_params=apsim_params,
            growth_curves=growth_curves,
            scalar_traits=scalar_traits,
            snp_mask=snp_mask,
            day_mask=day_mask,
            hour_mask=hour_mask,
            hybrid_id=sample_info.get("hybrid_id"),
            location_id=sample_info.get("location_id"),
            year=sample_info.get("year"),
        )

    def _load_sample_files(self, sample_info: Dict[str, Any]) -> HyGESample:
        """Load sample from separate files."""
        # Implementation for file-based loading
        raise NotImplementedError("File-based loading not implemented")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> HyGESample:
        sample = self._load_sample(idx)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class HyGECollator:
    """
    Collator for batching HyGE samples.

    Handles variable-length sequences with padding.

    Args:
        pad_snp_value: Value to use for padding SNPs (default -1).
        pad_weather_value: Value to use for padding weather (default 0).
        max_snps: Maximum SNPs in batch (truncate if longer).
        max_days: Maximum days in batch.
    """

    def __init__(
            self,
            pad_snp_value: int = -1,
            pad_weather_value: float = 0.0,
            max_snps: Optional[int] = None,
            max_days: Optional[int] = None,
    ) -> None:
        self.pad_snp_value = pad_snp_value
        self.pad_weather_value = pad_weather_value
        self.max_snps = max_snps
        self.max_days = max_days

    def __call__(self, batch: List[HyGESample]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples."""
        # Find max lengths
        max_snps = max(s.genotypes_a.shape[0] for s in batch)
        max_days = max(s.hourly_weather.shape[0] for s in batch)
        hours_per_day = batch[0].hourly_weather.shape[1]
        num_features = batch[0].hourly_weather.shape[2]

        if self.max_snps is not None:
            max_snps = min(max_snps, self.max_snps)
        if self.max_days is not None:
            max_days = min(max_days, self.max_days)

        batch_size = len(batch)

        # Initialize tensors
        genotypes_a = torch.full((batch_size, max_snps), self.pad_snp_value, dtype=torch.long)
        genotypes_b = torch.full((batch_size, max_snps), self.pad_snp_value, dtype=torch.long)
        chromosomes = torch.zeros((batch_size, max_snps), dtype=torch.long)
        positions = torch.zeros((batch_size, max_snps), dtype=torch.long)
        snp_mask = torch.zeros((batch_size, max_snps), dtype=torch.bool)

        hourly_weather = torch.full(
            (batch_size, max_days, hours_per_day, num_features),
            self.pad_weather_value,
            dtype=torch.float32,
        )
        day_indices = torch.zeros((batch_size, max_days), dtype=torch.long)
        day_mask = torch.zeros((batch_size, max_days), dtype=torch.bool)
        hour_mask = torch.zeros((batch_size, max_days, hours_per_day), dtype=torch.bool)

        # Optional tensors
        has_growth_stages = batch[0].growth_stages is not None
        has_yield = batch[0].yield_value is not None
        has_apsim = batch[0].apsim_params is not None
        has_curves = batch[0].growth_curves is not None
        has_scalars = batch[0].scalar_traits is not None

        growth_stages = None
        if has_growth_stages:
            growth_stages = torch.zeros((batch_size, max_days), dtype=torch.long)

        yields = None
        if has_yield:
            yields = torch.zeros(batch_size, dtype=torch.float32)

        apsim_params = None
        if has_apsim:
            apsim_params = torch.zeros((batch_size, batch[0].apsim_params.shape[0]), dtype=torch.float32)

        growth_curves = None
        if has_curves:
            growth_curves = torch.zeros(
                (batch_size, *batch[0].growth_curves.shape),
                dtype=torch.float32,
            )

        scalar_traits = None
        if has_scalars:
            scalar_traits = torch.zeros(
                (batch_size, batch[0].scalar_traits.shape[0]),
                dtype=torch.float32,
            )

        # Fill tensors
        for i, sample in enumerate(batch):
            # SNPs
            n_snps = min(sample.genotypes_a.shape[0], max_snps)
            genotypes_a[i, :n_snps] = sample.genotypes_a[:n_snps]
            genotypes_b[i, :n_snps] = sample.genotypes_b[:n_snps]
            chromosomes[i, :n_snps] = sample.chromosomes[:n_snps]
            positions[i, :n_snps] = sample.positions[:n_snps]
            snp_mask[i, :n_snps] = True

            # Weather
            n_days = min(sample.hourly_weather.shape[0], max_days)
            hourly_weather[i, :n_days] = sample.hourly_weather[:n_days]
            day_indices[i, :n_days] = sample.day_indices[:n_days]
            day_mask[i, :n_days] = True
            if sample.hour_mask is not None:
                hour_mask[i, :n_days] = sample.hour_mask[:n_days]
            else:
                hour_mask[i, :n_days] = True

            # Optional
            if has_growth_stages and sample.growth_stages is not None:
                growth_stages[i, :n_days] = sample.growth_stages[:n_days]

            if has_yield and sample.yield_value is not None:
                yields[i] = sample.yield_value

            if has_apsim and sample.apsim_params is not None:
                apsim_params[i] = sample.apsim_params

            if has_curves and sample.growth_curves is not None:
                growth_curves[i] = sample.growth_curves

            if has_scalars and sample.scalar_traits is not None:
                scalar_traits[i] = sample.scalar_traits

        # Build output dict
        output = {
            "genotypes_a": genotypes_a,
            "genotypes_b": genotypes_b,
            "chromosomes": chromosomes,
            "positions": positions,
            "snp_mask": snp_mask,
            "hourly_weather": hourly_weather,
            "day_indices": day_indices,
            "day_mask": day_mask,
            "hour_mask": hour_mask,
        }

        if growth_stages is not None:
            output["growth_stages"] = growth_stages

        if yields is not None:
            output["yields"] = yields

        if apsim_params is not None:
            output["apsim_params"] = apsim_params

        if growth_curves is not None:
            output["growth_curves"] = growth_curves

        if scalar_traits is not None:
            output["scalar_traits"] = scalar_traits

        return output
