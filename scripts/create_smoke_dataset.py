"""Create a tiny HDF5 dataset for installation and training smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def create_smoke_dataset(output: Path) -> None:
    """Write a deterministic dataset that matches the train.py HDF5 schema."""

    rng = np.random.default_rng(2026)
    num_samples = 12
    num_snps = 1024
    num_days = 30
    num_weather_features = 8
    curve_steps = 30

    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "w") as handle:
        handle.attrs["num_samples"] = num_samples
        handle.attrs["num_snps"] = num_snps
        handle.attrs["num_days"] = num_days
        handle.attrs["num_weather_features"] = num_weather_features
        handle.attrs["is_shared_genomic_coords"] = True

        handle.create_dataset(
            "genotypes_a",
            data=rng.integers(0, 3, size=(num_samples, num_snps), dtype=np.int8),
        )
        handle.create_dataset(
            "genotypes_b",
            data=rng.integers(0, 3, size=(num_samples, num_snps), dtype=np.int8),
        )
        handle.create_dataset(
            "chromosomes",
            data=rng.integers(1, 11, size=(num_snps,), dtype=np.int16),
        )
        handle.create_dataset(
            "positions",
            data=np.linspace(1, 100_000_000, num_snps, dtype=np.int64),
        )
        handle.create_dataset(
            "weather_data",
            data=rng.normal(size=(num_samples, num_days, num_weather_features)).astype(
                np.float32
            ),
        )
        handle.create_dataset(
            "targets",
            data=rng.normal(size=(num_samples, 4, curve_steps)).astype(np.float32),
        )
        handle.create_dataset(
            "scalar_targets",
            data=rng.normal(size=(num_samples, 3)).astype(np.float32),
        )
        handle.create_dataset(
            "apsim_params",
            data=rng.normal(size=(num_samples, 6)).astype(np.float32),
        )

        splits = handle.create_group("splits")
        for split_name in ("M1", "M2", "M3", "M4", "M5"):
            split = splits.create_group(split_name)
            split.attrs["description"] = f"{split_name} smoke-test split"
            split.attrs["use_genotype"] = split_name != "M4"
            split.attrs["use_environment"] = split_name != "M5"
            split.create_dataset("train_indices", data=np.arange(0, 8, dtype=np.int64))
            split.create_dataset("val_indices", data=np.arange(8, 10, dtype=np.int64))
            split.create_dataset("test_indices", data=np.arange(10, 12, dtype=np.int64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tiny HyGEMamba-compatible HDF5 smoke-test dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/hygemamba_smoke.h5"),
        help="Path to the output HDF5 file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_smoke_dataset(args.output)
    print(f"Wrote smoke dataset to {args.output}")


if __name__ == "__main__":
    main()
