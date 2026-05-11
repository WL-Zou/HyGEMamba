"""
DataLoader utilities for HyGE-Mamba.

This module provides functions for creating DataLoaders with appropriate
configurations for training, validation, and testing.
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from typing import Optional, Dict, Tuple, List, Any
import numpy as np

from .dataset import HyGEDataset, HyGECollator


def create_dataloaders(
        data_path: str,
        batch_size: int = 32,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        max_snps: int = 100000,
        max_days: int = 180,
        **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        data_path: Path to data file or directory.
        batch_size: Batch size for training.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        num_workers: Number of data loading workers.
        pin_memory: If True, pin memory for faster GPU transfer.
        seed: Random seed for splitting.
        max_snps: Maximum number of SNPs.
        max_days: Maximum number of days.
        **dataset_kwargs: Additional arguments for HyGEDataset.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-5

    # Create full dataset
    dataset = HyGEDataset(
        data_path=data_path,
        max_snps=max_snps,
        max_days=max_days,
        **dataset_kwargs,
    )

    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )

    # Create collator
    collator = HyGECollator(max_snps=max_snps, max_days=max_days)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )

    return train_loader, val_loader, test_loader


def create_cross_validation_loaders(
        data_path: str,
        n_folds: int = 5,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        **dataset_kwargs,
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Create DataLoaders for k-fold cross-validation.

    Args:
        data_path: Path to data.
        n_folds: Number of folds.
        batch_size: Batch size.
        num_workers: Number of workers.
        pin_memory: If True, pin memory.
        seed: Random seed.
        **dataset_kwargs: Additional dataset arguments.

    Returns:
        List of (train_loader, val_loader) tuples for each fold.
    """
    dataset = HyGEDataset(data_path=data_path, **dataset_kwargs)
    collator = HyGECollator()

    n_total = len(dataset)
    indices = np.arange(n_total)

    # Shuffle indices
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    fold_size = n_total // n_folds
    loaders = []

    for fold in range(n_folds):
        # Validation indices for this fold
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_total
        val_indices = indices[val_start:val_end].tolist()

        # Training indices (everything else)
        train_indices = np.concatenate([
            indices[:val_start],
            indices[val_end:],
        ]).tolist()

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collator,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collator,
        )

        loaders.append((train_loader, val_loader))

    return loaders


def create_stratified_loaders(
        data_path: str,
        stratify_by: str = "location",
        batch_size: int = 32,
        train_split: float = 0.8,
        num_workers: int = 4,
        seed: int = 42,
        **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders with stratified splitting.

    Useful for ensuring balanced representation of locations,
    years, or other categorical variables.

    Args:
        data_path: Path to data.
        stratify_by: Field to stratify by ('location', 'year', 'hybrid').
        batch_size: Batch size.
        train_split: Fraction for training.
        num_workers: Number of workers.
        seed: Random seed.
        **dataset_kwargs: Additional dataset arguments.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    dataset = HyGEDataset(data_path=data_path, **dataset_kwargs)
    collator = HyGECollator()

    # Get stratification groups
    if stratify_by == "location":
        groups = [s.get("location_id", "unknown") for s in dataset.samples]
    elif stratify_by == "year":
        groups = [s.get("year", 0) for s in dataset.samples]
    elif stratify_by == "hybrid":
        groups = [s.get("hybrid_id", "unknown") for s in dataset.samples]
    else:
        raise ValueError(f"Unknown stratify_by: {stratify_by}")

    # Group indices
    from collections import defaultdict
    group_indices = defaultdict(list)
    for i, g in enumerate(groups):
        group_indices[g].append(i)

    # Split each group
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for g, indices in group_indices.items():
        rng.shuffle(indices)
        n_train = int(len(indices) * train_split)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])

    # Shuffle final indices
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return train_loader, val_loader


class DistributedSamplerWrapper:
    """
    Wrapper for distributed training samplers.

    Ensures proper shuffling and epoch management for distributed training.
    """

    def __init__(
            self,
            dataset: HyGEDataset,
            num_replicas: int,
            rank: int,
            shuffle: bool = True,
            seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Compute samples per replica
        self.total_size = len(dataset)
        self.num_samples = int(np.ceil(self.total_size / self.num_replicas))

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = list(range(self.total_size))

        # Pad to make divisible
        padding_size = self.num_samples * self.num_replicas - self.total_size
        indices += indices[:padding_size]

        # Subsample for this replica
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch