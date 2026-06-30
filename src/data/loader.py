"""
DataLoader factory for per-client federated learning data.

Loads pre-partitioned .pt files from disk and wraps them in
PyTorch DataLoader objects ready for training and evaluation.

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.partitioner import (
    _get_partition_dir,
    _load_test_dataset,
    DATA_RAW_DIR,
)


def get_client_dataloader(
    client_id: int,
    dataset_name: str,
    alpha: float,
    seed: int,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Load a single client's partitioned data as a DataLoader.

    Args:
        client_id: Client index (0-indexed).
        dataset_name: "mnist" or "cifar10".
        alpha: Dirichlet alpha used when partitioning.
        seed: Seed used when partitioning.
        batch_size: Mini-batch size for training.
        shuffle: Whether to shuffle data each epoch.

    Returns:
        torch.utils.data.DataLoader over this client's data.

    Raises:
        FileNotFoundError: If the partition file does not exist.
        RuntimeError: If the saved file is corrupt or missing keys.
    """
    part_dir = _get_partition_dir(dataset_name, alpha, seed)
    pt_file = part_dir / f"client_{client_id}.pt"

    if not pt_file.exists():
        raise FileNotFoundError(
            f"Partition file not found: {pt_file}. "
            f"Run experiments/prepare_data.py first."
        )

    saved = torch.load(pt_file, weights_only=True)
    if "data" not in saved or "labels" not in saved:
        raise RuntimeError(f"Corrupt partition file: {pt_file}")

    dataset = TensorDataset(saved["data"], saved["labels"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_test_dataloader(
    dataset_name: str,
    batch_size: int = 256,
    data_dir: Path = None,
) -> DataLoader:
    """
    Return a DataLoader over the centralized test set.

    This is used for global evaluation after each communication round.
    The same test set is shared across all strategies for fair comparison.

    Args:
        dataset_name: "mnist" or "cifar10".
        batch_size: Batch size for evaluation (larger is fine — no grads).
        data_dir: Path to raw dataset directory.

    Returns:
        torch.utils.data.DataLoader over the full test set.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    test_dataset = _load_test_dataset(dataset_name, Path(data_dir))
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_client_data_info(
    client_id: int,
    dataset_name: str,
    alpha: float,
    seed: int,
) -> dict:
    """
    Return metadata about a client's partition without loading full tensors.

    Args:
        client_id: Client index.
        dataset_name: "mnist" or "cifar10".
        alpha: Dirichlet alpha.
        seed: Partition seed.

    Returns:
        Dict with keys: n_samples, input_shape, classes, labels_tensor.
    """
    part_dir = _get_partition_dir(dataset_name, alpha, seed)
    pt_file = part_dir / f"client_{client_id}.pt"

    if not pt_file.exists():
        raise FileNotFoundError(f"Partition not found: {pt_file}")

    saved = torch.load(pt_file, weights_only=True)
    data: torch.Tensor = saved["data"]
    labels: torch.Tensor = saved["labels"]

    return {
        "n_samples": len(data),
        "input_shape": tuple(data.shape[1:]),   # (C, H, W)
        "classes": sorted(labels.unique().tolist()),
        "labels_tensor": labels,
    }
