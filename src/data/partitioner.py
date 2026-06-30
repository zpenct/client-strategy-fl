"""
Dirichlet-based data partitioner for non-IID federated learning experiments.

Partitions are generated once and saved to disk as .pt files to ensure
reproducibility across all experiment runs without re-computation.

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from src.utils.tracer import trace_partition, TRACE_MODE


# Default paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARTITION_DIR = PROJECT_ROOT / "data" / "partitions"


def _get_partition_dir(dataset_name: str, alpha: float, seed: int) -> Path:
    """Return the directory for a specific partition configuration."""
    alpha_str = str(alpha).replace(".", "")
    # e.g. data/partitions/mnist/alpha01_seed42/
    return PARTITION_DIR / dataset_name / f"alpha{alpha_str}_seed{seed}"


def check_partition_exists(dataset_name: str, alpha: float, seed: int,
                           num_clients: int = 10) -> bool:
    """
    Check whether all client partition files already exist on disk.

    Args:
        dataset_name: "mnist" or "cifar10".
        alpha: Dirichlet concentration parameter.
        seed: Random seed used to generate the partition.
        num_clients: Expected number of client files.

    Returns:
        True if all client_{i}.pt files and partition_info.json exist.
    """
    part_dir = _get_partition_dir(dataset_name, alpha, seed)
    if not part_dir.exists():
        return False
    for i in range(num_clients):
        if not (part_dir / f"client_{i}.pt").exists():
            return False
    if not (part_dir / "partition_info.json").exists():
        return False
    return True


def _load_raw_dataset(dataset_name: str, data_dir: Path) -> Tuple:
    """
    Download and load the full training dataset.

    Args:
        dataset_name: "mnist" or "cifar10".
        data_dir: Directory for raw downloads.

    Returns:
        Tuple of (data_tensor, labels_tensor).
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = torchvision.datasets.MNIST(
            root=str(data_dir), train=True, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'mnist' or 'cifar10'.")

    # Extract all data at once for partitioning
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, labels = next(iter(loader))
    return data, labels


def _load_test_dataset(dataset_name: str, data_dir: Path):
    """
    Load the test dataset for centralized evaluation.

    Args:
        dataset_name: "mnist" or "cifar10".
        data_dir: Directory for raw downloads.

    Returns:
        torch.utils.data.Dataset
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return torchvision.datasets.MNIST(
            root=str(data_dir), train=False, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        return torchvision.datasets.CIFAR10(
            root=str(data_dir), train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dirichlet_partition(
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
    data_dir: Path = None,
    partition_dir: Path = None,
    logger=None,
    force: bool = False,
) -> Path:
    """
    Partition a dataset using Dirichlet distribution and save to disk.

    Each client gets a subset of training data whose label distribution
    follows Dir(alpha). Lower alpha = more skewed (non-IID).

    Args:
        dataset_name: "mnist" or "cifar10".
        num_clients: Number of clients to partition data across.
        alpha: Dirichlet concentration. 0.1=highly skewed, 1.0=mild.
        seed: Random seed for reproducibility.
        data_dir: Where to download raw datasets.
        partition_dir: Where to save client .pt files.
        logger: Optional logger for status messages.
        force: If True, regenerate even if files exist.

    Returns:
        Path to the partition directory.

    Raises:
        ValueError: If dataset_name is not supported.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    if partition_dir is None:
        partition_dir = PARTITION_DIR

    out_dir = _get_partition_dir(dataset_name, alpha, seed)

    # Skip if already generated
    if not force and check_partition_exists(dataset_name, alpha, seed, num_clients):
        if logger:
            logger.info(
                f"Partition exists: {dataset_name} alpha={alpha} seed={seed} — skipping",
                extra={"experiment_id": "PARTITIONER", "component": "DATA"}
            )
        return out_dir

    if logger:
        logger.info(
            f"Generating partition: {dataset_name} | alpha={alpha} | seed={seed} | "
            f"clients={num_clients}",
            extra={"experiment_id": "PARTITIONER", "component": "DATA"}
        )

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    data, labels = _load_raw_dataset(dataset_name, Path(data_dir))
    labels_np = labels.numpy()
    num_classes = len(np.unique(labels_np))
    n_total = len(labels_np)

    # Build index list per class
    class_indices: Dict[int, List[int]] = {
        c: np.where(labels_np == c)[0].tolist()
        for c in range(num_classes)
    }

    # Dirichlet allocation
    # proportions[c] is a (num_clients,) array summing to 1
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = class_indices[c]
        np.random.shuffle(idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to actual counts
        counts = (proportions * len(idx)).astype(int)
        # Distribute remainder to avoid losing samples
        remainder = len(idx) - counts.sum()
        for i in range(remainder):
            counts[i % num_clients] += 1
        # Slice indices per client
        start = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idx[start:start + cnt])
            start += cnt

    # Save per-client files
    out_dir.mkdir(parents=True, exist_ok=True)
    all_info = []

    for i in range(num_clients):
        idx = client_indices[i]
        if len(idx) == 0:
            if logger:
                logger.warning(
                    f"Client {i} got 0 samples! Consider higher alpha or fewer clients.",
                    extra={"experiment_id": "PARTITIONER", "component": "DATA"}
                )

        client_data = data[idx]
        client_labels = labels[idx]

        # Save
        torch.save({"data": client_data, "labels": client_labels},
                   out_dir / f"client_{i}.pt")

        # Compute metadata
        label_counts = {}
        for lbl in client_labels.numpy():
            label_counts[int(lbl)] = label_counts.get(int(lbl), 0) + 1

        total = len(idx)
        dominant_class = max(label_counts, key=label_counts.get) if label_counts else -1
        dominant_pct = (label_counts.get(dominant_class, 0) / total * 100) if total > 0 else 0.0

        info = {
            "client_id": i,
            "alpha": alpha,
            "seed": seed,
            "total_samples": total,
            "class_distribution": {str(k): v for k, v in sorted(label_counts.items())},
            "dominant_class": dominant_class,
            "dominant_class_pct": round(dominant_pct, 1),
            "missing_classes": [
                c for c in range(num_classes) if c not in label_counts
            ],
        }
        all_info.append(info)

        # Terminal trace (ASCII table)
        trace_partition(i, label_counts, total, alpha, logger)

        # Always print a concise summary line
        missing_str = str(info["missing_classes"]) if info["missing_classes"] else "none"
        print(
            f"  {dataset_name.upper()} | alpha={alpha} | seed={seed} | "
            f"Client {i}: {total} samples | "
            f"Dominant: class {dominant_class} ({dominant_pct:.1f}%) | "
            f"Missing classes: {missing_str}"
        )

    # Save combined partition_info.json
    with open(out_dir / "partition_info.json", "w") as f:
        json.dump(all_info, f, indent=2)

    if logger:
        logger.info(
            f"Partition saved to {out_dir}",
            extra={"experiment_id": "PARTITIONER", "component": "DATA"}
        )

    return out_dir


def load_partition_info(dataset_name: str, alpha: float, seed: int) -> List[Dict]:
    """
    Load saved partition metadata from disk.

    Args:
        dataset_name: "mnist" or "cifar10".
        alpha: Dirichlet alpha.
        seed: Random seed.

    Returns:
        List of per-client info dicts.

    Raises:
        FileNotFoundError: If partition has not been generated yet.
    """
    part_dir = _get_partition_dir(dataset_name, alpha, seed)
    info_path = part_dir / "partition_info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Partition info not found: {info_path}. "
            f"Run prepare_data.py first."
        )
    with open(info_path) as f:
        return json.load(f)
