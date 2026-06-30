"""
Pre-generate all dataset partitions for the 54-experiment grid.

Run this once before starting any experiments. It downloads MNIST and
CIFAR-10 (if not already cached) and creates Dirichlet-partitioned
.pt files for every (dataset, alpha, seed, client) combination.

Total files generated:
  2 datasets × 3 alpha × 3 seeds × 10 clients = 180 client .pt files
  + 18 partition_info.json files

Usage:
    python experiments/prepare_data.py
    python experiments/prepare_data.py --datasets mnist --alphas 0.1 0.5

Author: FL Experiment System
Date: 2026
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.partitioner import create_dirichlet_partition, check_partition_exists
from src.utils.logger import get_logger


DATASETS = ["mnist", "cifar10"]
ALPHAS = [0.1, 0.5, 1.0]
SEEDS = [42, 123, 456]
NUM_CLIENTS = 10


def prepare_all_partitions(
    datasets: list,
    alphas: list,
    seeds: list,
    num_clients: int,
    force: bool = False,
    logger=None,
):
    """
    Generate and save all required dataset partitions.

    Args:
        datasets: List of dataset names to partition.
        alphas: List of Dirichlet alpha values.
        seeds: List of random seeds.
        num_clients: Number of clients per partition.
        force: If True, regenerate even if files already exist.
        logger: Optional logger.
    """
    total = len(datasets) * len(alphas) * len(seeds)
    done = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    print("\n" + "=" * 70)
    print("  FL EXPERIMENT — DATA PREPARATION")
    print(f"  Datasets: {datasets} | Alphas: {alphas} | Seeds: {seeds}")
    print(f"  Total partitions to generate: {total}")
    print("=" * 70 + "\n")

    for dataset in datasets:
        for alpha in alphas:
            for seed in seeds:
                label = f"{dataset} | alpha={alpha} | seed={seed}"
                idx = done + skipped + failed + 1

                # Check if already done
                if not force and check_partition_exists(dataset, alpha, seed, num_clients):
                    print(f"  [{idx:02d}/{total}] {label:<35} SKIP (exists)")
                    skipped += 1
                    continue

                print(f"  [{idx:02d}/{total}] {label:<35} GENERATING...")
                t0 = time.time()

                try:
                    create_dirichlet_partition(
                        dataset_name=dataset,
                        num_clients=num_clients,
                        alpha=alpha,
                        seed=seed,
                        logger=logger,
                        force=force,
                    )
                    elapsed = time.time() - t0
                    print(f"  [{idx:02d}/{total}] {label:<35} DONE ({elapsed:.1f}s)")
                    done += 1

                except Exception as e:
                    print(f"  [{idx:02d}/{total}] {label:<35} FAILED: {e}")
                    if logger:
                        logger.error(f"Partition failed: {label} | {e}",
                                     extra={"experiment_id": "PREPARE_DATA",
                                            "component": "PARTITIONER"})
                    failed += 1

    total_elapsed = time.time() - t_start
    m, s = divmod(int(total_elapsed), 60)

    print("\n" + "=" * 70)
    print("  PARTITION GENERATION COMPLETE")
    print(f"  Generated : {done}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print(f"  Total time: {m}m {s}s")
    print("=" * 70 + "\n")

    if failed > 0:
        print(f"  WARNING: {failed} partition(s) failed. Check logs above.")
        return False
    return True


def verify_partitions(datasets: list, alphas: list, seeds: list, num_clients: int):
    """
    Verify all expected partition files exist and print a summary.

    Args:
        datasets, alphas, seeds, num_clients: Same as prepare_all_partitions.
    """
    from src.data.partitioner import load_partition_info

    print("\n" + "=" * 70)
    print("  PARTITION VERIFICATION")
    print("=" * 70)

    all_ok = True
    for dataset in datasets:
        for alpha in alphas:
            for seed in seeds:
                exists = check_partition_exists(dataset, alpha, seed, num_clients)
                status = "OK" if exists else "MISSING"
                if not exists:
                    all_ok = False
                    print(f"  {dataset} | alpha={alpha} | seed={seed} — {status} ⚠")
                    continue

                # Show per-client summary
                try:
                    infos = load_partition_info(dataset, alpha, seed)
                    for info in infos:
                        cid = info["client_id"]
                        total = info["total_samples"]
                        dom_cls = info["dominant_class"]
                        dom_pct = info["dominant_class_pct"]
                        missing = info.get("missing_classes", [])
                        missing_str = str(missing) if missing else "none"
                        print(
                            f"  {dataset.upper()} | alpha={alpha} | seed={seed} | "
                            f"Client {cid}: {total} samples | "
                            f"Dominant: class {dom_cls} ({dom_pct:.1f}%) | "
                            f"Missing classes: {missing_str}"
                        )
                except Exception as e:
                    print(f"  ERROR reading info: {e}")
                    all_ok = False

    print("=" * 70)
    if all_ok:
        print("  All partitions verified OK.\n")
    else:
        print("  Some partitions are missing or corrupt.\n")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate all FL experiment data partitions."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        choices=["mnist", "cifar10"],
        help="Datasets to partition (default: all)"
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=ALPHAS,
        help="Dirichlet alpha values (default: 0.1 0.5 1.0)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Random seeds (default: 42 123 456)"
    )
    parser.add_argument(
        "--num_clients", type=int, default=NUM_CLIENTS,
        help="Number of clients per partition (default: 10)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate partitions even if they already exist"
    )
    parser.add_argument(
        "--verify_only", action="store_true",
        help="Only verify existing partitions, don't generate"
    )

    args = parser.parse_args()
    logger = get_logger("prepare_data")

    if args.verify_only:
        ok = verify_partitions(args.datasets, args.alphas, args.seeds, args.num_clients)
        sys.exit(0 if ok else 1)

    success = prepare_all_partitions(
        datasets=args.datasets,
        alphas=args.alphas,
        seeds=args.seeds,
        num_clients=args.num_clients,
        force=args.force,
        logger=logger,
    )

    # Always verify at the end
    verify_partitions(args.datasets, args.alphas, args.seeds, args.num_clients)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
