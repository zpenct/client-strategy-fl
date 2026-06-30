"""
Batch runner for all 54 FL experiments.

Executes the full experiment grid (3 strategies × 3 alpha × 2 datasets
× 3 seeds = 54 runs) with incremental checkpointing so runs can be
safely interrupted and resumed.

Usage:
    # Run all 54 experiments
    python experiments/run_batch.py

    # Skip already-completed experiments (safe resume after crash/shutdown)
    python experiments/run_batch.py --skip_existing

    # Dry run: list all experiments without running
    python experiments/run_batch.py --dry_run

    # Resume from experiment #10 onward
    python experiments/run_batch.py --start_from 10 --skip_existing

    # Run only specific subset
    python experiments/run_batch.py --strategies random fairness --datasets mnist

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_single import run_experiment
from src.utils.logger import get_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


# ─── Experiment grid ─────────────────────────────────────────────────────────

STRATEGIES = ["random", "performance", "fairness"]
DATASETS = ["mnist", "cifar10"]
ALPHAS = [0.1, 0.5, 1.0]
SEEDS = [42, 123, 456]


@dataclass
class ExperimentSpec:
    """One row in the experiment grid."""
    strategy: str
    dataset: str
    alpha: float
    seed: int

    @property
    def experiment_id(self) -> str:
        return f"{self.strategy}_{self.dataset}_a{self.alpha}_s{self.seed}"

    @property
    def result_dir(self) -> Path:
        return RESULTS_DIR / self.experiment_id

    def is_complete(self) -> bool:
        """True if final_metrics.json exists in the result directory."""
        return (self.result_dir / "final_metrics.json").exists()


def build_experiment_grid(
    strategies: List[str],
    datasets: List[str],
    alphas: List[float],
    seeds: List[int],
) -> List[ExperimentSpec]:
    """Build the full ordered list of experiment specifications."""
    grid = []
    for strategy in strategies:
        for dataset in datasets:
            for alpha in alphas:
                for seed in seeds:
                    grid.append(ExperimentSpec(strategy, dataset, alpha, seed))
    return grid


# ─── Progress display ─────────────────────────────────────────────────────────

def _eta_str(elapsed_per_exp: List[float], remaining: int) -> str:
    """Estimate time remaining based on average experiment duration."""
    if not elapsed_per_exp:
        return "unknown"
    avg = sum(elapsed_per_exp) / len(elapsed_per_exp)
    eta_s = int(avg * remaining)
    h, rem = divmod(eta_s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _print_header(total: int, skip_existing: bool):
    print("\n" + "=" * 66)
    print("  FL EXPERIMENT BATCH RUNNER")
    print(f"  Total: {total} experiments | Skip existing: {skip_existing}")
    print("=" * 66)


def _print_final_summary(
    done: int, skipped: int, failed: int,
    failed_ids: List[str], total_elapsed: float
):
    m, s = divmod(int(total_elapsed), 60)
    h, m = divmod(m, 60)
    if h > 0:
        time_str = f"{h}h {m}m {s}s"
    else:
        time_str = f"{m}m {s}s"

    print("\n" + "=" * 66)
    print("  BATCH COMPLETE")
    print(f"  Done   : {done}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed : {failed}")
    print(f"  Total time: {time_str}")
    if failed_ids:
        print("\n  Failed experiments:")
        for fid in failed_ids:
            print(f"    - {fid}")
    print("=" * 66 + "\n")


def _print_results_table(grid: List[ExperimentSpec], statuses: List[str]):
    """Print a summary table of all experiments and their outcomes."""
    print("\n" + "=" * 80)
    print(f"  {'#':>3}  {'Experiment ID':<45}  {'Status':<10}  {'Time':>8}")
    print("  " + "-" * 76)

    for i, (spec, status) in enumerate(zip(grid, statuses), 1):
        # Try to read timing from final_metrics
        time_str = ""
        if spec.is_complete():
            try:
                import json
                with open(spec.result_dir / "final_metrics.json") as f:
                    m = json.load(f)
                secs = m.get("total_time_seconds", 0)
                time_str = _format_elapsed(secs)
            except Exception:
                pass

        status_display = {
            "DONE": "✓ DONE",
            "SKIP": "  SKIP",
            "FAILED": "✗ FAIL",
            "PENDING": "  ...",
        }.get(status, status)

        print(f"  {i:>3}  {spec.experiment_id:<45}  {status_display:<10}  {time_str:>8}")

    print("=" * 80 + "\n")


# ─── Main batch runner ────────────────────────────────────────────────────────

def run_batch(
    strategies: List[str],
    datasets: List[str],
    alphas: List[float],
    seeds: List[int],
    skip_existing: bool = True,
    dry_run: bool = False,
    start_from: int = 1,
    num_rounds: int = 20,
    num_clients: int = 10,
    clients_per_round: int = 5,
    local_epochs: int = 3,
    learning_rate: float = 0.01,
) -> dict:
    """
    Execute the batch of FL experiments.

    Args:
        strategies: List of strategy names to run.
        datasets: List of dataset names.
        alphas: List of alpha values.
        seeds: List of seeds.
        skip_existing: Skip experiments that already have results.
        dry_run: Print the list without executing.
        start_from: 1-indexed position to start from (for resuming).
        num_rounds: Communication rounds per experiment.
        num_clients: Number of simulated clients.
        clients_per_round: Clients selected per round.
        local_epochs: Local training epochs per round.
        learning_rate: Client learning rate.

    Returns:
        Dict with summary counts: done, skipped, failed.
    """
    grid = build_experiment_grid(strategies, datasets, alphas, seeds)
    total = len(grid)

    # Apply --start_from (convert to 0-indexed)
    if start_from > 1:
        grid = grid[start_from - 1:]
        print(f"  Resuming from experiment #{start_from} "
              f"({len(grid)} experiments remaining)")

    _print_header(total, skip_existing)

    if dry_run:
        print("\n  DRY RUN — experiment list:")
        for i, spec in enumerate(grid, start=start_from):
            marker = "EXISTS" if spec.is_complete() else "PENDING"
            print(f"  [{i:02d}/{total}] {spec.experiment_id:<45} [{marker}]")
        print(f"\n  Total: {len(grid)} experiments\n")
        return {"done": 0, "skipped": 0, "failed": 0}

    # ── Execution loop ─────────────────────────────────────────────────────
    done = 0
    skipped = 0
    failed = 0
    failed_ids: List[str] = []
    elapsed_per_exp: List[float] = []
    statuses: List[str] = ["PENDING"] * len(grid)

    batch_logger = get_logger("batch_runner")
    t_batch_start = time.time()

    for i, spec in enumerate(grid):
        global_idx = i + start_from
        prefix = f"  [{global_idx:02d}/{total}] {spec.experiment_id:<45}"

        # ── Skip check ────────────────────────────────────────────────────
        if skip_existing and spec.is_complete():
            print(f"{prefix} ... SKIP (already exists)")
            skipped += 1
            statuses[i] = "SKIP"
            continue

        # ── Run experiment ────────────────────────────────────────────────
        print(f"{prefix} ... RUNNING")
        exp_logger = get_logger(spec.experiment_id)
        t0 = time.time()

        try:
            run_experiment(
                strategy_name=spec.strategy,
                dataset_name=spec.dataset,
                alpha=spec.alpha,
                seed=spec.seed,
                num_rounds=num_rounds,
                num_clients=num_clients,
                clients_per_round=clients_per_round,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                trace=False,
                logger=exp_logger,
            )
            elapsed = time.time() - t0
            elapsed_per_exp.append(elapsed)

            remaining = len(grid) - (i + 1) - skipped
            eta = _eta_str(elapsed_per_exp, remaining)
            print(f"{prefix} ... DONE ({_format_elapsed(elapsed)}) | ETA: {eta}")

            batch_logger.info(
                f"[{global_idx}/{total}] DONE: {spec.experiment_id} "
                f"in {_format_elapsed(elapsed)}",
                extra={"experiment_id": "BATCH", "component": "RUNNER"}
            )
            done += 1
            statuses[i] = "DONE"

        except Exception as e:
            elapsed = time.time() - t0
            err_msg = f"{type(e).__name__}: {e}"
            print(f"{prefix} ... FAILED ({_format_elapsed(elapsed)}) | {err_msg}")

            batch_logger.error(
                f"[{global_idx}/{total}] FAILED: {spec.experiment_id}\n"
                f"{traceback.format_exc()}",
                extra={"experiment_id": "BATCH", "component": "RUNNER"}
            )
            failed += 1
            failed_ids.append(spec.experiment_id)
            statuses[i] = "FAILED"
            # Continue to next experiment — do NOT abort the batch

    total_elapsed = time.time() - t_batch_start

    # ── Final output ───────────────────────────────────────────────────────
    _print_results_table(grid, statuses)
    _print_final_summary(done, skipped, failed, failed_ids, total_elapsed)

    return {"done": done, "skipped": skipped, "failed": failed,
            "failed_ids": failed_ids}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all 54 FL experiments in the experiment grid."
    )
    parser.add_argument(
        "--strategies", nargs="+", default=STRATEGIES,
        choices=STRATEGIES,
        help=f"Strategies to run (default: all {STRATEGIES})"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        choices=DATASETS,
        help=f"Datasets to run (default: all {DATASETS})"
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=ALPHAS,
        help=f"Alpha values (default: {ALPHAS})"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help=f"Seeds (default: {SEEDS})"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip experiments that already have results saved"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print experiment list without running"
    )
    parser.add_argument(
        "--start_from", type=int, default=1,
        help="1-indexed experiment to start from (for resuming)"
    )
    parser.add_argument(
        "--rounds", type=int, default=20,
        help="Communication rounds per experiment (default: 20)"
    )
    parser.add_argument(
        "--num_clients", type=int, default=10,
        help="Number of simulated clients (default: 10)"
    )
    parser.add_argument(
        "--clients_per_round", type=int, default=5,
        help="Clients selected per round (default: 5)"
    )
    parser.add_argument(
        "--local_epochs", type=int, default=3,
        help="Local epochs per round (default: 3)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Client learning rate (default: 0.01)"
    )

    args = parser.parse_args()

    result = run_batch(
        strategies=args.strategies,
        datasets=args.datasets,
        alphas=args.alphas,
        seeds=args.seeds,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        start_from=args.start_from,
        num_rounds=args.rounds,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
    )

    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
