"""
Single experiment runner for FL client selection comparison.

Runs one complete federated learning experiment with a specified
strategy, dataset, alpha, and seed. Saves all results and metrics
to the results/ directory.

Usage:
    python experiments/run_single.py --strategy random --dataset mnist \
        --alpha 0.1 --seed 42 --rounds 20

    # Smoke test (1 round, with trace):
    python experiments/run_single.py --strategy random --dataset mnist \
        --alpha 0.1 --seed 42 --rounds 1 --trace

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg

from src.utils import tracer
from src.utils.logger import get_logger, log_round_summary, log_experiment_config
from src.data.partitioner import create_dirichlet_partition, check_partition_exists
from src.data.loader import get_test_dataloader
from src.models.mnist_cnn import SimpleCNN
from src.models.cifar_cnn import CIFARCNN
from src.client.fl_client import make_client_fn
from src.strategies.random_strategy import RandomStrategy
from src.strategies.performance_strategy import PerformanceBasedStrategy, generate_client_latencies
from src.strategies.fairness_strategy import FairnessAwareStrategy
from src.metrics.evaluator import compute_all_metrics, compute_global_accuracy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


# ─── Helpers ────────────────────────────────────────────────────────────────

def set_all_seeds(seed: int):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(dataset_name: str) -> torch.nn.Module:
    """Instantiate the correct model for the dataset."""
    if dataset_name == "mnist":
        return SimpleCNN()
    elif dataset_name == "cifar10":
        return CIFARCNN()
    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_strategy(
    strategy_name: str,
    num_clients: int,
    clients_per_round: int,
    seed: int,
    logger,
) -> FedAvg:
    """
    Instantiate and return the requested Flower strategy.

    Args:
        strategy_name: "random", "performance", or "fairness".
        num_clients: Total number of clients.
        clients_per_round: Number to select per round.
        seed: Seed for strategy internals.
        logger: Logger instance.

    Returns:
        Configured FedAvg subclass.
    """
    fraction_fit = clients_per_round / num_clients
    common_kwargs = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=clients_per_round,
        min_available_clients=num_clients,
    )

    if strategy_name == "random":
        strategy = RandomStrategy(logger=logger, **common_kwargs)

    elif strategy_name == "performance":
        latencies = generate_client_latencies(
            num_clients=num_clients,
            seed=seed,
            distribution="exponential",
            scale=100.0,
        )
        strategy = PerformanceBasedStrategy(
            client_latencies=latencies,
            clients_per_round=clients_per_round,
            logger=logger,
            **common_kwargs,
        )

    elif strategy_name == "fairness":
        strategy = FairnessAwareStrategy(
            clients_per_round=clients_per_round,
            seed=seed,
            logger=logger,
            **common_kwargs,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                         f"Choose from: random, performance, fairness")

    return strategy


# ─── Aggregation callbacks ───────────────────────────────────────────────────

def make_evaluate_fn(model, test_loader, device, round_results, logger, experiment_id):
    """
    Return a server-side evaluate function for centralized evaluation.

    This is called by Flower after each round's aggregation with the
    updated global parameters.

    Returns:
        Callable used as FedAvg's evaluate_fn argument.
    """
    def evaluate_fn(server_round: int, parameters, config):
        # Load aggregated parameters into model
        model.set_parameters(parameters.tensors if hasattr(parameters, 'tensors')
                             else [p for p in parameters])
        acc = compute_global_accuracy(model, test_loader, device)

        # Store for later metric computation
        round_results.append({
            "round": server_round,
            "global_accuracy": acc,
            "per_client_accuracies": [],  # filled by fit_metrics_aggregation_fn
        })

        if logger:
            extra = {"experiment_id": experiment_id,
                     "component": f"ROUND_{server_round:02d}"}
            logger.info(f"Global Accuracy: {acc:.4f}%", extra=extra)

        return float(acc), {"global_accuracy": acc}

    return evaluate_fn


def make_fit_metrics_fn(round_results, participation_log, strategy, logger, experiment_id):
    """
    Return a function to aggregate fit metrics from all clients.

    Flower calls this with a list of (num_samples, metrics_dict) from
    all clients that participated in the round.
    """
    def fit_metrics_aggregation_fn(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
        # Extract per-client accuracies
        client_accs = []
        client_losses = []
        participation_this_round = []

        for num_samples, m in metrics_list:
            client_accs.append(m.get("train_accuracy", 0.0))
            client_losses.append(m.get("train_loss", 0.0))

        # Update the most recent round_results entry
        if round_results:
            round_results[-1]["per_client_accuracies"] = client_accs

        # Track participation (strategy knows who was selected)
        if hasattr(strategy, "participation_count"):
            participation_log.append(dict(strategy.participation_count))
        else:
            participation_log.append({})

        # Aggregated metrics
        n_total = sum(n for n, _ in metrics_list)
        avg_loss = sum(m.get("train_loss", 0) * n for n, m in metrics_list) / n_total if n_total else 0
        avg_acc = sum(m.get("train_accuracy", 0) * n for n, m in metrics_list) / n_total if n_total else 0

        return {"avg_train_loss": avg_loss, "avg_train_accuracy": avg_acc}

    return fit_metrics_aggregation_fn


# ─── Main runner ─────────────────────────────────────────────────────────────

def run_experiment(
    strategy_name: str,
    dataset_name: str,
    alpha: float,
    seed: int,
    num_rounds: int = 20,
    num_clients: int = 10,
    clients_per_round: int = 5,
    local_epochs: int = 3,
    learning_rate: float = 0.01,
    output_dir: Path = None,
    trace: bool = False,
    logger=None,
) -> Dict:
    """
    Run a single FL experiment end-to-end.

    Args:
        strategy_name: "random", "performance", or "fairness".
        dataset_name: "mnist" or "cifar10".
        alpha: Dirichlet concentration parameter.
        seed: Master random seed.
        num_rounds: Number of communication rounds.
        num_clients: Total number of simulated clients.
        clients_per_round: Clients selected per round.
        local_epochs: Local training epochs per round.
        learning_rate: Client SGD learning rate.
        output_dir: Where to save results.
        trace: Enable tensor/shape tracing.
        logger: Logger instance.

    Returns:
        Dict of final metrics.
    """
    # ── Setup ─────────────────────────────────────────────────────────────
    experiment_id = f"{strategy_name}_{dataset_name}_a{alpha}_s{seed}"
    if output_dir is None:
        output_dir = RESULTS_DIR
    exp_dir = Path(output_dir) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    tracer.set_trace_mode(trace)

    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "experiment_id": experiment_id,
        "strategy": strategy_name,
        "dataset": dataset_name,
        "alpha": alpha,
        "seed": seed,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "clients_per_round": clients_per_round,
        "local_epochs": local_epochs,
        "learning_rate": learning_rate,
        "device": str(device),
        "trace_mode": trace,
    }

    if logger:
        log_experiment_config(logger, config)

    # Save config immediately
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Data ──────────────────────────────────────────────────────────────
    # Ensure partitions exist
    if not check_partition_exists(dataset_name, alpha, seed, num_clients):
        if logger:
            logger.info(f"Partition missing, generating...",
                        extra={"experiment_id": experiment_id, "component": "DATA"})
        create_dirichlet_partition(
            dataset_name=dataset_name,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            logger=logger,
        )

    test_loader = get_test_dataloader(dataset_name)
    global_model = build_model(dataset_name).to(device)
    tracer.trace_model_params(global_model, logger)

    # ── Strategy ──────────────────────────────────────────────────────────
    round_results: List[Dict] = []
    participation_log: List[Dict] = []

    strategy = build_strategy(strategy_name, num_clients, clients_per_round, seed, logger)
    if hasattr(strategy, "set_experiment_id"):
        strategy.set_experiment_id(experiment_id)

    # Attach server-side evaluation
    strategy.evaluate_fn = make_evaluate_fn(
        global_model, test_loader, device, round_results, logger, experiment_id
    )
    strategy.fit_metrics_aggregation_fn = make_fit_metrics_fn(
        round_results, participation_log, strategy, logger, experiment_id
    )

    # ── Client factory ────────────────────────────────────────────────────
    client_fn = make_client_fn(
        dataset_name=dataset_name,
        alpha=alpha,
        seed=seed,
        device=device,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        logger=logger,
    )

    # ── Run simulation ────────────────────────────────────────────────────
    t_start = time.time()
    if logger:
        logger.info(f"Starting simulation: {num_rounds} rounds | "
                    f"{num_clients} clients | {clients_per_round}/round",
                    extra={"experiment_id": experiment_id, "component": "SIMULATION"})

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    t_total = time.time() - t_start

    # ── Metrics ───────────────────────────────────────────────────────────
    # Get final participation counts
    if hasattr(strategy, "participation_count"):
        final_participation = dict(strategy.participation_count)
    else:
        # Reconstruct from participation log
        final_participation = defaultdict(int)
        for log_entry in participation_log:
            for cid, cnt in log_entry.items():
                final_participation[cid] = cnt
        final_participation = dict(final_participation)

    # Ensure all clients are represented
    for i in range(num_clients):
        final_participation.setdefault(str(i), 0)

    final_metrics = compute_all_metrics(
        round_results=round_results,
        participation_counts=final_participation,
        test_loader=test_loader,
        model=global_model,
        device=device,
        dataset_name=dataset_name,
    )
    final_metrics.update({
        "experiment_id": experiment_id,
        "strategy": strategy_name,
        "dataset": dataset_name,
        "alpha": alpha,
        "seed": seed,
        "total_time_seconds": round(t_total, 2),
        "global_accuracy": final_metrics["A1_global_accuracy"],
        "gini_coefficient": final_metrics["B2_gini_coefficient"],
    })

    # ── Save results ──────────────────────────────────────────────────────
    with open(exp_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    with open(exp_dir / "metrics_per_round.json", "w") as f:
        json.dump(round_results, f, indent=2)

    with open(exp_dir / "participation_log.json", "w") as f:
        json.dump({
            "final_counts": final_participation,
            "per_round_counts": participation_log,
        }, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────
    _print_summary(experiment_id, final_metrics, t_total)

    return final_metrics


def _print_summary(experiment_id: str, metrics: Dict, elapsed: float):
    """Print a clean summary table at the end of an experiment."""
    m, s = divmod(int(elapsed), 60)
    lines = [
        "",
        "=" * 62,
        f"  EXPERIMENT COMPLETE: {experiment_id}",
        "=" * 62,
        f"  A1  Global Accuracy     : {metrics.get('A1_global_accuracy', 0):.4f}%",
        f"  A2  Rounds to Target    : {metrics.get('A2_rounds_to_target', 'N/A')}",
        f"  B1  Accuracy Variance   : {metrics.get('B1_accuracy_variance', 0):.6f}",
        f"  B2  Gini Coefficient    : {metrics.get('B2_gini_coefficient', 0):.6f}",
        f"  B3  Participation Fair. : {metrics.get('B3_participation_fairness', 0):.6f}",
        f"  Target reached          : {metrics.get('target_reached', False)}",
        f"  Total time              : {m}m {s}s",
        "=" * 62,
        "",
    ]
    print("\n".join(lines))


def smoke_test():
    """
    Quick smoke test: 1 round of random/mnist to verify the pipeline.

    Run with:
        python experiments/run_single.py --strategy random --dataset mnist
            --alpha 0.1 --seed 42 --rounds 1 --trace
    """
    print("\n[SMOKE TEST] Running 1 round to verify pipeline...\n")
    logger = get_logger("smoke_test")
    metrics = run_experiment(
        strategy_name="random",
        dataset_name="mnist",
        alpha=0.1,
        seed=42,
        num_rounds=1,
        num_clients=10,
        clients_per_round=5,
        local_epochs=1,
        learning_rate=0.01,
        trace=True,
        logger=logger,
    )
    print("[SMOKE TEST] PASSED" if metrics else "[SMOKE TEST] FAILED")
    return metrics


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run a single FL client-selection experiment."
    )
    parser.add_argument("--strategy", required=True,
                        choices=["random", "performance", "fairness"],
                        help="Client selection strategy")
    parser.add_argument("--dataset", required=True,
                        choices=["mnist", "cifar10"],
                        help="Dataset to use")
    parser.add_argument("--alpha", required=True, type=float,
                        help="Dirichlet alpha (0.1, 0.5, or 1.0)")
    parser.add_argument("--seed", required=True, type=int,
                        help="Random seed")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of communication rounds (default: 20)")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="Total number of clients (default: 10)")
    parser.add_argument("--clients_per_round", type=int, default=5,
                        help="Clients selected per round (default: 5)")
    parser.add_argument("--local_epochs", type=int, default=3,
                        help="Local training epochs per round (default: 3)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Results output directory (default: results/)")
    parser.add_argument("--trace", action="store_true",
                        help="Enable tensor/shape tracer (verbose)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 1-round smoke test instead of full experiment")

    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        sys.exit(0)

    # Validate alpha
    if args.alpha not in [0.1, 0.5, 1.0]:
        print(f"WARNING: alpha={args.alpha} is non-standard. "
              f"Standard values are 0.1, 0.5, 1.0.")

    experiment_id = f"{args.strategy}_{args.dataset}_a{args.alpha}_s{args.seed}"
    logger = get_logger(experiment_id)

    try:
        metrics = run_experiment(
            strategy_name=args.strategy,
            dataset_name=args.dataset,
            alpha=args.alpha,
            seed=args.seed,
            num_rounds=args.rounds,
            num_clients=args.num_clients,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            learning_rate=args.lr,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            trace=args.trace,
            logger=logger,
        )
        sys.exit(0)

    except Exception as e:
        logger.error(
            f"Experiment FAILED: {e}\n{traceback.format_exc()}",
            extra={"experiment_id": experiment_id, "component": "RUNNER"}
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
