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
import ray
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
from src.strategies.performance_strategy import PerformanceBasedStrategy
from src.strategies.fairness_strategy import FairnessAwareStrategy
from src.data.partitioner import load_partition_info
from src.metrics.evaluator import compute_all_metrics, compute_global_accuracy
from src.system.device_model import DeviceModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
# Runs with simulated device heterogeneity go to a separate tree so they
# never mix with (or get skipped because of) the homogeneous results/ grid.
RESULTS_SYSTEM_DIR = PROJECT_ROOT / "results_system"

# Oort pacer window W for our 20-round runs. The paper's W=20 assumes
# hundreds of rounds; the pacer needs 2W rounds before it can fire.
OORT_PACER_WINDOW = 5


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
    client_num_samples: Dict[str, int] = None,
    eval_fn_for_fairness=None,
    device_model: Optional[DeviceModel] = None,
) -> FedAvg:
    """
    Instantiate and return the requested Flower strategy.

    Args:
        strategy_name: "random", "performance", or "fairness".
        num_clients: Total number of clients.
        clients_per_round: Number to select per round.
        seed: Seed for strategy internals.
        logger: Logger instance.
        client_num_samples: Dict mapping client_id → num_samples (for Oort
            statistical utility U(i) = |Bi| * loss_rms).
        eval_fn_for_fairness: Callable(ndarrays) -> accuracy, required for
            "fairness" (FairFedCS's Shapley Value contribution assessment).
            Must evaluate on a scratch model separate from the main global
            model (see make_shapley_eval_fn()).
        device_model: Optional simulated device heterogeneity. Only Oort
            uses it (system utility + pacer + speed-based exploration);
            random and FairFedCS selection are speed-agnostic by design.

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
        strategy = PerformanceBasedStrategy(
            client_num_samples=client_num_samples or {},
            clients_per_round=clients_per_round,
            seed=seed,
            logger=logger,
            device_model=device_model,
            pacer_window=OORT_PACER_WINDOW,
            **common_kwargs,
        )

    elif strategy_name == "fairness":
        if eval_fn_for_fairness is None:
            raise ValueError(
                "eval_fn_for_fairness is required to build the 'fairness' "
                "(FairFedCS) strategy — see make_shapley_eval_fn()."
            )
        strategy = FairnessAwareStrategy(
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            sigma=0.6,
            eval_fn=eval_fn_for_fairness,
            seed=seed,
            logger=logger,
            **common_kwargs,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                         f"Choose from: random, performance, fairness")

    return strategy


def make_shapley_eval_fn(dataset_name: str, test_loader, device):
    """
    Build an eval_fn(ndarrays) -> accuracy closure for FairFedCS's Shapley
    Value computation, backed by a dedicated scratch model instance.

    A separate model instance is essential: FairFedCS evaluates many
    candidate parameter subsets per round (2^m for a coalition of size m),
    and these evaluations must never overwrite the actual round-committed
    global model used for the experiment's real accuracy tracking.

    Args:
        dataset_name: "mnist" or "cifar10".
        test_loader: Centralized test DataLoader.
        device: Torch device.

    Returns:
        Callable[[NDArrays], float] returning accuracy on a 0-100 scale.
    """
    scratch_model = build_model(dataset_name).to(device)

    def eval_fn(ndarrays):
        scratch_model.set_parameters(ndarrays)
        return compute_global_accuracy(scratch_model, test_loader, device)

    return eval_fn


# ─── Aggregation callbacks ───────────────────────────────────────────────────

def make_callbacks(model, test_loader, device, strategy, logger, experiment_id):
    """
    Build the aggregation/evaluation callbacks Flower calls each round.

    Note: Flower's generic `fit_metrics_aggregation_fn` and
    `evaluate_metrics_aggregation_fn` callbacks only receive
    (num_examples, metrics) tuples — they carry NO client identity. Any
    per-client bookkeeping that needs real client identity (e.g. Oort's
    loss_rms statistical utility, FairFedCS's Shapley reputation update)
    is therefore handled inside the strategy's own aggregate_fit()
    override (see performance_strategy.py / fairness_strategy.py), which
    receives the full (ClientProxy, FitRes) pairs. These callbacks only
    compute order-independent aggregate statistics (means, std, Gini),
    which don't require client identity.
    """
    round_data: Dict[int, Dict] = {}
    participation_log: List[Dict] = []
    _current_round = [0]

    def fit_metrics_aggregation_fn(metrics_list):
        if hasattr(strategy, "participation_count"):
            participation_log.append(dict(strategy.participation_count))
        else:
            participation_log.append({})
        n_total = sum(n for n, _ in metrics_list)
        avg_loss = (sum(m.get("train_loss", 0) * n for n, m in metrics_list) / n_total
                    if n_total else 0.0)
        avg_acc = (sum(m.get("train_accuracy", 0) * n for n, m in metrics_list) / n_total
                   if n_total else 0.0)
        return {"avg_train_loss": avg_loss, "avg_train_accuracy": avg_acc}

    def evaluate_fn(server_round, parameters, config):
        from flwr.common import parameters_to_ndarrays, Parameters as FlwrParameters
        if isinstance(parameters, FlwrParameters):
            ndarrays = parameters_to_ndarrays(parameters)
        else:
            ndarrays = list(parameters)
        model.set_parameters(ndarrays)

        acc = compute_global_accuracy(model, test_loader, device)
        _current_round[0] = server_round
        round_data[server_round] = {
            "round": server_round,
            "global_accuracy": acc,
            "per_client_accuracies": [],
            "per_client_eval": [],
        }

        if logger:
            logger.info(
                f"Global Accuracy: {acc:.4f}%",
                extra={"experiment_id": experiment_id,
                       "component": f"ROUND_{server_round:02d}"}
            )
        return float(acc), {"global_accuracy": acc}

    def evaluate_metrics_aggregation_fn(metrics_list):
        r = _current_round[0]
        client_accs = [float(m.get("accuracy", 0.0)) for _, m in metrics_list]
        # No client identity available here (see docstring above) — kept as
        # an order-parallel list, not a (mis)labeled id -> value mapping.
        per_client_eval = [
            {"accuracy": float(m.get("accuracy", 0.0)), "loss": float(m.get("loss", 0.0))}
            for _, m in metrics_list
        ]

        if r in round_data:
            round_data[r]["per_client_accuracies"] = client_accs
            round_data[r]["per_client_eval"] = per_client_eval
        n_total = sum(n for n, _ in metrics_list)
        avg_acc = (sum(m.get("accuracy", 0.0) * n for n, m in metrics_list) / n_total
                   if n_total else 0.0)
        return {"avg_eval_accuracy": avg_acc}

    return (round_data, participation_log,
            fit_metrics_aggregation_fn, evaluate_fn, evaluate_metrics_aggregation_fn)


if ray.is_initialized():
    ray.shutdown()

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
    system_hetero: bool = False,
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
        system_hetero: Simulate device heterogeneity (per-client compute
            speed / bandwidth). Enables Oort's system utility and records
            simulated wall-clock time for every strategy. Output dir
            defaults to results_system/.

    Returns:
        Dict of final metrics.
    """
    # ── Setup ─────────────────────────────────────────────────────────────
    experiment_id = f"{strategy_name}_{dataset_name}_a{alpha}_s{seed}"
    if output_dir is None:
        output_dir = RESULTS_SYSTEM_DIR if system_hetero else RESULTS_DIR
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
        "system_hetero": system_hetero,
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

    # ── Load client metadata (for Oort utility computation) ────────────────
    client_num_samples: Dict[str, int] = {}
    try:
        partition_info = load_partition_info(dataset_name, alpha, seed)
        for info in partition_info:
            client_num_samples[str(info["client_id"])] = info["total_samples"]
    except FileNotFoundError:
        if logger:
            logger.warning("partition_info.json not found, using uniform sample counts")
        # Fallback: uniform distribution
        total_samples = 60000 if dataset_name == "mnist" else 50000
        per_client = total_samples // num_clients
        for i in range(num_clients):
            client_num_samples[str(i)] = per_client

    # ── Simulated device heterogeneity ─────────────────────────────────────
    device_model = None
    if system_hetero:
        model_size_mb = sum(p.numel() * p.element_size()
                            for p in global_model.parameters()) / 1e6
        device_model = DeviceModel(
            client_num_samples=client_num_samples,
            local_epochs=local_epochs,
            model_size_mb=model_size_mb,
            seed=seed,
        )
        with open(exp_dir / "device_profiles.json", "w") as f:
            json.dump(device_model.to_dict(), f, indent=2)

    # ── Strategy ──────────────────────────────────────────────────────────
    eval_fn_for_fairness = None
    if strategy_name == "fairness":
        eval_fn_for_fairness = make_shapley_eval_fn(dataset_name, test_loader, device)

    strategy = build_strategy(
        strategy_name, num_clients, clients_per_round, seed, logger,
        client_num_samples=client_num_samples,
        eval_fn_for_fairness=eval_fn_for_fairness,
        device_model=device_model,
    )
    if hasattr(strategy, "set_experiment_id"):
        strategy.set_experiment_id(experiment_id)

    (round_data, participation_log,
     fit_metrics_fn, evaluate_fn, evaluate_metrics_fn) = make_callbacks(
        global_model, test_loader, device, strategy, logger, experiment_id
    )
    strategy.evaluate_fn = evaluate_fn
    strategy.fit_metrics_aggregation_fn = fit_metrics_fn
    strategy.evaluate_metrics_aggregation_fn = evaluate_metrics_fn

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

    # ── Metrics ───────────────────────────────────────────────────────────
    round_results = [round_data[r] for r in sorted(round_data.keys())]

    final_participation = dict(strategy.participation_count)
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
        "system_hetero": system_hetero,
    })
    if device_model is not None:
        final_metrics.update(compute_simulated_time_metrics(
            participation_log, device_model,
            final_metrics["accuracy_history"], final_metrics["target_accuracy"]))

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


def compute_simulated_time_metrics(
    participation_log: List[Dict],
    device_model: DeviceModel,
    accuracy_history: List[float],
    target_accuracy: float,
) -> Dict:
    """
    Simulated wall-clock metrics under device heterogeneity.

    The clients selected in round r are recovered from the cumulative
    participation counts (round r minus round r-1). A synchronous round
    lasts as long as its slowest selected client.

    Returns:
        Dict with per-round and cumulative simulated seconds, total
        simulated time, and A2_time_to_target_seconds (simulated seconds
        until global accuracy first reaches the target; None if never).
    """
    round_durations: List[float] = []
    prev: Dict[str, int] = {}
    for counts in participation_log:
        selected = [cid for cid, c in counts.items() if c > prev.get(cid, 0)]
        round_durations.append(device_model.round_duration(selected))
        prev = counts

    cumulative = [float(t) for t in np.cumsum(round_durations)]
    time_to_target = None
    for r, acc in enumerate(accuracy_history):
        if acc >= target_accuracy and r < len(cumulative):
            time_to_target = cumulative[r]
            break

    return {
        "sim_round_durations": [float(d) for d in round_durations],
        "sim_cumulative_time": cumulative,
        "sim_total_time_seconds": cumulative[-1] if cumulative else 0.0,
        "A2_time_to_target_seconds": time_to_target,
    }


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
        *([f"  Sim. time-to-target     : {metrics.get('A2_time_to_target_seconds')}",
           f"  Sim. total time         : {metrics['sim_total_time_seconds']:.1f}s"]
          if "sim_total_time_seconds" in metrics else []),
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
    parser.add_argument("--system_hetero", action="store_true",
                        help="Simulate device heterogeneity (Oort system utility "
                             "+ simulated wall-clock). Results go to results_system/")
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
            system_hetero=args.system_hetero,
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
