"""
Metrics evaluator for federated learning experiments.

Implements all 7 evaluation metrics across 3 groups:
  Group A — Global Performance : accuracy, rounds-to-target
  Group B — Fairness           : accuracy variance, Gini, participation fairness
  Group C — Trade-off analysis : Pareto data builder, two-way ANOVA

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─── Group A: Global Performance ────────────────────────────────────────────

def compute_global_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute accuracy of the global model on the centralized test set.

    Args:
        model: The global model (SimpleCNN or CIFARCNN).
        test_loader: DataLoader over the full test set (not partitioned).
        device: Torch device.

    Returns:
        Accuracy as a percentage (0–100).

    Example:
        >>> acc = compute_global_accuracy(model, test_loader, device)
        >>> acc
        72.34
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_labels).sum().item()
            total_samples += len(batch_labels)

    return (total_correct / total_samples * 100) if total_samples > 0 else 0.0


def compute_rounds_to_target(
    accuracy_history: List[float],
    target: float,
) -> Optional[int]:
    """
    Find the first round where global accuracy meets or exceeds the target.

    Args:
        accuracy_history: List of global accuracy values (% scale, 0-100)
                          indexed by round (index 0 = round 1).
        target: Target accuracy in % (e.g. 85.0 for MNIST, 70.0 for CIFAR-10).

    Returns:
        Round number (1-indexed) when target was first reached,
        or None if target was never reached within the given history.

    Example:
        >>> history = [40.0, 55.0, 70.0, 88.0, 92.0]
        >>> compute_rounds_to_target(history, target=85.0)
        4
    """
    for round_idx, acc in enumerate(accuracy_history):
        if acc >= target:
            return round_idx + 1  # 1-indexed
    return None


# ─── Group B: Per-Client Fairness ───────────────────────────────────────────

def compute_accuracy_variance(per_client_accuracies: List[float]) -> float:
    """
    Compute standard deviation of per-client model accuracies.

    Lower std = more uniform performance across clients.

    Args:
        per_client_accuracies: List of accuracy values (0–1 or 0–100 scale,
                               consistent within a run).

    Returns:
        Standard deviation (same scale as input).

    Raises:
        ValueError: If the list is empty.

    Example:
        >>> compute_accuracy_variance([0.72, 0.68, 0.75, 0.65, 0.80])
        0.0535...
    """
    if not per_client_accuracies:
        raise ValueError("per_client_accuracies cannot be empty.")
    return float(np.std(per_client_accuracies))


def compute_gini_coefficient(per_client_accuracies: List[float]) -> float:
    """
    Compute Gini coefficient of per-client accuracies.

    Standard formula: measures inequality of accuracy distribution.
    0 = all clients have identical accuracy (perfectly fair).
    1 = one client has all the accuracy (perfectly unfair).

    Args:
        per_client_accuracies: List of non-negative accuracy values.

    Returns:
        Gini coefficient in [0, 1].

    Raises:
        ValueError: If list is empty or all values are zero.

    Example:
        >>> compute_gini_coefficient([0.8, 0.8, 0.8, 0.8])
        0.0
        >>> compute_gini_coefficient([0.0, 0.0, 0.0, 1.0])
        0.75
    """
    if not per_client_accuracies:
        raise ValueError("per_client_accuracies cannot be empty.")

    values = np.array(per_client_accuracies, dtype=float)

    if values.sum() == 0:
        return 0.0

    values = np.sort(values)
    n = len(values)
    indices = np.arange(1, n + 1)
    return float((2 * np.sum(indices * values) - (n + 1) * values.sum())
                 / (n * values.sum()))


def compute_participation_fairness(participation_counts: Dict[str, int]) -> float:
    """
    Compute standard deviation of client participation rates.

    Lower std = more equal participation across clients.

    Args:
        participation_counts: Dict mapping client_id → number of rounds selected.

    Returns:
        Standard deviation of participation counts.

    Raises:
        ValueError: If dict is empty.

    Example:
        >>> compute_participation_fairness({"0": 5, "1": 5, "2": 5})
        0.0
        >>> compute_participation_fairness({"0": 10, "1": 1, "2": 1})
        4.2...
    """
    if not participation_counts:
        raise ValueError("participation_counts cannot be empty.")
    values = np.array(list(participation_counts.values()), dtype=float)
    return float(np.std(values))


# ─── Group C: Trade-off Analysis ────────────────────────────────────────────

def build_pareto_data(results_dict: Dict) -> "pd.DataFrame":
    """
    Build a DataFrame suitable for Pareto front analysis.

    Each entry in results_dict corresponds to one experiment run.
    The Pareto front identifies strategy-alpha combinations that are
    not dominated on both accuracy AND fairness simultaneously.

    Args:
        results_dict: Dict mapping experiment_id → final_metrics dict.
                      Each metrics dict must contain at minimum:
                      "strategy", "alpha", "global_accuracy", "gini_coefficient".

    Returns:
        pd.DataFrame with columns:
            experiment_id, strategy, alpha, accuracy, gini

    Raises:
        ImportError: If pandas is not installed.

    Example:
        >>> df = build_pareto_data(results)
        >>> df.columns.tolist()
        ['experiment_id', 'strategy', 'alpha', 'accuracy', 'gini']
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for build_pareto_data(). pip install pandas")

    rows = []
    for exp_id, metrics in results_dict.items():
        rows.append({
            "experiment_id": exp_id,
            "strategy": metrics.get("strategy", "unknown"),
            "alpha": metrics.get("alpha", -1.0),
            "accuracy": metrics.get("global_accuracy", 0.0),
            "gini": metrics.get("gini_coefficient", 0.0),
        })

    df = pd.DataFrame(rows)

    # Mark Pareto-optimal points (higher accuracy AND lower gini = better)
    if not df.empty:
        pareto_flags = []
        for i, row in df.iterrows():
            dominated = any(
                (other["accuracy"] >= row["accuracy"] and other["gini"] <= row["gini"] and
                 (other["accuracy"] > row["accuracy"] or other["gini"] < row["gini"]))
                for j, other in df.iterrows() if j != i
            )
            pareto_flags.append(not dominated)
        df["pareto_optimal"] = pareto_flags

    return df


def run_two_way_anova(
    results_df: "pd.DataFrame",
    dependent_var: str,
) -> Dict:
    """
    Run a two-way ANOVA: strategy × alpha → dependent_var.

    Tests whether strategy type, alpha level, or their interaction
    significantly affects the outcome metric.

    Args:
        results_df: DataFrame with columns "strategy", "alpha", and the
                    dependent variable column.
        dependent_var: Name of the outcome column (e.g. "accuracy", "gini").

    Returns:
        Dict with keys:
            F_strategy, p_strategy: Main effect of strategy.
            F_alpha, p_alpha: Main effect of alpha.
            F_interaction, p_interaction: Interaction effect.
            significant_strategy (bool): p < 0.05.
            significant_alpha (bool): p < 0.05.
            significant_interaction (bool): p < 0.05.

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If required columns are missing.

    Example:
        >>> result = run_two_way_anova(df, "accuracy")
        >>> result["F_strategy"]
        12.43
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for ANOVA. pip install scipy")
    if not HAS_PANDAS:
        raise ImportError("pandas is required for run_two_way_anova().")

    required_cols = {"strategy", "alpha", dependent_var}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(f"Missing columns in results_df: {missing}")

    # Group by strategy and alpha for one-way ANOVAs
    # Full two-way ANOVA via OLS if pingouin/statsmodels available,
    # otherwise fall back to individual one-way ANOVAs
    try:
        import pingouin as pg
        aov = pg.anova(
            data=results_df,
            dv=dependent_var,
            between=["strategy", "alpha"],
            detailed=True,
        )
        def _extract(source):
            row = aov[aov["Source"] == source]
            if row.empty:
                return float("nan"), float("nan")
            return float(row["F"].values[0]), float(row["p-unc"].values[0])

        F_strat, p_strat = _extract("strategy")
        F_alpha, p_alpha = _extract("alpha")
        F_inter, p_inter = _extract("strategy * alpha")

    except ImportError:
        # Fallback: separate one-way ANOVAs (approximate)
        groups_strategy = [
            results_df[results_df["strategy"] == s][dependent_var].values
            for s in results_df["strategy"].unique()
        ]
        groups_alpha = [
            results_df[results_df["alpha"] == a][dependent_var].values
            for a in results_df["alpha"].unique()
        ]
        F_strat, p_strat = scipy_stats.f_oneway(*groups_strategy)
        F_alpha, p_alpha = scipy_stats.f_oneway(*groups_alpha)
        # Interaction: not computable with one-way fallback
        F_inter, p_inter = float("nan"), float("nan")

    return {
        "F_strategy": round(float(F_strat), 4),
        "p_strategy": round(float(p_strat), 4),
        "F_alpha": round(float(F_alpha), 4),
        "p_alpha": round(float(p_alpha), 4),
        "F_interaction": round(float(F_inter), 4),
        "p_interaction": round(float(p_inter), 4),
        "significant_strategy": float(p_strat) < 0.05,
        "significant_alpha": float(p_alpha) < 0.05,
        "significant_interaction": float(p_inter) < 0.05,
        "dependent_var": dependent_var,
    }


# ─── Combined metric computation ────────────────────────────────────────────

def compute_all_metrics(
    round_results: List[Dict],
    participation_counts: Dict[str, int],
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    dataset_name: str,
) -> Dict:
    """
    Compute all 7 metrics for one completed experiment.

    Args:
        round_results: List of per-round result dicts. Each must have:
                       "global_accuracy" (float, % scale) and
                       "per_client_accuracies" (List[float], 0–1 scale).
        participation_counts: Dict[client_id → selection count] at end of run.
        test_loader: Centralized test DataLoader.
        model: Final global model (parameters already loaded).
        device: Torch device.
        dataset_name: "mnist" or "cifar10" (determines target accuracy).

    Returns:
        Dict with all 7 metrics:
            A1_global_accuracy      : Final round global accuracy (%).
            A2_rounds_to_target     : Round when target was first reached (int or None).
            B1_accuracy_variance    : Std dev of per-client accuracies (final round).
            B2_gini_coefficient     : Gini of per-client accuracies (final round).
            B3_participation_fairness: Std dev of participation counts.
            accuracy_history        : Full list of per-round global accuracy.
            per_client_final        : Per-client accuracies from final round.
    """
    targets = {"mnist": 85.0, "cifar10": 70.0}
    target_acc = targets.get(dataset_name, 80.0)

    # A1: Global accuracy on centralized test set
    a1 = compute_global_accuracy(model, test_loader, device)

    # A2: Rounds to target
    accuracy_history = [r["global_accuracy"] for r in round_results]
    a2 = compute_rounds_to_target(accuracy_history, target=target_acc)

    # B1, B2: Use per-client accuracies from the final round
    final_round = round_results[-1] if round_results else {}
    per_client_accs = final_round.get("per_client_accuracies", [])

    if per_client_accs:
        b1 = compute_accuracy_variance(per_client_accs)
        b2 = compute_gini_coefficient(per_client_accs)
    else:
        b1 = 0.0
        b2 = 0.0

    # B3: Participation fairness
    b3 = compute_participation_fairness(participation_counts) if participation_counts else 0.0

    return {
        "A1_global_accuracy": round(a1, 4),
        "A2_rounds_to_target": a2,
        "B1_accuracy_variance": round(b1, 6),
        "B2_gini_coefficient": round(b2, 6),
        "B3_participation_fairness": round(b3, 6),
        "accuracy_history": accuracy_history,
        "per_client_final": per_client_accs,
        "target_accuracy": target_acc,
        "target_reached": a2 is not None,
    }
