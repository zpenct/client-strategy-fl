"""
Cross-experiment analysis for the full FL client-selection experiment grid.

Aggregates every completed run under results/ into a single table, then
produces the two proposal-required cross-experiment analyses that a single
run's final_metrics.json cannot answer on its own:

    C1 — Pareto frontier (accuracy vs. per-client fairness trade-off)
    C2 — Two-way ANOVA (strategy x alpha significance on each metric)

Also emits a mean+-std summary table per (dataset, strategy, alpha), and a
short answer to the proposal's Rumusan Masalah 3 ("at which alpha does
fairness-aware selection become Pareto-optimal, and is this consistent
across MNIST and CIFAR-10?").

Usage:
    python experiments/analyze_results.py
    python experiments/analyze_results.py --results_dir results --out_dir results/analysis

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.metrics.evaluator import run_two_way_anova, HAS_SCIPY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEPENDENT_VARS = [
    "A1_global_accuracy",
    "B1_accuracy_variance",
    "B2_gini_coefficient",
    "B3_participation_fairness",
]


# ─── Loading ──────────────────────────────────────────────────────────────

def load_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Read every results/<experiment_id>/final_metrics.json into one DataFrame.

    Skips the `analysis/` and `_validation/` housekeeping subdirectories.

    Args:
        results_dir: Directory containing one subfolder per experiment.

    Returns:
        DataFrame with one row per completed experiment.
    """
    rows: List[Dict] = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in {"analysis", "_validation"}:
            continue
        metrics_path = exp_dir / "final_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append(metrics)

    if not rows:
        raise FileNotFoundError(
            f"No final_metrics.json files found under {results_dir}. "
            f"Run experiments first (experiments/run_batch.py)."
        )

    df = pd.DataFrame(rows)
    df["alpha"] = df["alpha"].astype(float)
    return df


# ─── Summary table (mean +- std per dataset x strategy x alpha) ───────────

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mean+-std across seeds for every (dataset, strategy, alpha).

    Returns:
        DataFrame with one row per (dataset, strategy, alpha) combination.
    """
    agg_cols = ["A1_global_accuracy", "A2_rounds_to_target"] + DEPENDENT_VARS[1:]
    grouped = df.groupby(["dataset", "strategy", "alpha"])

    summary_rows = []
    for (dataset, strategy, alpha), group in grouped:
        row = {"dataset": dataset, "strategy": strategy, "alpha": alpha, "n_seeds": len(group)}
        for col in agg_cols:
            if col not in group:
                continue
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()
        row["target_reached_rate"] = group["target_reached"].mean() if "target_reached" in group else None
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values(["dataset", "alpha", "strategy"]).reset_index(drop=True)


# ─── C1: Pareto frontier ───────────────────────────────────────────────────

def build_pareto_frontier(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark Pareto-optimal (strategy, alpha) points per dataset, using mean
    global accuracy (higher better) vs. mean Gini coefficient (lower
    better) as the two trade-off axes.

    A point is Pareto-optimal if no other point in the same dataset has
    both >= accuracy AND <= Gini, with at least one strictly better.

    Returns:
        DataFrame (one row per dataset x strategy x alpha) with a
        `pareto_optimal` boolean column.
    """
    rows = []
    for dataset, group in summary_df.groupby("dataset"):
        group = group.reset_index(drop=True)
        flags = []
        for i, row in group.iterrows():
            acc_i = row["A1_global_accuracy_mean"]
            gini_i = row["B2_gini_coefficient_mean"]
            dominated = any(
                (other["A1_global_accuracy_mean"] >= acc_i and other["B2_gini_coefficient_mean"] <= gini_i and
                 (other["A1_global_accuracy_mean"] > acc_i or other["B2_gini_coefficient_mean"] < gini_i))
                for j, other in group.iterrows() if j != i
            )
            flags.append(not dominated)
        group["pareto_optimal"] = flags
        rows.append(group)

    return pd.concat(rows, ignore_index=True)


# ─── C2: Two-way ANOVA ──────────────────────────────────────────────────────

def run_all_anovas(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Run strategy x alpha two-way ANOVA per dataset, for each dependent
    variable in DEPENDENT_VARS, using the raw per-seed observations (not
    the aggregated summary table — ANOVA needs the seed-level replicates).

    Returns:
        Nested dict: {dataset: {dependent_var: anova_result_dict}}.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy (and ideally pingouin) is required for ANOVA.")

    results: Dict[str, Dict] = {}
    for dataset, group in df.groupby("dataset"):
        results[dataset] = {}
        for dep_var in DEPENDENT_VARS:
            if dep_var not in group.columns:
                continue
            try:
                results[dataset][dep_var] = run_two_way_anova(group, dep_var)
            except Exception as e:
                results[dataset][dep_var] = {"error": str(e)}
    return results


# ─── Rumusan Masalah 3: alpha threshold where fairness becomes optimal ─────

def summarize_fairness_threshold(pareto_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each dataset and alpha, report which strategies are Pareto-optimal,
    directly answering "at which alpha (if any) is fairness-aware selection
    Pareto-optimal, and is this consistent between MNIST and CIFAR-10?".
    """
    rows = []
    for (dataset, alpha), group in pareto_df.groupby(["dataset", "alpha"]):
        optimal_strategies = sorted(group.loc[group["pareto_optimal"], "strategy"].tolist())
        rows.append({
            "dataset": dataset,
            "alpha": alpha,
            "pareto_optimal_strategies": ", ".join(optimal_strategies) if optimal_strategies else "none",
            "fairness_is_optimal": "fairness" in optimal_strategies,
        })
    return pd.DataFrame(rows).sort_values(["dataset", "alpha"]).reset_index(drop=True)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate and analyze the full FL experiment grid.")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--out_dir", type=str, default=None,
                         help="Default: <results_dir>/analysis")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CROSS-EXPERIMENT ANALYSIS")
    print("=" * 70)

    df = load_all_results(results_dir)
    print(f"  Loaded {len(df)} completed experiments from {results_dir}")

    expected = 3 * 3 * 2 * 3  # strategies x alphas x datasets x seeds
    if len(df) < expected:
        print(f"  WARNING: expected {expected} experiments (full grid), found {len(df)}. "
              f"Analysis will run on the available subset.")

    # Summary table.
    summary_df = build_summary_table(df)
    summary_path = out_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary table -> {summary_path}")

    # C1: Pareto frontier.
    pareto_df = build_pareto_frontier(summary_df)
    pareto_path = out_dir / "pareto_data.csv"
    pareto_df.to_csv(pareto_path, index=False)
    print(f"  Saved Pareto frontier data (C1) -> {pareto_path}")

    # C2: two-way ANOVA.
    try:
        anova_results = run_all_anovas(df)
        anova_path = out_dir / "anova_results.json"
        with open(anova_path, "w") as f:
            json.dump(anova_results, f, indent=2)
        print(f"  Saved two-way ANOVA results (C2) -> {anova_path}")
    except ImportError as e:
        print(f"  SKIPPED ANOVA: {e}")

    # Rumusan Masalah 3: fairness threshold consistency.
    threshold_df = summarize_fairness_threshold(pareto_df)
    threshold_path = out_dir / "fairness_threshold_summary.csv"
    threshold_df.to_csv(threshold_path, index=False)
    print(f"  Saved fairness-threshold summary -> {threshold_path}")

    print("\n" + "-" * 70)
    print("  Pareto-optimal strategy per (dataset, alpha):")
    print("-" * 70)
    print(threshold_df.to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
