"""
Pre-batch validation script.

Jalankan ini sebelum run_batch.py untuk memastikan tidak ada error
syntax, import, data, maupun runtime sebelum eksperimen panjang dimulai.

Checks:
  Level 1 — Syntax   : compile semua file .py
  Level 2 — Imports  : import semua modul kode
  Level 3 — Data     : validasi semua partisi MNIST ada dan tidak corrupt
  Level 4 — Pipeline : smoke test 3 strategi × 1 round (end-to-end)

Usage:
    python experiments/validate.py
    python experiments/validate.py --datasets mnist cifar10  # kalau cifar10 juga ada
    python experiments/validate.py --skip_pipeline           # hanya level 1-3

Author: FL Experiment System
Date: 2026
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─── Helpers ─────────────────────────────────────────────────────────────────

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"
WARN = "  [WARN]"

results = []   # (level, name, status, detail)

def record(level, name, status, detail=""):
    results.append((level, name, status, detail))
    icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "~", "WARN": "!"}.get(status, "?")
    color = {"PASS": "\033[92m", "FAIL": "\033[91m",
             "WARN": "\033[93m", "SKIP": "\033[90m"}.get(status, "")
    reset = "\033[0m"
    detail_str = f" — {detail}" if detail else ""
    print(f"  {color}{icon}{reset} {name}{detail_str}")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─── Level 1: Syntax check ───────────────────────────────────────────────────

def check_syntax():
    section("LEVEL 1: Syntax Check")
    import py_compile

    src_files = list(Path("src").rglob("*.py")) + list(Path("experiments").glob("*.py"))
    all_ok = True

    for f in sorted(src_files):
        try:
            py_compile.compile(str(f), doraise=True)
            record(1, str(f), "PASS")
        except py_compile.PyCompileError as e:
            record(1, str(f), "FAIL", str(e))
            all_ok = False

    return all_ok


# ─── Level 2: Import check ───────────────────────────────────────────────────

def check_imports():
    section("LEVEL 2: Import Check")
    all_ok = True

    modules = [
        ("flwr", "import flwr as fl; print(fl.__version__)"),
        ("torch", "import torch; print(torch.__version__)"),
        ("torchvision", "import torchvision"),
        ("numpy", "import numpy as np"),
        ("scipy", "import scipy"),
        ("pandas", "import pandas"),
        ("colorlog", "import colorlog"),
        ("src.utils.logger", "from src.utils.logger import get_logger"),
        ("src.utils.tracer", "from src.utils.tracer import trace_tensor"),
        ("src.data.partitioner", "from src.data.partitioner import create_dirichlet_partition, check_partition_exists"),
        ("src.data.loader", "from src.data.loader import get_client_dataloader, get_test_dataloader"),
        ("src.models.mnist_cnn", "from src.models.mnist_cnn import SimpleCNN"),
        ("src.models.cifar_cnn", "from src.models.cifar_cnn import CIFARCNN"),
        ("src.strategies.random_strategy", "from src.strategies.random_strategy import RandomStrategy"),
        ("src.strategies.performance_strategy", "from src.strategies.performance_strategy import PerformanceBasedStrategy, generate_client_latencies"),
        ("src.strategies.fairness_strategy", "from src.strategies.fairness_strategy import FairnessAwareStrategy"),
        ("src.client.fl_client", "from src.client.fl_client import FLClient, make_client_fn"),
        ("src.metrics.evaluator", "from src.metrics.evaluator import compute_all_metrics, compute_gini_coefficient"),
        ("experiments.run_single", "from experiments.run_single import run_experiment, make_callbacks, build_strategy"),
    ]

    for name, code in modules:
        try:
            exec(code)
            record(2, name, "PASS")
        except Exception as e:
            record(2, name, "FAIL", str(e))
            all_ok = False

    return all_ok


# ─── Level 3: Data validation ────────────────────────────────────────────────

def check_data(datasets=("mnist",)):
    section("LEVEL 3: Data Partition Validation")
    from src.data.partitioner import check_partition_exists

    ALPHAS = [0.1, 0.5, 1.0]
    SEEDS = [42, 123, 456]
    NUM_CLIENTS = 10

    all_ok = True

    for dataset in datasets:
        for alpha in ALPHAS:
            for seed in SEEDS:
                label = f"{dataset} | alpha={alpha} | seed={seed}"

                # Check files exist
                if not check_partition_exists(dataset, alpha, seed, NUM_CLIENTS):
                    record(3, label, "FAIL", "partisi tidak ada — jalankan prepare_data.py dulu")
                    all_ok = False
                    continue

                # Load and validate each client file
                errors = []
                try:
                    import torch
                    from src.data.partitioner import _get_partition_dir
                    part_dir = _get_partition_dir(dataset, alpha, seed)

                    total_samples = 0
                    for i in range(NUM_CLIENTS):
                        pt = part_dir / f"client_{i}.pt"
                        saved = torch.load(pt, weights_only=True)

                        if "data" not in saved or "labels" not in saved:
                            errors.append(f"client_{i}: missing keys")
                            continue

                        data = saved["data"]
                        labels = saved["labels"]

                        # Shape checks
                        expected_shape = (1, 28, 28) if dataset == "mnist" else (3, 32, 32)
                        if data.shape[1:] != torch.Size(expected_shape):
                            errors.append(f"client_{i}: wrong shape {tuple(data.shape[1:])}")

                        if len(data) == 0:
                            errors.append(f"client_{i}: 0 samples")

                        if len(data) != len(labels):
                            errors.append(f"client_{i}: data/label length mismatch")

                        total_samples += len(data)

                    # Check partition_info.json
                    info_path = part_dir / "partition_info.json"
                    with open(info_path) as f:
                        info = json.load(f)
                    if len(info) != NUM_CLIENTS:
                        errors.append(f"partition_info.json: expected {NUM_CLIENTS} entries, got {len(info)}")

                    if errors:
                        record(3, label, "FAIL", "; ".join(errors))
                        all_ok = False
                    else:
                        record(3, label, "PASS", f"{total_samples} total samples")

                except Exception as e:
                    record(3, label, "FAIL", str(e))
                    all_ok = False

    return all_ok


# ─── Level 4: Pipeline smoke test ────────────────────────────────────────────

def check_pipeline(datasets=("mnist",)):
    section("LEVEL 4: Pipeline Smoke Test (1 round per strategy)")
    import torch
    from experiments.run_single import run_experiment

    all_ok = True

    # Only test first dataset and alpha=0.5 (moderate skew, not extreme)
    # Use seed=42 consistently
    SMOKE_CONFIG = dict(
        alpha=0.5,
        seed=42,
        num_rounds=1,
        num_clients=10,
        clients_per_round=5,
        local_epochs=1,   # 1 epoch saja untuk speed
        learning_rate=0.01,
        trace=False,
    )

    dataset = datasets[0]  # test dengan dataset pertama saja

    for strategy in ["random", "performance", "fairness"]:
        label = f"{strategy} | {dataset} | alpha=0.5 | 1 round"
        t0 = time.time()

        try:
            metrics = run_experiment(
                strategy_name=strategy,
                dataset_name=dataset,
                output_dir=Path("results/_validation"),  # folder terpisah
                **SMOKE_CONFIG,
            )
            elapsed = time.time() - t0

            # Validate metric values
            issues = []

            a1 = metrics.get("A1_global_accuracy", 0)
            if not (0 < a1 <= 100):
                issues.append(f"A1={a1} tidak valid (harus 0-100)")

            b1 = metrics.get("B1_accuracy_variance", -1)
            if b1 < 0:
                issues.append(f"B1={b1} negatif")
            elif b1 == 0.0:
                issues.append("B1=0.0 (per_client_accuracies kosong?)")

            b2 = metrics.get("B2_gini_coefficient", -1)
            if not (0 <= b2 <= 1):
                issues.append(f"B2={b2} di luar range [0,1]")

            b3 = metrics.get("B3_participation_fairness", -1)
            if b3 < 0:
                issues.append(f"B3={b3} negatif")

            # Check result files were saved
            exp_id = metrics.get("experiment_id", "")
            result_dir = Path("results/_validation") / exp_id
            for fname in ["config.json", "final_metrics.json",
                          "metrics_per_round.json", "participation_log.json"]:
                if not (result_dir / fname).exists():
                    issues.append(f"file tidak tersimpan: {fname}")

            if issues:
                record(4, label, "WARN",
                       f"{elapsed:.0f}s | A1={a1:.1f}% | Issues: {'; '.join(issues)}")
            else:
                record(4, label, "PASS",
                       f"{elapsed:.0f}s | A1={a1:.1f}% | B1={b1:.4f} | B2={b2:.4f} | B3={b3:.4f}")

        except Exception as e:
            elapsed = time.time() - t0
            record(4, label, "FAIL", f"{elapsed:.0f}s | {type(e).__name__}: {e}")
            print(f"\n    Traceback:\n    {traceback.format_exc().strip()}\n")
            all_ok = False

    return all_ok


# ─── Final summary ────────────────────────────────────────────────────────────

def print_summary(skip_pipeline: bool):
    section("VALIDATION SUMMARY")

    level_names = {1: "Syntax", 2: "Imports", 3: "Data", 4: "Pipeline"}
    by_level = {}
    for level, name, status, detail in results:
        by_level.setdefault(level, []).append(status)

    overall_ok = True
    for level in sorted(by_level.keys()):
        if level == 4 and skip_pipeline:
            continue
        statuses = by_level[level]
        n_pass = statuses.count("PASS")
        n_fail = statuses.count("FAIL")
        n_warn = statuses.count("WARN")
        total = len(statuses)

        if n_fail > 0:
            status_str = f"\033[91m{n_fail} FAILED\033[0m, {n_pass} passed"
            overall_ok = False
        elif n_warn > 0:
            status_str = f"\033[93m{n_warn} WARNING\033[0m, {n_pass} passed"
        else:
            status_str = f"\033[92m{n_pass}/{total} passed\033[0m"

        print(f"  Level {level} ({level_names[level]:<10}): {status_str}")

    print()
    if overall_ok:
        print("\033[92m  ✓ SEMUA VALIDASI PASSED — aman untuk run_batch.py\033[0m")
    else:
        print("\033[91m  ✗ ADA ERROR — perbaiki dulu sebelum run_batch.py\033[0m")
    print()

    return overall_ok


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate FL experiment system before running the full batch."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["mnist"],
        choices=["mnist", "cifar10"],
        help="Datasets to validate (default: mnist)"
    )
    parser.add_argument(
        "--skip_pipeline", action="store_true",
        help="Skip Level 4 pipeline smoke test (faster, but less thorough)"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  FL EXPERIMENT — PRE-BATCH VALIDATION")
    print("="*60)
    print(f"  Datasets : {args.datasets}")
    print(f"  Pipeline : {'SKIP' if args.skip_pipeline else 'YES (1 round per strategy)'}")

    t_start = time.time()

    ok1 = check_syntax()
    ok2 = check_imports()
    ok3 = check_data(args.datasets)

    if not args.skip_pipeline:
        ok4 = check_pipeline(args.datasets)
    else:
        ok4 = True

    print(f"\n  Total validation time: {time.time()-t_start:.0f}s")

    overall = print_summary(args.skip_pipeline)

    # Cleanup validation results folder
    import shutil
    val_dir = Path("results/_validation")
    if val_dir.exists():
        shutil.rmtree(val_dir)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()