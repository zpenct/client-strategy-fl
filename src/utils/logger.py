"""
Structured logging utility for FL experiments.

Provides colored terminal output + file logging with consistent format:
[TIMESTAMP] [LEVEL] [EXPERIMENT_ID] [COMPONENT] MESSAGE

Author: FL Experiment System
Date: 2026
"""

import logging
import os
from datetime import datetime
from pathlib import Path

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)


class ExperimentFormatter(logging.Formatter):
    """Custom formatter: [TIMESTAMP] [LEVEL] [EXP_ID] [COMPONENT] MESSAGE"""

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        exp_id = getattr(record, "experiment_id", "GLOBAL")
        component = getattr(record, "component", "SYSTEM")
        level = record.levelname
        msg = record.getMessage()
        return f"[{timestamp}] [{level:<7}] [{exp_id}] [{component}] {msg}"


def get_logger(experiment_id: str, log_dir: Path = None) -> logging.Logger:
    """
    Return a configured logger for a given experiment.

    Args:
        experiment_id: Unique experiment identifier, e.g. "random_mnist_a0.1_s42"
        log_dir: Directory for log files. Defaults to logs/ at project root.

    Returns:
        logging.Logger: Configured logger with file + colored terminal handlers.
    """
    if log_dir is None:
        log_dir = LOG_DIR

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(experiment_id)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    # --- File handler (no color) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_{experiment_id}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(ExperimentFormatter())
    logger.addHandler(fh)

    # --- Terminal handler (with color if available) ---
    if HAS_COLORLOG:
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] [%(levelname)-7s]%(reset)s "
            "%(cyan)s[%(experiment_id)s]%(reset)s "
            "%(blue)s[%(component)s]%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        ch = colorlog.StreamHandler()
        ch.setFormatter(color_formatter)
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(ExperimentFormatter())

    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger


class ComponentAdapter(logging.LoggerAdapter):
    """Adapter that injects experiment_id and component into every log record."""

    def __init__(self, logger, experiment_id: str, component: str):
        super().__init__(logger, {})
        self.experiment_id = experiment_id
        self.component = component

    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"]["experiment_id"] = self.experiment_id
        kwargs["extra"]["component"] = self.component
        return msg, kwargs


def get_component_logger(experiment_id: str, component: str, log_dir: Path = None):
    """
    Return a logger adapter scoped to a specific component.

    Args:
        experiment_id: Experiment identifier.
        component: Component name, e.g. "PARTITIONER", "CLIENT_03", "STRATEGY".

    Returns:
        ComponentAdapter: Logger with component context baked in.

    Example:
        >>> log = get_component_logger("random_mnist_a0.1_s42", "ROUND_05")
        >>> log.info("Global Acc: 72.34%")
    """
    base_logger = get_logger(experiment_id, log_dir)
    return ComponentAdapter(base_logger, experiment_id, component)


def log_round_summary(logger, round_num: int, total_rounds: int, metrics: dict):
    """
    Print a formatted round summary table.

    Args:
        logger: Logger or ComponentAdapter instance.
        round_num: Current round number (1-indexed).
        total_rounds: Total number of rounds.
        metrics: Dict with keys: global_accuracy, gini_coefficient,
                 accuracy_variance, selected_clients, elapsed_time.
    """
    acc = metrics.get("global_accuracy", 0.0)
    gini = metrics.get("gini_coefficient", 0.0)
    var = metrics.get("accuracy_variance", 0.0)
    selected = metrics.get("selected_clients", [])
    elapsed = metrics.get("elapsed_time", 0.0)

    clients_str = ",".join(str(c) for c in selected)
    msg = (
        f"Round {round_num:02d}/{total_rounds} | "
        f"Acc: {acc:.4f} | "
        f"Gini: {gini:.4f} | "
        f"σ: {var:.4f} | "
        f"Selected: [{clients_str}] | "
        f"Time: {elapsed:.1f}s"
    )

    extra = {"experiment_id": getattr(logger, "experiment_id", "GLOBAL"),
              "component": f"ROUND_{round_num:02d}"}
    if hasattr(logger, "logger"):
        logger.logger.info(msg, extra=extra)
    else:
        logger.info(msg, extra=extra)


def log_experiment_config(logger, config: dict):
    """
    Log experiment configuration as a formatted block at the start.

    Args:
        logger: Logger or ComponentAdapter.
        config: Dict of configuration parameters.
    """
    lines = ["=" * 60, "EXPERIMENT CONFIGURATION", "=" * 60]
    for k, v in config.items():
        lines.append(f"  {k:<25}: {v}")
    lines.append("=" * 60)
    for line in lines:
        extra = {"experiment_id": getattr(logger, "experiment_id", "GLOBAL"),
                 "component": "CONFIG"}
        if hasattr(logger, "logger"):
            logger.logger.info(line, extra=extra)
        else:
            logger.info(line, extra=extra)
