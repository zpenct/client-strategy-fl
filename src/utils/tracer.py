"""
Tensor and data shape tracer for FL experiments.

All trace output is gated behind TRACE_MODE flag — safe to leave
in production code without log spam.

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List


# Global toggle — set via config or env var before any trace call
TRACE_MODE: bool = False


def set_trace_mode(enabled: bool):
    """Enable or disable all tracing globally."""
    global TRACE_MODE
    TRACE_MODE = enabled


def trace_tensor(name: str, tensor: Any, logger: logging.Logger = None):
    """
    Log shape, dtype, and basic stats of a tensor.

    Args:
        name: Human-readable label for this tensor.
        tensor: A torch.Tensor (or anything with .shape, .dtype, .min, .max, .float).
        logger: Logger to write to. Falls back to print if None.

    Output format:
        TRACE | {name} | shape={shape} | dtype={dtype} |
               min={min:.4f} | max={max:.4f} | mean={mean:.4f}
    """
    if not TRACE_MODE:
        return
    try:
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        t_float = tensor.float()
        mn = t_float.min().item()
        mx = t_float.max().item()
        mean = t_float.mean().item()
        msg = (
            f"TRACE | {name} | shape={shape} | dtype={dtype} | "
            f"min={mn:.4f} | max={mx:.4f} | mean={mean:.4f}"
        )
    except Exception as e:
        msg = f"TRACE | {name} | ERROR: {e}"

    _emit(msg, logger, level=logging.DEBUG)


def trace_partition(client_id: int, class_distribution: Dict[int, int],
                    total_samples: int, alpha: float, logger: logging.Logger = None):
    """
    Print an ASCII table of class distribution for one client partition.

    Args:
        client_id: Client index.
        class_distribution: Dict mapping class label -> sample count.
        total_samples: Total number of samples for this client.
        alpha: Dirichlet alpha used to generate partition.
        logger: Logger to write to.

    Example output:
        ┌─────────────────────────────────────────┐
        │ Client 3 | Alpha=0.1 | Total=512 samples│
        ├───────┬────────┬────────────────────────┤
        │ Class │ Count  │ Pct                    │
        ├───────┼────────┼────────────────────────┤
        │   0   │   450  │ ████████████ 87.9%     │
        │   1   │    10  │ ░  1.9%                │
        ...
        └───────┴────────┴────────────────────────┘
    """
    if not TRACE_MODE:
        return

    lines = []
    header = f" Client {client_id} | Alpha={alpha} | Total={total_samples} samples "
    width = max(len(header) + 2, 50)
    lines.append("┌" + "─" * width + "┐")
    lines.append("│" + header.center(width) + "│")
    lines.append("├───────┬────────┬" + "─" * (width - 17) + "┤")
    lines.append("│ Class │  Count │ Distribution" + " " * (width - 30) + "│")
    lines.append("├───────┼────────┼" + "─" * (width - 17) + "┤")

    bar_width = width - 30
    for cls in sorted(class_distribution.keys()):
        count = class_distribution[cls]
        pct = count / total_samples * 100 if total_samples > 0 else 0
        filled = int(pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"│  {cls:3d}  │ {count:6d} │ {bar} {pct:5.1f}%│")

    lines.append("└───────┴────────┴" + "─" * (width - 17) + "┘")

    for line in lines:
        _emit(line, logger, level=logging.DEBUG)


def trace_model_params(model: Any, logger: logging.Logger = None):
    """
    Log total parameter count and estimated model size.

    Args:
        model: A torch.nn.Module.
        logger: Logger to write to.

    Output:
        TRACE | ModelParams | total_params=X | trainable=Y | size=Z KB
    """
    if not TRACE_MODE:
        return
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Approximate: assume float32 (4 bytes)
        size_kb = total * 4 / 1024
        msg = (
            f"TRACE | ModelParams | total_params={total:,} | "
            f"trainable={trainable:,} | size={size_kb:.1f} KB"
        )
    except Exception as e:
        msg = f"TRACE | ModelParams | ERROR: {e}"

    _emit(msg, logger, level=logging.DEBUG)


def trace_client_init(cid: int, dataset_name: str, n_train: int,
                      input_shape: tuple, classes: List[int],
                      alpha: float, logger: logging.Logger = None):
    """
    Log client initialization summary.

    Args:
        cid: Client ID.
        dataset_name: "mnist" or "cifar10".
        n_train: Number of training samples.
        input_shape: Shape of one input tensor (C, H, W).
        classes: List of class labels present in this client's data.
        alpha: Dirichlet alpha.
        logger: Logger to write to.
    """
    if not TRACE_MODE:
        return
    msg = (
        f"TRACE | Client {cid} | Dataset: {dataset_name} | "
        f"Train: {n_train} samples | Input shape: {input_shape} | "
        f"Classes: {classes} | Alpha: {alpha}"
    )
    _emit(msg, logger, level=logging.DEBUG)


def trace_participation(round_num: int, participation_counts: Dict[str, int],
                        logger: logging.Logger = None):
    """
    Print participation count per client every N rounds.

    Args:
        round_num: Current round number.
        participation_counts: Dict mapping client_id -> count.
        logger: Logger to write to.
    """
    if not TRACE_MODE:
        return
    lines = [f"TRACE | Participation after round {round_num}"]
    lines.append(f"  {'Client':<10} {'Count':>6}")
    lines.append("  " + "-" * 18)
    for cid in sorted(participation_counts.keys(), key=lambda x: int(x)):
        count = participation_counts[cid]
        lines.append(f"  {cid:<10} {count:>6}")
    for line in lines:
        _emit(line, logger, level=logging.DEBUG)


def _emit(msg: str, logger, level: int = logging.DEBUG):
    """Internal helper: send message to logger or print."""
    if logger is not None:
        logger.log(level, msg)
    else:
        print(msg)
