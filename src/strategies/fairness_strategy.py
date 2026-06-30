"""
Fairness-aware client selection strategy for federated learning.

Implements inverse-count weighted probabilistic selection:
  fairness_weight[i] = 1 / (participation_count[i] + 1)

Clients that have been selected less often get higher probability
of being selected, promoting equitable participation.

Based on: Huang et al. (2021) fairness-aware FL client selection concepts.

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FairnessAwareStrategy(FedAvg):
    """
    FedAvg variant with fairness-corrected probabilistic client selection.

    Maintains a participation counter per client. Each round, selection
    probability is inversely proportional to how often each client has
    been selected. This prevents high-capability clients from dominating
    training at the expense of under-represented clients.

    Formula:
        weight_i = 1 / (count_i + 1)
        prob_i   = weight_i / sum(weights)

    Args:
        clients_per_round: Number of clients to select per round (K).
        seed: Seed for numpy random choice (reproducibility).
        logger: Optional logger for structured per-round output.
        **kwargs: Forwarded to FedAvg.

    Attributes:
        participation_count: Dict[str, int] tracking cumulative
            selection count per client_id.
    """

    def __init__(
        self,
        clients_per_round: int = 5,
        seed: int = 42,
        logger: logging.Logger = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clients_per_round = clients_per_round
        self.seed = seed
        self.logger = logger
        self._experiment_id = "UNKNOWN"

        # Core fairness state
        self.participation_count: Dict[str, int] = defaultdict(int)
        self._rng = np.random.default_rng(seed)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select K clients with probability inversely proportional to
        their historical participation count.

        Note: In some Flower versions, ClientProxy.cid is an internal
        node identifier (UUID-like) that is NOT guaranteed to be the
        same string across rounds in a way useful for fairness bookkeeping
        tied to "0","1","2"... Like PerformanceBasedStrategy, we map
        available clients to a STABLE index based on sorted cid order,
        and track participation using that stable index.

        Args:
            server_round: Current communication round.
            parameters: Current global model parameters.
            client_manager: Manages available client connections.

        Returns:
            List of (ClientProxy, FitIns) for selected clients.
        """
        config = {}
        fit_ins = FitIns(parameters, config)

        available: Dict[str, ClientProxy] = client_manager.all()
        sorted_raw_cids = sorted(available.keys())

        # Map stable index -> raw cid, and keep reverse for instructions
        index_to_raw = {str(i): raw_cid for i, raw_cid in enumerate(sorted_raw_cids)}
        stable_indices = list(index_to_raw.keys())

        k = min(self.clients_per_round, len(stable_indices))

        # Compute weights: inversely proportional to participation count
        weights = np.array([
            1.0 / (self.participation_count[idx] + 1)
            for idx in stable_indices
        ])
        probs = weights / weights.sum()

        # Sample without replacement
        chosen_positions = self._rng.choice(
            len(stable_indices), size=k, replace=False, p=probs
        )
        selected_indices = [stable_indices[i] for i in chosen_positions]

        # Update participation counts (keyed by stable index)
        for idx in selected_indices:
            self.participation_count[idx] += 1

        client_instructions = [
            (available[index_to_raw[idx]], fit_ins) for idx in selected_indices
        ]

        # Log selection table
        self._log_selection(server_round, stable_indices, weights, probs, selected_indices)

        return client_instructions

    def _log_selection(
        self,
        server_round: int,
        all_indices: List[str],
        weights: np.ndarray,
        probs: np.ndarray,
        selected_indices: List[str],
    ):
        """
        Print a formatted table of participation counts, weights, and probs.

        Args:
            server_round: Current round number.
            all_indices: All available client stable indices ("0","1",...).
            weights: Raw fairness weights (before normalization).
            probs: Normalized selection probabilities.
            selected_indices: Stable indices chosen this round.
        """
        selected_set = set(selected_indices)
        header = (
            f"[FAIRNESS] Round {server_round:02d} | "
            f"Selected {len(selected_indices)} clients via weighted sampling"
        )
        divider = "  " + "-" * 55
        col_header = f"  {'Client':>6} | {'Count':>5} | {'Weight':>7} | {'Prob':>6} | {'Picked':>6}"

        rows = [header, divider, col_header, divider]
        for idx, w, p in zip(all_indices, weights, probs):
            picked = "  ✓" if idx in selected_set else ""
            count = self.participation_count[idx]  # already updated for selected
            display_count = count - (1 if idx in selected_set else 0)  # before this round
            rows.append(
                f"  {idx:>6} | {display_count:>5} | {w:>7.4f} | {p:>6.4f} |{picked}"
            )
        rows.append(divider)
        rows.append(f"  → Selected: {sorted(selected_indices, key=int)}")

        full_msg = "\n".join(rows)
        extra = {"experiment_id": self._experiment_id, "component": "STRATEGY"}
        if self.logger:
            self.logger.info(full_msg, extra=extra)
        else:
            print(full_msg)

    def get_participation_stats(self) -> Dict:
        """
        Return final participation statistics across all rounds.

        Returns:
            Dict with keys:
                counts (Dict[str, int]): Per-client selection counts.
                std (float): Standard deviation of counts.
                min_count (int): Least-selected client's count.
                max_count (int): Most-selected client's count.
                gini (float): Gini coefficient of participation counts.

        Example:
            >>> stats = strategy.get_participation_stats()
            >>> stats["std"]
            0.943
        """
        counts = dict(self.participation_count)
        if not counts:
            return {"counts": {}, "std": 0.0, "min_count": 0,
                    "max_count": 0, "gini": 0.0}

        values = np.array(list(counts.values()), dtype=float)
        gini = _gini(values)

        return {
            "counts": counts,
            "std": float(np.std(values)),
            "min_count": int(values.min()),
            "max_count": int(values.max()),
            "gini": gini,
        }

    def set_experiment_id(self, experiment_id: str):
        """Inject experiment ID for structured logging."""
        self._experiment_id = experiment_id


def _gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of non-negative values."""
    if values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return float((2 * np.sum((np.arange(1, n + 1)) * values) - (n + 1) * cumsum[-1])
                 / (n * cumsum[-1]))
