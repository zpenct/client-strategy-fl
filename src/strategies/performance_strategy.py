"""
Performance-based (latency-aware) client selection strategy.

Selects the fastest clients (lowest simulated latency) each round.
Models a scenario where the server knows approximate communication
or computation latency per client and prefers low-latency ones.

Analogous to Oort / Power-of-Choice: prioritizes responsiveness.

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from collections import defaultdict


def generate_client_latencies(
    num_clients: int,
    seed: int,
    distribution: str = "exponential",
    scale: float = 100.0,
) -> Dict[str, float]:
    """
    Generate simulated per-client latencies.

    Latencies are generated once with a fixed seed and reused across
    all experiments that use this strategy, ensuring fair comparison.

    Args:
        num_clients: Number of clients.
        seed: Random seed for reproducibility.
        distribution: "exponential" or "uniform".
        scale: Scale parameter for the distribution (mean for exponential).

    Returns:
        Dict mapping client_id (str) → latency (ms).

    Example:
        >>> lat = generate_client_latencies(10, seed=42)
        >>> lat["0"]
        74.34...
    """
    rng = np.random.default_rng(seed)
    if distribution == "exponential":
        values = rng.exponential(scale=scale, size=num_clients)
    elif distribution == "uniform":
        values = rng.uniform(low=10.0, high=scale * 2, size=num_clients)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return {str(i): round(float(v), 2) for i, v in enumerate(values)}


class PerformanceBasedStrategy(FedAvg):
    """
    FedAvg variant that always selects the lowest-latency clients.

    In each round, clients are ranked by their simulated latency and
    the top-K fastest are selected. This can cause fairness issues
    if slow clients are never selected.

    Args:
        client_latencies: Dict mapping client_id (str) → latency (float, ms).
        clients_per_round: Number of clients to select each round (K).
        logger: Optional logger for structured output.
        **kwargs: Forwarded to FedAvg.

    Raises:
        ValueError: If client_latencies is empty or None.
    """

    def __init__(
        self,
        client_latencies: Dict[str, float],
        clients_per_round: int = 5,
        logger: logging.Logger = None,
        **kwargs,
    ):
        if not client_latencies:
            raise ValueError("client_latencies must be a non-empty dict.")
        super().__init__(**kwargs)
        self.client_latencies = client_latencies
        self.clients_per_round = clients_per_round
        self.logger = logger
        self._experiment_id = "UNKNOWN"
        self.participation_count: Dict[str, int] = defaultdict(int)  # ← TAMBAH
        self.current_round: int = 0 

        # Pre-sort once (latencies are fixed)
        self._sorted_cids: List[str] = sorted(
            client_latencies.keys(), key=lambda cid: client_latencies[cid]
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select the K fastest available clients for this round.

        Note: In some Flower versions, ClientProxy.cid is an internal
        node identifier (UUID-like), not the simple "0","1","2" index
        used elsewhere in this codebase (e.g. for latency lookup or
        data partition loading). To stay robust across versions, we
        map available clients to a STABLE index based on connection
        order (sorted by cid string for determinism), then use that
        index ("0","1","2",...) to look up latency.

        Args:
            server_round: Current communication round.
            parameters: Current global model parameters.
            client_manager: Manages available client connections.

        Returns:
            List of (ClientProxy, FitIns) for the K fastest clients.
        """
        config = {}
        fit_ins = FitIns(parameters, config)

        # Get available clients, sorted by raw cid for a deterministic order
        available = client_manager.all()  # Dict[str, ClientProxy]
        sorted_raw_cids = sorted(available.keys())

        # Map stable index -> raw cid (e.g. "0" -> "11545664...")
        index_to_raw = {str(i): raw_cid for i, raw_cid in enumerate(sorted_raw_cids)}

        # Select top-K stable indices by latency (pre-sorted)
        selected_indices = [
            idx for idx in self._sorted_cids
            if idx in index_to_raw
        ][:self.clients_per_round]

        self.current_round = server_round
        for idx in selected_indices:
            self.participation_count[idx] += 1

        client_instructions = [
            (available[index_to_raw[idx]], fit_ins) for idx in selected_indices
        ]

        # Log selection with latency and rank
        self._log_selection(server_round, selected_indices, index_to_raw)

        return client_instructions

    def _log_selection(
        self,
        server_round: int,
        selected_indices: List[str],
        index_to_raw: Dict[str, str],
    ):
        """Log per-client latency and rank for selected clients."""
        header = (
            f"[PERFORMANCE] Round {server_round:02d} | "
            f"Selected {len(selected_indices)} clients (by latency ascending)"
        )
        rows = []
        for rank, idx in enumerate(selected_indices, start=1):
            lat = self.client_latencies.get(idx, -1.0)
            rows.append(f"  Rank {rank:02d} | Client {idx:>3} | Latency: {lat:7.2f} ms")

        # Also show skipped clients for transparency
        all_sorted = [c for c in self._sorted_cids if c in index_to_raw]
        not_selected = [c for c in all_sorted if c not in selected_indices]
        rows.append(f"  [NOT SELECTED]: {not_selected}")

        full_msg = "\n".join([header] + rows)

        extra = {"experiment_id": self._experiment_id, "component": "STRATEGY"}
        if self.logger:
            self.logger.info(full_msg, extra=extra)
        else:
            print(full_msg)

    def set_experiment_id(self, experiment_id: str):
        """Inject experiment ID for structured logging."""
        self._experiment_id = experiment_id
