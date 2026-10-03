"""
Random client selection strategy for federated learning.

Wraps Flower's default FedAvg behavior (which already does random
selection) and adds logging so we know which clients were picked.

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import logging

import flwr as fl
from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from collections import defaultdict 

from src.strategies.client_ids import build_client_index_map


class RandomStrategy(FedAvg):
    """
    Federated Averaging with random client selection.

    This is the baseline strategy. Selection is entirely random per round,
    which matches Flower's FedAvg default behavior. We override
    configure_fit() only to add structured logging of which clients
    were selected.

    Args:
        logger: Optional logger for per-round selection output.
        **kwargs: All other arguments forwarded to FedAvg.

    Example:
        >>> strategy = RandomStrategy(
        ...     fraction_fit=0.5,
        ...     min_fit_clients=5,
        ...     min_available_clients=10,
        ...     logger=my_logger,
        ... )
    """

    def __init__(self, logger: logging.Logger = None, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self._round_num = 0
        self.current_round = 0                                    # ← TAMBAH
        self.participation_count: Dict[str, int] = defaultdict(int)  # ← TAMBAH

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select clients randomly (default FedAvg behavior) and log selection.

        Note: Logging maps raw ClientProxy.cid values (random node ids in
        Flower's simulation engine) to the client's data-partition index
        (see client_ids.py). Selection itself is unaffected.

        Args:
            server_round: Current communication round (1-indexed).
            parameters: Current global model parameters.
            client_manager: Manages available client connections.

        Returns:
            List of (ClientProxy, FitIns) tuples for selected clients.
        """
        self._round_num = server_round

        # Map raw cid -> partition index for logging/participation counts
        # only (does not affect selection).
        available = client_manager.all()
        raw_to_index = build_client_index_map(available)

        # Use parent's random selection
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        selected_indices = sorted(
            (raw_to_index.get(proxy.cid, proxy.cid) for proxy, _ in client_instructions),
            key=lambda x: int(x) if x.isdigit() else x,
        )

         # ← TAMBAH 3 BARIS INI
        self.current_round = server_round
        for idx in selected_indices:
            self.participation_count[idx] += 1

        if self.logger:
            self.logger.info(
                f"[RANDOM] Round {server_round:02d} | "
                f"Selected {len(selected_indices)} clients: {selected_indices}",
                extra={"experiment_id": getattr(self, "_experiment_id", "UNKNOWN"),
                       "component": f"STRATEGY"}
            )
        else:
            print(f"[RANDOM] Round {server_round:02d} | "
                  f"Selected: {selected_indices}")

        return client_instructions

    def set_experiment_id(self, experiment_id: str):
        """Inject experiment ID for structured logging."""
        self._experiment_id = experiment_id
