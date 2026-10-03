"""
Fairness-aware client selection strategy for federated learning.

Faithful implementation of FairFedCS from:

    Shi, Y., Liu, Z., Shi, Z., & Yu, H. (2023).
    Fairness-Aware Client Selection for Federated Learning.
    ICME. https://arxiv.org/pdf/2307.10738

FairFedCS jointly considers a Beta-Reputation-System estimate of each
client's contribution quality and a Lyapunov-optimization virtual queue
tracking accumulated selection unfairness, combined into a per-round
Client Suitability Index (CSI) used to pick the top-m clients.

Contribution assessment (client reputation) is grounded in the exact
Shapley Value of each participating client's update, computed over the
2^m subsets of this round's m-client coalition. The original paper uses
GTG-Shapley to approximate this at the scale of thousands of clients; at
our scale (m=5 clients selected per round) exact enumeration is feasible
and strictly more faithful to the paper's Eq. 2 definition, so it is used
here instead of an approximation.

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
import itertools
import logging
import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    NDArrays,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.strategies.client_ids import build_client_index_map
from flwr.server.strategy.aggregate import aggregate as fedavg_aggregate


EvalFn = Callable[[NDArrays], float]


class FairnessAwareStrategy(FedAvg):
    """
    FairFedCS: Fairness-Aware Federated Client Selection.

    Maintains, for every client i in the population:
      - A Beta Reputation r_i = (a_i+1)/(a_i+b_i+2), updated each round a
        client participates based on the sign of its Shapley Value
        contribution to global accuracy.
      - A virtual fairness queue Q_i, which grows whenever a client is
        NOT selected (proportional to its current reputation) and shrinks
        by 1 whenever it is selected — bounding long-run selection
        unfairness via Lyapunov optimization.

    Each round, clients are ranked by the Client Suitability Index
    CSI_i = sigma * r_i + Q_i, and the top-m are selected.

    Args:
        num_clients: Total number of clients N in the population.
        clients_per_round: Number of clients to select per round (m).
        sigma: Trade-off control parameter between reputation and the
            fairness queue in the CSI (paper default 0.6).
        eval_fn: Callable(ndarrays) -> accuracy (0-100 scale) used to
            evaluate arbitrary aggregated-parameter subsets on the
            centralized test set for Shapley Value computation. Must use
            a scratch model instance separate from the main global model
            so subset evaluations never clobber the round-committed model.
        seed: RNG seed (kept for interface parity with other strategies).
        logger: Optional logger for structured output.
        **kwargs: Forwarded to FedAvg.
    """

    def __init__(
        self,
        num_clients: int,
        clients_per_round: int = 5,
        sigma: float = 0.6,
        eval_fn: Optional[EvalFn] = None,
        seed: int = 42,
        logger: logging.Logger = None,
        **kwargs,
    ):
        if eval_fn is None:
            raise ValueError(
                "FairnessAwareStrategy requires eval_fn (ndarrays -> accuracy) "
                "for Shapley Value based reputation updates."
            )
        super().__init__(**kwargs)

        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.sigma = sigma
        self.eval_fn = eval_fn
        self.logger = logger
        self._experiment_id = "UNKNOWN"

        # Beta Reputation System state: Beta(1,1) prior -> r_i = 0.5.
        self.reputation_a: Dict[str, int] = {str(i): 1 for i in range(num_clients)}
        self.reputation_b: Dict[str, int] = {str(i): 1 for i in range(num_clients)}

        # Virtual fairness queue.
        self.queue_Q: Dict[str, float] = {str(i): 0.0 for i in range(num_clients)}

        self.participation_count: Dict[str, int] = defaultdict(int)

        # Cache of the global parameters sent out this round -- used as the
        # empty-coalition Shapley baseline f(w_empty).
        self._current_global_ndarrays: Optional[NDArrays] = None
        self._raw_to_index: Dict[str, str] = {}
        self._last_shapley: Dict[str, float] = {}

        self._rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────────
    # A. Reputation (Beta Reputation System, Eq. 1)
    # ─────────────────────────────────────────────────────────────

    def reputation_of(self, client_id: str) -> float:
        """r_i = E[Beta(a_i+1, b_i+1)] = (a_i+1)/(a_i+b_i+2)."""
        a = self.reputation_a.get(client_id, 1)
        b = self.reputation_b.get(client_id, 1)
        return (a + 1) / (a + b + 2)

    # ─────────────────────────────────────────────────────────────
    # B. configure_fit: Client Suitability Index + virtual queue update
    # ─────────────────────────────────────────────────────────────

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select the top-m clients by Client Suitability Index (Eq. 13),
        then update every client's virtual fairness queue (Eq. 3-4).

        Args:
            server_round: Current communication round.
            parameters: Current global model parameters.
            client_manager: Manages available client connections.

        Returns:
            List of (ClientProxy, FitIns) for selected clients.
        """
        fit_ins = FitIns(parameters, {})

        available: Dict[str, ClientProxy] = client_manager.all()
        available_cids = list(available.keys())
        if not available_cids:
            return []

        raw_to_index = build_client_index_map(available)
        index_to_raw = {v: k for k, v in raw_to_index.items()}
        all_indices = list(raw_to_index.values())
        self._raw_to_index = raw_to_index

        N = len(all_indices)
        m = min(self.clients_per_round, N)
        eps = m / N  # discount factor eps = m/N (paper §3.2)

        # 1-2. Reputation and Client Suitability Index.
        reputations = {idx: self.reputation_of(idx) for idx in all_indices}
        csi = {idx: self.sigma * reputations[idx] + self.queue_Q.get(idx, 0.0)
               for idx in all_indices}

        # 3. Select top-m by CSI (ties broken by client index for determinism).
        ranked = sorted(all_indices, key=lambda idx: (-csi[idx], int(idx)))
        selected_indices = ranked[:m]
        selected_set = set(selected_indices)

        # 4. Update virtual queue for ALL clients (Eq. 3-4).
        for idx in all_indices:
            x_i = 1.0 if idx in selected_set else 0.0
            c_i = eps * reputations[idx] * (1.0 - x_i)
            self.queue_Q[idx] = max(0.0, self.queue_Q.get(idx, 0.0) + c_i - x_i)

        # 5. Cache global params for this round's Shapley baseline f(w_empty).
        self._current_global_ndarrays = parameters_to_ndarrays(parameters)

        for idx in selected_indices:
            self.participation_count[idx] += 1

        client_instructions = [
            (available[index_to_raw[idx]], fit_ins) for idx in selected_indices
        ]

        self._log_selection(server_round, selected_indices, all_indices, reputations, csi)

        return client_instructions

    # ─────────────────────────────────────────────────────────────
    # C. aggregate_fit: FedAvg aggregation + Shapley-based reputation update
    # ─────────────────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict]:
        """
        Aggregate client updates with standard FedAvg weighted averaging,
        then compute the exact Shapley Value of every participating
        client's contribution and update its reputation accordingly.
        """
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        coalition: List[Tuple[str, NDArrays, int]] = []
        for client_proxy, fit_res in results:
            cid_index = self._raw_to_index.get(client_proxy.cid, client_proxy.cid)
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            coalition.append((cid_index, ndarrays, fit_res.num_examples))

        weights_results = [(nd, n) for _, nd, n in coalition]
        aggregated_ndarrays = fedavg_aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        if self._current_global_ndarrays is not None:
            self._update_reputation_via_shapley(coalition)
            self._log_shapley(server_round)

        metrics_aggregated: Dict = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

    def _update_reputation_via_shapley(
        self, coalition: List[Tuple[str, NDArrays, int]]
    ) -> None:
        """
        Compute the exact Shapley Value phi_i for every client i in this
        round's coalition (Eq. 2), then update Beta reputation counts by
        the sign of phi_i: a_i += 1 if phi_i >= 0, else b_i += 1.
        """
        m = len(coalition)
        if m == 0:
            return

        client_ids = [c[0] for c in coalition]
        client_data = {cid: (nd, n) for cid, nd, n in coalition}

        # f(S) cache over all 2^m subsets of the coalition.
        f_cache: Dict[frozenset, float] = {
            frozenset(): self.eval_fn(self._current_global_ndarrays)
        }
        for r in range(1, m + 1):
            for subset in itertools.combinations(client_ids, r):
                weights_results = [client_data[cid] for cid in subset]
                agg_ndarrays = fedavg_aggregate(weights_results)
                f_cache[frozenset(subset)] = self.eval_fn(agg_ndarrays)

        # phi_i = (1/m) * sum_{S subseteq others} [f(S u {i}) - f(S)] / C(m-1, |S|)
        shapley: Dict[str, float] = {}
        for i in client_ids:
            others = [c for c in client_ids if c != i]
            total = 0.0
            for r in range(0, m):
                weight = 1.0 / math.comb(m - 1, r)
                for subset in itertools.combinations(others, r):
                    S = frozenset(subset)
                    S_with_i = S | {i}
                    marginal = f_cache[S_with_i] - f_cache[S]
                    total += marginal * weight
            shapley[i] = total / m

        for cid, phi in shapley.items():
            if phi >= 0:
                self.reputation_a[cid] = self.reputation_a.get(cid, 1) + 1
            else:
                self.reputation_b[cid] = self.reputation_b.get(cid, 1) + 1

        self._last_shapley = shapley

    # ─────────────────────────────────────────────────────────────
    # D. Logging
    # ─────────────────────────────────────────────────────────────

    def _log_selection(
        self,
        server_round: int,
        selected_indices: List[str],
        all_indices: List[str],
        reputations: Dict[str, float],
        csi: Dict[str, float],
    ):
        selected_set = set(selected_indices)
        header = f"[FAIRNESS/FairFedCS] Round {server_round:02d} | sigma={self.sigma}"
        rows = [header]
        for idx in sorted(all_indices, key=int):
            picked = "✓" if idx in selected_set else "✗"
            rows.append(
                f"  Client {idx:>2} | r_i: {reputations[idx]:.4f} | "
                f"Q_i: {self.queue_Q[idx]:7.4f} | CSI: {csi[idx]:.4f} | "
                f"Selected: {picked}"
            )
        rows.append(f"→ Selected clients: {sorted(selected_indices, key=int)}")
        full_msg = "\n".join(rows)

        extra = {"experiment_id": self._experiment_id, "component": "STRATEGY"}
        if self.logger:
            self.logger.info(full_msg, extra=extra)
        else:
            print(full_msg)

    def _log_shapley(self, server_round: int):
        if not self._last_shapley:
            return
        parts = ", ".join(
            f"{cid}:{phi:+.3f}" for cid, phi in sorted(self._last_shapley.items(), key=lambda x: int(x[0]))
        )
        msg = f"[FAIRNESS/FairFedCS] Round {server_round:02d} | Shapley phi_i: {parts}"
        extra = {"experiment_id": self._experiment_id, "component": "SHAPLEY"}
        if self.logger:
            self.logger.info(msg, extra=extra)

    # ─────────────────────────────────────────────────────────────
    # E. Server Integration Helpers
    # ─────────────────────────────────────────────────────────────

    def set_experiment_id(self, experiment_id: str):
        """Inject experiment ID for structured logging."""
        self._experiment_id = experiment_id
