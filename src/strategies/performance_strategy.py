"""
Performance-based (Oort) client selection strategy.

Faithful implementation of the statistical-utility + exploration-exploitation
bandit mechanism from:

    Lai, F., Zhu, X., Madhyastha, H. V., & Chowdhury, M. (2021).
    Oort: Efficient Federated Learning via Guided Participant Selection.
    OSDI. https://arxiv.org/pdf/2010.06081

Scope note: Oort's original design also weighs a *system utility* term
(round duration vs. a device-speed pacer, straggler penalty) to trade off
statistical efficiency against wall-clock time across heterogeneous devices.
This experiment runs entirely as a single-machine simulation with no real
device heterogeneity, and the thesis proposal does not evaluate system
efficiency at all (only accuracy and fairness under label skew). The system
utility / pacer component is therefore intentionally omitted; this class
implements the statistical utility (Eq. 1's U(i) term) and the full
exploration-exploitation bandit machinery from Algorithm 1 (explored-set
tracking, temporal-uncertainty staleness bonus, percentile-based robustness
clipping, cutoff-pool proportional sampling, epsilon decay).

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, FitRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class PerformanceBasedStrategy(FedAvg):
    """
    Oort-style client selection strategy (statistical utility only).

    Combines:
    - Statistical utility: U(i) = |Bi| * sqrt(mean(Loss(k)^2)), from each
      client's most recent local-training loss report (rolling window).
    - Temporal-uncertainty staleness bonus: sqrt(0.1 * log(R) / L(i)),
      rewarding clients that have not been selected in a while.
    - Robustness: utility values are clipped to a percentile cap before
      ranking, so a single outlier client cannot dominate selection.
    - Exploration-exploitation bandit: a cutoff-pool of top-utility
      *explored* clients is sampled probabilistically (exploitation),
      while never-before-selected clients are sampled uniformly at random
      (exploration). The exploration fraction epsilon decays each round
      down to a floor, per Oort's Section 7.1 configuration.

    Args:
        client_num_samples: Dict mapping client_id (str) → number of local
            training samples |Bi|.
        clients_per_round: Number of clients to select each round (K).
        loss_history_window: Rolling window size for loss_rms averaging.
        exploration_factor: Initial epsilon (fraction of K explored).
        exploration_decay: Multiplicative decay applied to epsilon each
            round while epsilon > exploration_floor.
        exploration_floor: Lower bound epsilon decays toward.
        utility_clip_percentile: Percentile (0-100) used to cap utility
            values before ranking (outlier robustness).
        cutoff_confidence: Confidence factor c used to size the
            exploitation candidate pool (paper default 0.95).
        seed: RNG seed for reproducible bandit sampling.
        logger: Optional logger for structured output.
        **kwargs: Forwarded to FedAvg.
    """

    def __init__(
        self,
        client_num_samples: Dict[str, int],
        clients_per_round: int = 5,
        loss_history_window: int = 10,
        exploration_factor: float = 0.9,
        exploration_decay: float = 0.98,
        exploration_floor: float = 0.2,
        utility_clip_percentile: float = 95.0,
        cutoff_confidence: float = 0.95,
        seed: int = 42,
        logger: logging.Logger = None,
        **kwargs,
    ):
        if not client_num_samples:
            raise ValueError("client_num_samples must be a non-empty dict.")
        super().__init__(**kwargs)

        self.client_num_samples = client_num_samples
        self.clients_per_round = clients_per_round

        # Loss tracking for statistical utility (rolling window of loss_rms)
        self.client_loss_rms: Dict[str, List[float]] = defaultdict(list)
        self.loss_history_window = loss_history_window

        # Exploration-exploitation bandit state
        self.exploration_factor = exploration_factor
        self.exploration_decay = exploration_decay
        self.exploration_floor = exploration_floor
        self.utility_clip_percentile = utility_clip_percentile
        self.cutoff_confidence = cutoff_confidence

        # Explored set E and per-client last-selected round L(i)
        self.explored: Set[str] = set()
        self.last_selected_round: Dict[str, int] = {}
        self.participation_count: Dict[str, int] = defaultdict(int)
        self._raw_to_index: Dict[str, str] = {}

        # State
        self.logger = logger
        self._experiment_id = "UNKNOWN"
        self.current_round = 0

        self._rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────────
    # A. Loss Update (called by server after each fit round)
    # ─────────────────────────────────────────────────────────────

    def update_client_utility(self, client_id: str, loss_rms: float) -> None:
        """
        Record a client's per-sample loss RMS (sqrt(mean(Loss(k)^2))) from
        its most recent local training round, maintaining a rolling window.

        Args:
            client_id: Client identifier (e.g., "0", "1", ...).
            loss_rms: sqrt(mean of squared per-sample training losses),
                reported by FLClient.fit() as the "loss_rms" metric.
        """
        hist = self.client_loss_rms[client_id]
        hist.append(float(loss_rms))
        if len(hist) > self.loss_history_window:
            hist.pop(0)

    # ─────────────────────────────────────────────────────────────
    # B. Statistical Utility (Eq. 1's U(i) term)
    # ─────────────────────────────────────────────────────────────

    def compute_statistical_utility(self, client_id: str) -> float:
        """
        Compute U(i) = |Bi| * sqrt(mean(Loss(k)^2)).

        Uses the rolling-window average of the client's reported loss_rms
        values (each already sqrt(mean squared per-sample loss) for the
        round it was computed in) as the loss term.

        Args:
            client_id: Client identifier.

        Returns:
            Scalar utility value (higher = more valuable to train on).
        """
        num_samples = self.client_num_samples.get(client_id, 1)
        hist = self.client_loss_rms.get(client_id, [])

        if not hist:
            # No history yet: neutral utility based on sample count alone.
            return float(num_samples)

        avg_loss_rms = float(np.mean(hist))
        return float(num_samples * avg_loss_rms)

    # ─────────────────────────────────────────────────────────────
    # C. Temporal Uncertainty (staleness bonus, Algorithm 1 Line 10)
    # ─────────────────────────────────────────────────────────────

    def compute_temporal_uncertainty(self, client_id: str) -> float:
        """
        Compute sqrt(0.1 * log(R) / L(i)), where L(i) is the round the
        client was last selected and R is the current round counter.

        Only meaningful for explored clients (L(i) is defined). Clients
        that were selected a long time ago (small L(i) relative to R)
        receive a larger bonus.

        Args:
            client_id: Client identifier.

        Returns:
            Non-negative staleness bonus.
        """
        L = self.last_selected_round.get(client_id)
        if not L or L <= 0:
            return 0.0
        R = max(2, self.current_round)
        return math.sqrt(0.1 * math.log(R) / L)

    def compute_client_utility(self, client_id: str) -> float:
        """Combined utility: statistical utility + temporal uncertainty."""
        return (
            self.compute_statistical_utility(client_id)
            + self.compute_temporal_uncertainty(client_id)
        )

    # ─────────────────────────────────────────────────────────────
    # D. Exploitation: cutoff pool + proportional sampling
    # ─────────────────────────────────────────────────────────────

    def _select_exploitation(self, explored_candidates: List[str], k: int) -> List[str]:
        """
        Rank explored candidates by utility (clipped for robustness), then
        sample k of them from a cutoff pool, with probability proportional
        to utility (Algorithm 1, Lines 13-15).

        Args:
            explored_candidates: Explored client indices available this round.
            k: Number of clients to select via exploitation.

        Returns:
            List of selected client indices (length <= k).
        """
        if k <= 0 or not explored_candidates:
            return []

        utilities = {c: self.compute_client_utility(c) for c in explored_candidates}

        # Robustness: clip utility to a percentile cap before ranking.
        vals = np.array(list(utilities.values()), dtype=float)
        clip_val = np.percentile(vals, self.utility_clip_percentile)
        utilities = {c: min(u, clip_val) for c, u in utilities.items()}

        ranked = sorted(explored_candidates, key=lambda c: utilities[c], reverse=True)

        # Cutoff pool: top-k candidates with a small confidence margin (c%),
        # approximating Oort's CutoffUtil admission threshold.
        pool_size = min(len(ranked), max(k, math.ceil(k / self.cutoff_confidence)))
        pool = ranked[:pool_size]

        if len(pool) <= k:
            return pool

        weights = np.clip(np.array([utilities[c] for c in pool], dtype=float), 1e-9, None)
        probs = weights / weights.sum()
        chosen = self._rng.choice(pool, size=k, replace=False, p=probs)
        return chosen.tolist()

    # ─────────────────────────────────────────────────────────────
    # E. Exploration: uniform sampling of never-before-selected clients
    # ─────────────────────────────────────────────────────────────

    def _select_exploration(self, unexplored_candidates: List[str], k: int) -> List[str]:
        """
        Sample k clients uniformly at random from those never selected
        before (Algorithm 1, Line 16 — without the speed-based prioritization
        that depended on the omitted system-utility model; see module
        docstring).

        Args:
            unexplored_candidates: Client indices never selected before.
            k: Number of clients to select via exploration.

        Returns:
            List of selected client indices (length <= k).
        """
        if k <= 0 or not unexplored_candidates:
            return []
        k = min(k, len(unexplored_candidates))
        chosen = self._rng.choice(unexplored_candidates, size=k, replace=False)
        return chosen.tolist()

    # ─────────────────────────────────────────────────────────────
    # F. configure_fit (main entry point)
    # ─────────────────────────────────────────────────────────────

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select clients using Oort's statistical-utility bandit.

        Flow:
        1. Split available clients into explored / unexplored sets.
        2. Exploitation: sample (1-eps)*K from explored clients by utility.
        3. Exploration: sample eps*K from unexplored clients uniformly.
        4. Top up with random selection if either pool is exhausted.
        5. Decay epsilon toward its floor.

        Args:
            server_round: Current communication round (1-indexed).
            parameters: Current global model parameters.
            client_manager: Manages available client connections.

        Returns:
            List of (ClientProxy, FitIns) tuples for selected clients.
        """
        self.current_round = server_round
        fit_ins = FitIns(parameters, {})

        available: Dict[str, ClientProxy] = client_manager.all()
        available_cids = list(available.keys())
        if not available_cids:
            return []

        sorted_raw_cids = sorted(available_cids)
        raw_to_index = {raw: str(i) for i, raw in enumerate(sorted_raw_cids)}
        index_to_raw = {v: k for k, v in raw_to_index.items()}
        all_indices = list(raw_to_index.values())
        self._raw_to_index = raw_to_index

        K = min(self.clients_per_round, len(all_indices))
        eps = self.exploration_factor
        explore_k = int(round(K * eps))
        exploit_k = K - explore_k

        explored_indices = [i for i in all_indices if i in self.explored]
        unexplored_indices = [i for i in all_indices if i not in self.explored]

        # Exploitation first; if not enough explored clients exist yet
        # (e.g. early rounds), top up from the unexplored pool.
        exploit_selected = self._select_exploitation(explored_indices, exploit_k)
        if len(exploit_selected) < exploit_k:
            shortfall = exploit_k - len(exploit_selected)
            topup = self._select_exploration(unexplored_indices, shortfall)
            exploit_selected = exploit_selected + topup
            unexplored_indices = [i for i in unexplored_indices if i not in topup]

        remaining_unexplored = [i for i in unexplored_indices if i not in exploit_selected]
        explore_selected = self._select_exploration(remaining_unexplored, explore_k)

        selected_indices = list(dict.fromkeys(exploit_selected + explore_selected))

        # Final safety top-up (tiny client pools / rounding edge cases).
        if len(selected_indices) < K:
            remaining = [i for i in all_indices if i not in selected_indices]
            extra = self._select_exploration(remaining, K - len(selected_indices))
            selected_indices += extra

        utilities_for_log = {c: self.compute_client_utility(c) for c in explored_indices}

        # Update tracking state.
        for idx in selected_indices:
            self.explored.add(idx)
            self.last_selected_round[idx] = server_round
            self.participation_count[idx] += 1

        # Epsilon decay with floor (Oort §7.1: decay by 0.98 while eps > 0.2).
        if self.exploration_factor > self.exploration_floor:
            self.exploration_factor = max(
                self.exploration_floor, self.exploration_factor * self.exploration_decay
            )

        client_instructions = [
            (available[index_to_raw[idx]], fit_ins) for idx in selected_indices
        ]

        self._log_selection(server_round, selected_indices, all_indices, utilities_for_log)

        return client_instructions

    # ─────────────────────────────────────────────────────────────
    # G. aggregate_fit: capture per-client loss_rms, then standard FedAvg
    # ─────────────────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        """
        Extract each participating client's reported loss_rms metric
        (with correct client identity via ClientProxy.cid) to update its
        statistical utility history, then delegate to FedAvg for the
        actual weighted-average aggregation.

        Flower's generic fit_metrics_aggregation_fn callback only receives
        (num_examples, metrics) tuples with no client identity, so per-client
        utility bookkeeping must happen here instead, where `results`
        carries the real ClientProxy for each report.
        """
        for client_proxy, fit_res in results:
            cid_index = self._raw_to_index.get(client_proxy.cid, client_proxy.cid)
            loss_rms = fit_res.metrics.get("loss_rms")
            if loss_rms is not None:
                self.update_client_utility(cid_index, float(loss_rms))

        return super().aggregate_fit(server_round, results, failures)

    # ─────────────────────────────────────────────────────────────
    # H. Logging
    # ─────────────────────────────────────────────────────────────

    def _log_selection(
        self,
        server_round: int,
        selected_indices: List[str],
        all_indices: List[str],
        utilities: Dict[str, float],
    ):
        """Log detailed per-client utility breakdown."""
        selected_set = set(selected_indices)
        header = (
            f"[PERFORMANCE/OORT] Round {server_round:02d} | "
            f"Exploration ε={self.exploration_factor:.4f} | "
            f"Explored={len(self.explored)}/{len(all_indices)}"
        )

        rows = []
        for idx in sorted(all_indices, key=lambda x: int(x)):
            is_explored = idx in self.explored
            is_selected = idx in selected_set
            hist = self.client_loss_rms.get(idx, [])
            avg_loss_rms = np.mean(hist) if hist else -1.0
            util = utilities.get(idx, -1.0 if is_explored else float("nan"))
            last_sel = self.last_selected_round.get(idx, -1)

            row = (
                f"Client {idx:>2} | Explored: {'Y' if is_explored else 'N'} | "
                f"loss_rms: {avg_loss_rms:6.3f} | Util: {util:8.2f} | "
                f"LastSel: {last_sel:3d} | Selected: {'✓' if is_selected else '✗'}"
            )
            rows.append(row)

        rows.append(f"→ Selected clients: {sorted(selected_indices, key=int)}")
        full_msg = "\n".join([header] + rows)

        extra = {"experiment_id": self._experiment_id, "component": "STRATEGY"}
        if self.logger:
            self.logger.info(full_msg, extra=extra)
        else:
            print(full_msg)

    # ─────────────────────────────────────────────────────────────
    # I. Server Integration Helpers
    # ─────────────────────────────────────────────────────────────

    def set_experiment_id(self, experiment_id: str):
        """Inject experiment ID for structured logging."""
        self._experiment_id = experiment_id

    def get_state(self) -> Dict:
        """Get current strategy state for serialization/checkpointing."""
        return {
            "client_loss_rms": dict(self.client_loss_rms),
            "explored": list(self.explored),
            "last_selected_round": dict(self.last_selected_round),
            "participation_count": dict(self.participation_count),
            "exploration_factor": self.exploration_factor,
            "current_round": self.current_round,
        }

    def load_state(self, state: Dict):
        """Restore strategy state from checkpoint."""
        self.client_loss_rms = defaultdict(list, state.get("client_loss_rms", {}))
        self.explored = set(state.get("explored", []))
        self.last_selected_round = state.get("last_selected_round", {})
        self.participation_count = defaultdict(int, state.get("participation_count", {}))
        self.exploration_factor = state.get("exploration_factor", 0.9)
        self.current_round = state.get("current_round", 0)
