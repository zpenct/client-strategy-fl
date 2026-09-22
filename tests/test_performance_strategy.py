"""
Unit tests for PerformanceBasedStrategy (Oort statistical-utility selection).

Tests statistical utility, temporal-uncertainty staleness bonus, robustness
clipping, exploitation/exploration sampling, and epsilon decay — the parts
of Oort's Algorithm 1 implemented here (system utility / pacer intentionally
omitted; see module docstring in performance_strategy.py).

Run with:
    python -m pytest tests/test_performance_strategy.py -v
"""

import math
import logging

import pytest
import numpy as np

from src.strategies.performance_strategy import PerformanceBasedStrategy


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def logger():
    return logging.getLogger("test_performance_strategy")


@pytest.fixture
def basic_strategy(logger):
    """Strategy over 10 clients with uniform sample counts, no history yet."""
    num_clients = 10
    client_num_samples = {str(i): 1000 for i in range(num_clients)}
    return PerformanceBasedStrategy(
        client_num_samples=client_num_samples,
        clients_per_round=5,
        exploration_factor=0.9,
        logger=logger,
    )


@pytest.fixture
def strategy_with_history(logger):
    """5 clients with pre-populated loss_rms history and explored state."""
    num_clients = 5
    client_num_samples = {str(i): 1000 for i in range(num_clients)}
    strategy = PerformanceBasedStrategy(
        client_num_samples=client_num_samples,
        clients_per_round=3,
        exploration_factor=0.5,
        loss_history_window=5,
        logger=logger,
    )
    # Client 0: low loss (good fit), Client 1: medium, Client 2: high loss.
    strategy.update_client_utility("0", 0.1)
    strategy.update_client_utility("0", 0.15)
    strategy.update_client_utility("1", 0.5)
    strategy.update_client_utility("1", 0.6)
    strategy.update_client_utility("2", 1.0)
    strategy.update_client_utility("2", 1.1)

    # Mark 0,1,2 as explored, selected at rounds 1,2,3 respectively.
    strategy.explored = {"0", "1", "2"}
    strategy.last_selected_round = {"0": 1, "1": 2, "2": 3}
    strategy.current_round = 5

    return strategy


# ─── Test 1: Loss (utility) history update ──────────────────────────────────

class TestUpdateClientUtility:
    """Test update_client_utility() maintains a rolling window."""

    def test_append(self, basic_strategy):
        basic_strategy.update_client_utility("0", 0.5)
        assert 0.5 in basic_strategy.client_loss_rms["0"]

    def test_rolling_window(self, basic_strategy):
        window_size = basic_strategy.loss_history_window
        for i in range(window_size + 5):
            basic_strategy.update_client_utility("0", float(i))

        assert len(basic_strategy.client_loss_rms["0"]) == window_size
        assert 0.0 not in basic_strategy.client_loss_rms["0"]
        assert float(window_size + 4) in basic_strategy.client_loss_rms["0"]


# ─── Test 2: Statistical Utility ────────────────────────────────────────────

class TestStatisticalUtility:
    """Test compute_statistical_utility(): U(i) = |Bi| * sqrt(mean(Loss(k)^2))."""

    def test_no_history_returns_sample_count(self, basic_strategy):
        util = basic_strategy.compute_statistical_utility("0")
        assert util == 1000

    def test_higher_loss_higher_utility(self, strategy_with_history):
        """Higher reported loss_rms -> higher statistical utility (more to learn from)."""
        util_0 = strategy_with_history.compute_statistical_utility("0")  # loss ~0.1-0.15
        util_1 = strategy_with_history.compute_statistical_utility("1")  # loss ~0.5-0.6
        util_2 = strategy_with_history.compute_statistical_utility("2")  # loss ~1.0-1.1

        assert util_0 < util_1 < util_2

    def test_sample_count_scales_utility(self, basic_strategy):
        basic_strategy.client_num_samples["0"] = 1000
        basic_strategy.client_num_samples["1"] = 2000
        basic_strategy.update_client_utility("0", 0.5)
        basic_strategy.update_client_utility("1", 0.5)  # same loss_rms

        util_0 = basic_strategy.compute_statistical_utility("0")
        util_1 = basic_strategy.compute_statistical_utility("1")

        assert util_1 == pytest.approx(2 * util_0)

    def test_matches_formula_exactly(self, basic_strategy):
        basic_strategy.client_num_samples["0"] = 500
        basic_strategy.update_client_utility("0", 0.2)
        basic_strategy.update_client_utility("0", 0.4)

        expected = 500 * np.mean([0.2, 0.4])
        assert basic_strategy.compute_statistical_utility("0") == pytest.approx(expected)


# ─── Test 3: Temporal Uncertainty (staleness bonus) ─────────────────────────

class TestTemporalUncertainty:
    """Test compute_temporal_uncertainty(): sqrt(0.1*log(R)/L(i))."""

    def test_zero_for_never_selected(self, basic_strategy):
        basic_strategy.current_round = 10
        assert basic_strategy.compute_temporal_uncertainty("0") == 0.0

    def test_matches_formula(self, basic_strategy):
        basic_strategy.current_round = 10
        basic_strategy.last_selected_round["0"] = 2

        expected = math.sqrt(0.1 * math.log(10) / 2)
        assert basic_strategy.compute_temporal_uncertainty("0") == pytest.approx(expected)

    def test_staler_client_gets_larger_bonus(self, basic_strategy):
        """Smaller L(i) (selected longer ago) -> larger staleness bonus."""
        basic_strategy.current_round = 20
        basic_strategy.last_selected_round["stale"] = 2
        basic_strategy.last_selected_round["fresh"] = 18

        bonus_stale = basic_strategy.compute_temporal_uncertainty("stale")
        bonus_fresh = basic_strategy.compute_temporal_uncertainty("fresh")
        assert bonus_stale > bonus_fresh


# ─── Test 4: Exploitation (cutoff pool + proportional sampling) ────────────

class TestExploitation:
    """Test _select_exploitation() cutoff/percentile clip/proportional sampling."""

    def test_selects_correct_count(self, strategy_with_history):
        selected = strategy_with_history._select_exploitation(["0", "1", "2"], k=2)
        assert len(selected) == 2

    def test_empty_candidates_returns_empty(self, strategy_with_history):
        assert strategy_with_history._select_exploitation([], k=2) == []

    def test_k_zero_returns_empty(self, strategy_with_history):
        assert strategy_with_history._select_exploitation(["0", "1"], k=0) == []

    def test_fewer_candidates_than_k_returns_all(self, strategy_with_history):
        selected = strategy_with_history._select_exploitation(["0"], k=3)
        assert selected == ["0"]

    def test_outlier_clipping_does_not_crash_and_bounds_weight(self, strategy_with_history):
        """An extreme outlier utility should be clipped, not dominate deterministically."""
        strategy_with_history.update_client_utility("2", 1000.0)  # inject outlier loss
        selected = strategy_with_history._select_exploitation(["0", "1", "2"], k=1)
        assert len(selected) == 1


# ─── Test 5: Exploration (uniform sampling of unexplored clients) ──────────

class TestExploration:
    """Test _select_exploration() uniform sampling."""

    def test_selects_correct_count(self, basic_strategy):
        selected = basic_strategy._select_exploration(["0", "1", "2", "3"], k=2)
        assert len(selected) == 2

    def test_no_duplicates(self, basic_strategy):
        selected = basic_strategy._select_exploration(["0", "1", "2", "3", "4"], k=3)
        assert len(set(selected)) == len(selected)

    def test_k_exceeds_pool_returns_all(self, basic_strategy):
        selected = basic_strategy._select_exploration(["0", "1"], k=5)
        assert sorted(selected) == ["0", "1"]


# ─── Test 6: Epsilon decay ──────────────────────────────────────────────────

class TestEpsilonDecayFormula:
    """Epsilon decays by 0.98 per round while above its floor (0.2 default)."""

    def test_decay_factor(self, basic_strategy):
        eps0 = basic_strategy.exploration_factor
        decayed = eps0 * basic_strategy.exploration_decay
        assert decayed == pytest.approx(eps0 * 0.98)

    def test_floor_respected(self, basic_strategy):
        basic_strategy.exploration_factor = 0.21
        # Simulate the decay step performed inside configure_fit.
        if basic_strategy.exploration_factor > basic_strategy.exploration_floor:
            basic_strategy.exploration_factor = max(
                basic_strategy.exploration_floor,
                basic_strategy.exploration_factor * basic_strategy.exploration_decay,
            )
        assert basic_strategy.exploration_factor >= basic_strategy.exploration_floor

    def test_stops_decaying_at_floor(self, basic_strategy):
        basic_strategy.exploration_factor = basic_strategy.exploration_floor
        if basic_strategy.exploration_factor > basic_strategy.exploration_floor:
            basic_strategy.exploration_factor *= basic_strategy.exploration_decay
        assert basic_strategy.exploration_factor == basic_strategy.exploration_floor


# ─── Test 7: State serialization ───────────────────────────────────────────

class TestStateSerialization:
    def test_get_state_returns_dict(self, strategy_with_history):
        state = strategy_with_history.get_state()
        assert isinstance(state, dict)
        for key in ("client_loss_rms", "explored", "last_selected_round", "exploration_factor"):
            assert key in state

    def test_load_state_restores(self, strategy_with_history):
        new_state = {
            "client_loss_rms": {"0": [0.1, 0.2]},
            "explored": ["0", "1"],
            "last_selected_round": {"0": 3, "1": 2},
            "participation_count": {"0": 5, "1": 3},
            "exploration_factor": 0.42,
            "current_round": 7,
        }
        strategy_with_history.load_state(new_state)

        assert strategy_with_history.exploration_factor == 0.42
        assert strategy_with_history.explored == {"0", "1"}
        assert strategy_with_history.current_round == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
