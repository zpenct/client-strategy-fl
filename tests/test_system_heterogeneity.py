"""
Unit tests for simulated device heterogeneity and Oort's system utility.

Run with:
    python -m pytest tests/test_system_heterogeneity.py -v
"""

import logging

import pytest

from src.system.device_model import DeviceModel
from src.strategies.client_ids import build_client_index_map
from src.strategies.performance_strategy import PerformanceBasedStrategy
from experiments.run_single import compute_simulated_time_metrics


@pytest.fixture
def samples():
    return {str(i): 1000 * (i + 1) for i in range(10)}


@pytest.fixture
def device_model(samples):
    return DeviceModel(samples, local_epochs=3, model_size_mb=4.0, seed=42)


# ─── DeviceModel ─────────────────────────────────────────────────────────────

class TestDeviceModel:

    def test_deterministic_per_seed(self, samples):
        a = DeviceModel(samples, 3, 4.0, seed=42).to_dict()
        b = DeviceModel(samples, 3, 4.0, seed=42).to_dict()
        c = DeviceModel(samples, 3, 4.0, seed=123).to_dict()
        assert a == b
        assert a != c

    def test_profiles_independent_of_sample_counts(self, samples):
        """Speed must not depend on data (no correlation with label skew)."""
        other = {k: 7 for k in samples}
        a = DeviceModel(samples, 3, 4.0, seed=42).profiles
        b = DeviceModel(other, 3, 4.0, seed=42).profiles
        assert a == b

    def test_duration_formula(self, device_model, samples):
        p = device_model.profiles["2"]
        expected = samples["2"] * 3 / p.compute_speed + 2 * 4.0 / p.bandwidth
        assert device_model.client_duration("2") == pytest.approx(expected)

    def test_round_duration_is_slowest_client(self, device_model):
        sel = ["0", "3", "7"]
        assert device_model.round_duration(sel) == max(
            device_model.client_duration(c) for c in sel)
        assert device_model.round_duration([]) == 0.0

    def test_heterogeneity_spread(self, device_model):
        speeds = [p.compute_speed for p in device_model.profiles.values()]
        assert max(speeds) / min(speeds) > 3


# ─── Oort system utility & pacer ─────────────────────────────────────────────

@pytest.fixture
def oort(samples, device_model):
    return PerformanceBasedStrategy(
        client_num_samples=samples,
        clients_per_round=5,
        device_model=device_model,
        pacer_window=2,
        logger=logging.getLogger("test"),
    )


class TestOortSystemUtility:

    def test_no_device_model_means_no_penalty(self, samples):
        s = PerformanceBasedStrategy(client_num_samples=samples)
        assert s.preferred_duration is None
        assert all(s.compute_system_penalty(c) == 1.0 for c in samples)

    def test_penalty_only_for_stragglers(self, oort, device_model, samples):
        T = oort.preferred_duration
        for c in samples:
            t = device_model.client_duration(c)
            pen = oort.compute_system_penalty(c)
            if t <= T:
                assert pen == 1.0
            else:
                assert pen == pytest.approx((T / t) ** 2)
                assert pen < 1.0

    def test_utility_multiplied_by_penalty(self, oort, samples):
        oort.update_client_utility("0", 1.0)
        base = oort.compute_statistical_utility("0") + oort.compute_temporal_uncertainty("0")
        assert oort.compute_client_utility("0") == pytest.approx(
            base * oort.compute_system_penalty("0"))

    def test_pacer_relaxes_T_when_utility_drops(self, oort):
        T0 = oort.preferred_duration
        oort.round_utility_history = [10.0, 10.0, 1.0, 1.0]  # 2W rounds, dropped
        oort.update_pacer()
        assert oort.preferred_duration > T0
        assert oort.round_threshold_pct == pytest.approx(35.0)

    def test_pacer_keeps_T_when_utility_rises(self, oort):
        T0 = oort.preferred_duration
        oort.round_utility_history = [1.0, 1.0, 10.0, 10.0]
        oort.update_pacer()
        assert oort.preferred_duration == T0

    def test_pacer_waits_for_two_windows(self, oort):
        T0 = oort.preferred_duration
        oort.round_utility_history = [10.0, 1.0, 1.0]
        oort.update_pacer()
        assert oort.preferred_duration == T0

    def test_exploration_prefers_fast_devices(self, oort, device_model, samples):
        fastest = min(samples, key=device_model.client_duration)
        slowest = max(samples, key=device_model.client_duration)
        picks = {fastest: 0, slowest: 0}
        for _ in range(500):
            chosen = oort._select_exploration([fastest, slowest], 1)[0]
            picks[chosen] += 1
        assert picks[fastest] > picks[slowest]


# ─── Client id mapping ──────────────────────────────────────────────────────

class _Proxy:
    def __init__(self, cid, partition_id=None):
        self.cid = cid
        if partition_id is not None:
            self.partition_id = partition_id


class TestClientIndexMap:

    def test_uses_partition_id(self):
        """Flower cids are random node ids; the index must be partition_id."""
        available = {"998877": _Proxy("998877", 0), "112233": _Proxy("112233", 1)}
        assert build_client_index_map(available) == {"998877": "0", "112233": "1"}

    def test_fallback_numeric_sort(self):
        available = {c: _Proxy(c) for c in ["10", "2", "0"]}
        assert build_client_index_map(available) == {"0": "0", "2": "1", "10": "2"}


# ─── Simulated time metrics ─────────────────────────────────────────────────

class TestSimulatedTime:

    def test_round_durations_from_participation_log(self, device_model):
        log = [{"0": 1, "1": 1}, {"0": 1, "1": 2, "2": 1}]
        out = compute_simulated_time_metrics(log, device_model, [50.0, 90.0], 85.0)
        d0 = device_model.round_duration(["0", "1"])
        d1 = device_model.round_duration(["1", "2"])
        assert out["sim_round_durations"] == pytest.approx([d0, d1])
        assert out["sim_total_time_seconds"] == pytest.approx(d0 + d1)
        assert out["A2_time_to_target_seconds"] == pytest.approx(d0 + d1)

    def test_target_never_reached(self, device_model):
        out = compute_simulated_time_metrics([{"0": 1}], device_model, [10.0], 85.0)
        assert out["A2_time_to_target_seconds"] is None
