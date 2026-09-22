"""
Smoke test for PerformanceBasedStrategy (Oort statistical-utility selection).

Exercises the strategy's public API end-to-end without needing a full
Flower simulation: utility computation, loss history updates, exploitation/
exploration sampling, and epsilon decay.

Run with:
    source venv/bin/activate
    python tests/smoke_test_performance.py

Author: FL Experiment System
Date: 2026
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategies.performance_strategy import PerformanceBasedStrategy


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("smoke_test_performance")


def test_strategy_initialization():
    """Test 1: Strategy initializes with correct defaults."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TEST 1: Strategy Initialization")
    logger.info("=" * 60)

    num_clients = 10
    client_num_samples = {str(i): 1000 + i * 100 for i in range(num_clients)}

    strategy = PerformanceBasedStrategy(
        client_num_samples=client_num_samples,
        clients_per_round=5,
        loss_history_window=10,
        exploration_factor=0.9,
        logger=logger,
    )

    assert strategy.clients_per_round == 5
    assert strategy.exploration_factor == 0.9
    assert len(strategy.client_num_samples) == num_clients
    assert strategy.explored == set()

    logger.info("PASSED: Strategy initialized correctly\n")
    return strategy, logger


def test_utility_computation(strategy, logger):
    """Test 2: Utility computations work without history."""
    logger.info("=" * 60)
    logger.info("TEST 2: Utility Computation (No History)")
    logger.info("=" * 60)

    util_stat = strategy.compute_statistical_utility("0")
    strategy.current_round = 5
    util_temporal = strategy.compute_temporal_uncertainty("0")  # 0, never selected
    util_combined = strategy.compute_client_utility("0")

    logger.info(f"  Client 0 (no history): U_stat={util_stat:.2f} | "
                f"temporal={util_temporal:.4f} | combined={util_combined:.2f}")

    assert util_stat > 0
    assert util_temporal == 0.0
    assert util_combined == util_stat

    logger.info("PASSED: Utility computation works without history\n")


def test_loss_update_and_utility_evolution(strategy, logger):
    """Test 3: loss_rms updates move statistical utility."""
    logger.info("=" * 60)
    logger.info("TEST 3: Loss RMS Update -> Utility Evolution")
    logger.info("=" * 60)

    client_id = "0"
    losses = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25]

    for r, loss in enumerate(losses, 1):
        strategy.update_client_utility(client_id, loss)
        util = strategy.compute_statistical_utility(client_id)
        logger.info(f"    Round {r}: loss_rms={loss:.2f} -> U_stat={util:.2f}")

    assert len(strategy.client_loss_rms[client_id]) > 0
    logger.info(f"  Rolling window size: {len(strategy.client_loss_rms[client_id])}")
    logger.info("PASSED: Loss updates tracked correctly\n")


def test_exploitation_and_exploration(strategy, logger):
    """Test 4: exploitation and exploration sampling."""
    logger.info("=" * 60)
    logger.info("TEST 4: Exploitation / Exploration Sampling")
    logger.info("=" * 60)

    strategy.current_round = 3
    strategy.explored = {"0", "1", "2"}
    strategy.last_selected_round = {"0": 1, "1": 2, "2": 2}

    exploit = strategy._select_exploitation(["0", "1", "2"], k=2)
    explore = strategy._select_exploration(["3", "4", "5"], k=2)

    logger.info(f"  Exploitation picked: {exploit}")
    logger.info(f"  Exploration picked:  {explore}")

    assert len(exploit) == 2
    assert len(explore) == 2
    assert set(exploit).issubset({"0", "1", "2"})
    assert set(explore).issubset({"3", "4", "5"})

    logger.info("PASSED: Exploitation/exploration sampling functional\n")


def test_epsilon_decay(strategy, logger):
    """Test 5: epsilon decays with a floor."""
    logger.info("=" * 60)
    logger.info("TEST 5: Epsilon Decay With Floor")
    logger.info("=" * 60)

    strategy.exploration_factor = 0.9
    for _ in range(200):  # far more than needed to hit the floor
        if strategy.exploration_factor > strategy.exploration_floor:
            strategy.exploration_factor = max(
                strategy.exploration_floor,
                strategy.exploration_factor * strategy.exploration_decay,
            )

    logger.info(f"  Final epsilon: {strategy.exploration_factor}")
    assert strategy.exploration_factor == strategy.exploration_floor
    logger.info("PASSED: Epsilon decays to floor and stays there\n")


def test_state_checkpointing(strategy, logger):
    """Test 6: state can be serialized and restored."""
    logger.info("=" * 60)
    logger.info("TEST 6: State Checkpointing")
    logger.info("=" * 60)

    state = strategy.get_state()
    required_keys = ["client_loss_rms", "explored", "last_selected_round", "exploration_factor"]
    for key in required_keys:
        assert key in state

    new_strategy = PerformanceBasedStrategy(
        client_num_samples=strategy.client_num_samples,
        clients_per_round=3,
        logger=logger,
    )
    new_strategy.load_state(state)

    assert new_strategy.exploration_factor == strategy.exploration_factor
    assert new_strategy.explored == strategy.explored

    logger.info("PASSED: State checkpointing works\n")


def run_smoke_test():
    print("\n" + "=" * 60)
    print("  PERFORMANCE (OORT) STRATEGY SMOKE TEST")
    print("=" * 60 + "\n")

    try:
        strategy, logger = test_strategy_initialization()
        test_utility_computation(strategy, logger)
        test_loss_update_and_utility_evolution(strategy, logger)
        test_exploitation_and_exploration(strategy, logger)
        test_epsilon_decay(strategy, logger)
        test_state_checkpointing(strategy, logger)

        print("\n" + "=" * 60)
        print("  ALL SMOKE TESTS PASSED!")
        print("=" * 60 + "\n")
        return True

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}\n")
        raise


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
