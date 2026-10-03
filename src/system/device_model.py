"""
Simulated device (system) heterogeneity for federated learning experiments.

The experiments run every client on one machine, so real per-device speed
differences do not exist. Oort's system utility and pacer (Lai et al., 2021,
Eq. 1 and Algorithm 1) need a per-client round duration t_i, and the
time-to-accuracy metric needs a simulated wall clock. This module provides
both, following Oort's own evaluation methodology of emulating device
runtimes rather than measuring them (§7.1, FedScale traces from AI
Benchmark and MobiPerf).

Modelling assumptions (must be stated in the thesis):
  - Each client gets a fixed compute speed (samples/second) and network
    bandwidth (MB/s), drawn once per seed from log-normal distributions.
    Log-normal gives the order-of-magnitude spread across devices that
    Oort reports for real phones (Figure 2).
  - Device profiles are drawn from their own RNG stream, independent of
    the data partition, so speed is never correlated with label skew.
  - Round duration of client i:
        t_i = |B_i| * E / compute_speed_i + 2 * model_size_MB / bandwidth_i
    (local training over all samples for E epochs + model download/upload).
  - Synchronous FedAvg: a round lasts as long as its slowest selected
    client, i.e. round_duration = max_{i in P} t_i. No over-commitment or
    dropouts are simulated.

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


# Offset keeps the device RNG stream independent from the data-partition
# and model-init RNG streams that use the plain experiment seed.
DEVICE_SEED_OFFSET = 10_000

# Log-normal medians/spreads. sigma=1.0 puts ~95% of devices within a
# factor of ~7 of the median on either side (≈50x fastest-to-slowest),
# the same order of magnitude as Oort Figure 2.
DEFAULT_COMPUTE_MEDIAN = 200.0     # samples / second
DEFAULT_COMPUTE_SIGMA = 1.0
DEFAULT_BANDWIDTH_MEDIAN = 5.0     # MB / second
DEFAULT_BANDWIDTH_SIGMA = 1.0


@dataclass(frozen=True)
class DeviceProfile:
    """Static system capability of one simulated client device."""
    compute_speed: float   # samples / second
    bandwidth: float       # MB / second


class DeviceModel:
    """
    Per-client device profiles and simulated round durations.

    Args:
        client_num_samples: Dict client index (str) -> local sample count |B_i|.
        local_epochs: Local epochs E per round.
        model_size_mb: Size of the model parameters in MB (sent both ways).
        seed: Experiment seed (offset internally, see DEVICE_SEED_OFFSET).
        compute_median / compute_sigma: Log-normal params for compute speed.
        bandwidth_median / bandwidth_sigma: Log-normal params for bandwidth.

    Example:
        >>> dm = DeviceModel({"0": 1000, "1": 5000}, local_epochs=3,
        ...                  model_size_mb=4.3, seed=42)
        >>> dm.round_duration(["0", "1"])
        41.7...
    """

    def __init__(
        self,
        client_num_samples: Dict[str, int],
        local_epochs: int,
        model_size_mb: float,
        seed: int,
        compute_median: float = DEFAULT_COMPUTE_MEDIAN,
        compute_sigma: float = DEFAULT_COMPUTE_SIGMA,
        bandwidth_median: float = DEFAULT_BANDWIDTH_MEDIAN,
        bandwidth_sigma: float = DEFAULT_BANDWIDTH_SIGMA,
    ):
        if not client_num_samples:
            raise ValueError("client_num_samples must be a non-empty dict.")
        self.client_num_samples = client_num_samples
        self.local_epochs = local_epochs
        self.model_size_mb = model_size_mb

        rng = np.random.default_rng(seed + DEVICE_SEED_OFFSET)
        # Sort by integer index so profile assignment does not depend on
        # dict insertion order.
        cids = sorted(client_num_samples, key=int)
        compute = rng.lognormal(np.log(compute_median), compute_sigma, size=len(cids))
        bandwidth = rng.lognormal(np.log(bandwidth_median), bandwidth_sigma, size=len(cids))
        self.profiles: Dict[str, DeviceProfile] = {
            cid: DeviceProfile(float(c), float(b))
            for cid, c, b in zip(cids, compute, bandwidth)
        }

    def client_duration(self, client_id: str) -> float:
        """Simulated seconds for client i to finish one round (t_i)."""
        profile = self.profiles[client_id]
        n = self.client_num_samples[client_id]
        compute_time = n * self.local_epochs / profile.compute_speed
        comm_time = 2.0 * self.model_size_mb / profile.bandwidth
        return compute_time + comm_time

    def round_duration(self, selected: Iterable[str]) -> float:
        """Synchronous round duration: the slowest selected client."""
        durations = [self.client_duration(cid) for cid in selected]
        return max(durations) if durations else 0.0

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Serializable per-client profile + duration, for results logging."""
        return {
            cid: {
                "compute_speed": p.compute_speed,
                "bandwidth": p.bandwidth,
                "num_samples": self.client_num_samples[cid],
                "duration": self.client_duration(cid),
            }
            for cid, p in sorted(self.profiles.items(), key=lambda x: int(x[0]))
        }
