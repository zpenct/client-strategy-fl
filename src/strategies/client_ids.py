"""
Stable mapping from Flower ClientProxy ids to data-partition indices.

In Flower's simulation engine (flwr>=1.8, verified on 1.32) `ClientProxy.cid`
is a random 64-bit node id, NOT the partition id the client's data was
loaded from (client_fn receives `context.node_config["partition-id"]`).
Strategies that look up per-client metadata (sample counts, device
profiles) by index must therefore use the proxy's `partition_id`, otherwise
client "3" in the strategy is not the client holding client_3.pt.

Author: FL Experiment System
Date: 2026
"""

from __future__ import annotations
from typing import Dict

from flwr.server.client_proxy import ClientProxy


def build_client_index_map(available: Dict[str, ClientProxy]) -> Dict[str, str]:
    """
    Map raw ClientProxy.cid -> partition index string ("0", "1", ...).

    Uses `proxy.partition_id` when the proxy exposes it (Ray simulation
    proxies do). Falls back to enumerating sorted raw cids, which keeps
    unit tests with fake proxies whose cid already equals the index working.

    Args:
        available: client_manager.all() result (raw cid -> proxy).

    Returns:
        Dict raw cid -> partition index as a string.
    """
    if all(getattr(p, "partition_id", None) is not None for p in available.values()):
        return {raw: str(p.partition_id) for raw, p in available.items()}
    return {raw: str(i) for i, raw in enumerate(sorted(available, key=_sort_key))}


def _sort_key(raw_cid: str):
    return (0, int(raw_cid)) if raw_cid.isdigit() else (1, raw_cid)
