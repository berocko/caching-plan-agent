"""Keyword cache (kw_cache) with drift-aware adaptive TTL.

Blueprint §3, path ②, §3.2:
- kw_cache stores query→keyword mappings to skip embedding + LLM on repeats.
- TTL is adaptive: drift rate adjusts it between min and max.
- Side-band drift sampling: 5% of hits re-run normalization to measure drift.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as redis

from apc_cache.config import APCConfig

logger = logging.getLogger(__name__)

# Tracked in-process for adaptive TTL computation
_drift_window: list[bool] = []  # True = stable, False = drifted
_MAX_DRIFT_WINDOW = 100


def record_drift_result(stable: bool) -> None:
    """Record a single drift check result."""
    _drift_window.append(stable)
    if len(_drift_window) > _MAX_DRIFT_WINDOW:
        _drift_window.pop(0)


def compute_drift_rate() -> float:
    """Return the current drift rate from the sliding window."""
    if not _drift_window:
        return 0.0
    return 1.0 - (sum(_drift_window) / len(_drift_window))


def adaptive_ttl(cfg: APCConfig) -> int:
    """Compute adaptive TTL based on current drift rate.

    Blueprint §3.2 table:
    - < 5% drift → 3600s (max)
    - 5-15% drift → 1200s
    - > 15% drift → 300s (min)
    """
    rate = compute_drift_rate()
    if rate < cfg.drift_low_threshold:
        return cfg.kw_cache_ttl_max
    if rate < cfg.drift_high_threshold:
        return 1200
    return cfg.kw_cache_ttl_min


async def get_kw_cache(qhash: str, cfg: APCConfig, r: redis.Redis) -> Optional[str]:
    """Look up a query hash in kw_cache.

    Returns the cached keyword string, or None.
    """
    kw_cache_key = f"{cfg.key_prefix}:kw_cache:{qhash}"
    raw = await r.get(kw_cache_key)
    if raw is not None:
        return raw.decode()
    return None


async def set_kw_cache(qhash: str, keyword: str, cfg: APCConfig, r: redis.Redis) -> None:
    """Store a query→keyword mapping with adaptive TTL."""
    ttl = adaptive_ttl(cfg)
    kw_cache_key = f"{cfg.key_prefix}:kw_cache:{qhash}"
    await r.setex(kw_cache_key, ttl, keyword)
