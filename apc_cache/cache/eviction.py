"""Cascading cache eviction.

Blueprint §3.6 (path ⑤ step 5.9), §9.3:
- LRU eviction of templates when cache_max_size is exceeded
- Cascading cleanup: tpl_refs → L1 keys → tpl_idx entries → template body
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import redis.asyncio as redis

from apc_cache.config import APCConfig


async def evict_lru(cfg: APCConfig, r: redis.Redis) -> int:
    """Evict the least-recently-used templates until under cache_max_size.

    Returns the number of templates evicted.
    """
    tpl_pattern = f"{cfg.key_prefix}:tpl:*"
    keys = await r.keys(tpl_pattern)
    current_count = len(keys)
    to_evict = max(0, current_count - cfg.cache_max_size)

    if to_evict <= 0:
        return 0

    # Gather (tpl_id, created_at) for sorting by age (oldest first = LRU)
    entries: list[tuple[str, float]] = []
    for k in keys:
        tpl_id = k.decode().split(":")[-1]
        created_raw = await r.hget(k, "created_at")
        if created_raw:
            entries.append((tpl_id, float(created_raw.decode())))
        else:
            entries.append((tpl_id, 0.0))

    # Oldest first
    entries.sort(key=lambda x: x[1])

    evicted = 0
    for tpl_id, _ in entries:
        if evicted >= to_evict:
            break
        await _evict_single(tpl_id, cfg, r)
        evicted += 1

    return evicted


async def _evict_single(tpl_id: str, cfg: APCConfig, r: redis.Redis) -> None:
    """Cascading eviction of a single template.

    Steps (blueprint §9.3):
    1. Delete all L1 keys referenced by tpl_refs
    2. Remove tpl_id from all tpl_idx sets
    3. If keyword has no more templates, clean up keyword metadata
    4. Delete template body
    5. Delete reverse index
    """
    prefix = cfg.key_prefix

    # 1. Find and delete L1 keys via reverse index
    refs_key = f"{prefix}:tpl_refs:{tpl_id}"
    l1_keys = await r.smembers(refs_key)
    if l1_keys:
        await r.delete(*l1_keys)

    # 2. Remove from tpl_idx sets
    # We need to scan tpl_idx keys to find which ones contain this tpl_id.
    # This is a linear scan over keyword index keys.
    idx_pattern = f"{prefix}:tpl_idx:*"
    cursor = 0
    affected_keywords: list[str] = []
    while True:
        cursor, keys = await r.scan(cursor, match=idx_pattern, count=100)
        for k in keys:
            removed = await r.srem(k, tpl_id)
            if removed:
                key_str = k.decode()
                # Extract keyword from "apc:tpl_idx:{agent}:{keyword}"
                parts = key_str.split(":")
                if len(parts) >= 4:
                    affected_keywords.append(parts[-1])
        if cursor == 0:
            break

    # 3. Clean up empty keywords
    for kw in affected_keywords:
        # Check all tpl_idx for this keyword across agents
        kw_pattern = f"{prefix}:tpl_idx:*:{kw}"
        empty = True
        cursor2 = 0
        while True:
            cursor2, idx_keys = await r.scan(cursor2, match=kw_pattern, count=100)
            for ik in idx_keys:
                card = await r.scard(ik)
                if card > 0:
                    empty = False
                    break
            if not empty or cursor2 == 0:
                break
            if cursor2 == 0:
                break

        if empty:
            await r.delete(f"{prefix}:kw_meta:{kw}")
            await r.zrem(f"{prefix}:kw_timeline", kw)

    # 4. Delete template body
    await r.delete(f"{prefix}:tpl:{tpl_id}")

    # 5. Delete reverse index
    await r.delete(refs_key)


async def evict_stale_template(tpl_id: str, cfg: APCConfig, r: redis.Redis) -> None:
    """Public entry point for stale template eviction triggered during lookup."""
    await _evict_single(tpl_id, cfg, r)
