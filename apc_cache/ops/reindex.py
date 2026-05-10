"""Reindex keywords — regenerate embeddings for all keywords.

Blueprint §8: When embedding model version changes, use this script
to recompute all embeddings without downtime. Old and new model_ver
keywords coexist in Redis; search filters by model_ver.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import redis.asyncio as redis

from apc_cache.config import APCConfig
from apc_cache.keyword.embedding import encode, init_embedding, shutdown_embedding

logger = logging.getLogger(__name__)


async def reindex_keywords(
    cfg: APCConfig,
    r: redis.Redis,
    new_model_ver: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Recompute embeddings for all keywords using the current model.

    Process:
    1. Scan all apc:kw_meta:* keys.
    2. For each keyword, embed(keyword) → update embedding + model_ver.
    3. Old model_ver keywords remain in Redis (coexistence).

    Args:
        cfg: Configuration.
        r: Redis client.
        new_model_ver: The model version to stamp (e.g. "v2").
        dry_run: If True, only count without writing.

    Returns:
        {"total": int, "reindexed": int, "skipped": int}
    """
    pattern = f"{cfg.key_prefix}:kw_meta:*"
    total = 0
    reindexed = 0
    skipped = 0

    cursor = 0
    while True:
        cursor, keys = await r.scan(cursor, match=pattern, count=100)

        for k in keys:
            total += 1
            kw = k.decode().split(":")[-1]

            data = await r.hgetall(k)
            current_ver = data.get(b"model_ver", b"").decode()

            if current_ver == new_model_ver:
                skipped += 1
                continue

            if dry_run:
                reindexed += 1
                continue

            # Generate new embedding
            try:
                vec = await encode(kw)
                emb_bytes = vec.astype("float32").tobytes()

                await r.hset(
                    k,
                    mapping={
                        "embedding": emb_bytes,
                        "model_ver": new_model_ver,
                        "dim": str(len(vec)),
                    },
                )
                reindexed += 1
                logger.debug("Reindexed keyword: %s (ver=%s)", kw, new_model_ver)
            except Exception:
                logger.exception("Failed to reindex keyword: %s", kw)
                skipped += 1

        if cursor == 0:
            break

    logger.info(
        "Reindex complete: total=%d reindexed=%d skipped=%d dry_run=%s",
        total,
        reindexed,
        skipped,
        dry_run,
    )
    return {"total": total, "reindexed": reindexed, "skipped": skipped}


async def check_model_ver_consistency(cfg: APCConfig, r: redis.Redis) -> dict[str, int]:
    """Report the distribution of model_ver across all keywords.

    Used at startup to detect version skew.
    """
    pattern = f"{cfg.key_prefix}:kw_meta:*"
    dist: dict[str, int] = {}

    cursor = 0
    while True:
        cursor, keys = await r.scan(cursor, match=pattern, count=100)
        for k in keys:
            data = await r.hgetall(k)
            ver = data.get(b"model_ver", b"unknown").decode()
            dist[ver] = dist.get(ver, 0) + 1
        if cursor == 0:
            break

    return dist
