"""Keyword alias management CLI.

Blueprint §9.5: Operator tool for merging semantically equivalent keywords.
Alias mapping lives in apc:kw_alias (Hash): alias → canonical_keyword.

Not on the hot path — used via admin API or CLI.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import redis.asyncio as redis

from apc_cache.config import APCConfig

logger = logging.getLogger(__name__)


async def set_alias(
    alias: str,
    canonical: str,
    cfg: APCConfig,
    r: redis.Redis,
) -> bool:
    """Create an alias mapping.

    Returns True if created, False if alias already exists in the index
    as a real keyword (conflict).
    """
    # Check that alias is not already a canonical keyword with templates
    idx_pattern = f"{cfg.key_prefix}:tpl_idx:*:{alias}"
    cursor = 0
    while True:
        cursor, keys = await r.scan(cursor, match=idx_pattern, count=10)
        for k in keys:
            card = await r.scard(k)
            if card > 0:
                logger.warning(
                    "Cannot create alias: %s is an active keyword with templates", alias
                )
                return False
        if cursor == 0:
            break

    alias_key = f"{cfg.key_prefix}:kw_alias"
    await r.hset(alias_key, alias, canonical)
    logger.info("Alias set: %s → %s", alias, canonical)
    return True


async def remove_alias(alias: str, cfg: APCConfig, r: redis.Redis) -> bool:
    """Remove an alias mapping."""
    alias_key = f"{cfg.key_prefix}:kw_alias"
    removed = await r.hdel(alias_key, alias)
    if removed:
        logger.info("Alias removed: %s", alias)
        return True
    logger.warning("Alias not found: %s", alias)
    return False


async def list_aliases(cfg: APCConfig, r: redis.Redis) -> dict[str, str]:
    """List all alias mappings."""
    alias_key = f"{cfg.key_prefix}:kw_alias"
    raw = await r.hgetall(alias_key)
    return {k.decode(): v.decode() for k, v in raw.items()}


async def resolve_alias(keyword: str, cfg: APCConfig, r: redis.Redis) -> str:
    """Resolve a keyword through the alias table (one hop).

    Returns the canonical form, or the original if no alias exists.
    """
    alias_key = f"{cfg.key_prefix}:kw_alias"
    canonical = await r.hget(alias_key, keyword)
    if canonical:
        return canonical.decode()
    return keyword


async def merge_keywords(
    source: str,
    target: str,
    cfg: APCConfig,
    r: redis.Redis,
) -> dict[str, int]:
    """Merge all templates from *source* keyword into *target* keyword.

    This is a deeper operation than alias: it actually moves template
    references and removes the source keyword from the index.

    Returns {"moved_templates": int}.
    """
    moved = 0
    idx_pattern = f"{cfg.key_prefix}:tpl_idx:*:{source}"

    cursor = 0
    while True:
        cursor, keys = await r.scan(cursor, match=idx_pattern, count=100)
        for k in keys:
            # Get all template IDs for source
            tpl_ids = await r.smembers(k)
            if tpl_ids:
                # Build target key by replacing source with target
                target_key = k.decode().replace(f":{source}", f":{target}")
                await r.sadd(target_key, *tpl_ids)
                moved += len(tpl_ids)
            # Delete source index entry
            await r.delete(k)
        if cursor == 0:
            break

    # Clean up source keyword metadata
    await r.delete(f"{cfg.key_prefix}:kw_meta:{source}")
    await r.zrem(f"{cfg.key_prefix}:kw_timeline", source)

    # Create alias
    await set_alias(source, target, cfg, r)

    logger.info("Merged keyword %s → %s: %d templates moved", source, target, moved)
    return {"moved_templates": moved}
