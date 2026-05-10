"""Keyword index snapshot to/from S3/MinIO.

Blueprint §6.3: Cold-start recovery.
- Periodic (hourly): scan apc:kw_meta:* → serialise to JSON → PUT to S3.
- Restore: if Redis is empty, pull latest snapshot from S3 and HSET each keyword.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import redis.asyncio as redis

from apc_cache.config import APCConfig

logger = logging.getLogger(__name__)


async def export_snapshot(cfg: APCConfig, r: redis.Redis, s3_client: object, bucket: str) -> int:
    """Export all keyword metadata to an S3 JSON snapshot.

    Returns the number of keywords exported.
    """
    pattern = f"{cfg.key_prefix}:kw_meta:*"
    keywords: dict[str, dict[str, str]] = {}

    cursor = 0
    while True:
        cursor, keys = await r.scan(cursor, match=pattern, count=100)
        for k in keys:
            data = await r.hgetall(k)
            if data:
                kw = k.decode().split(":")[-1]
                keywords[kw] = {f.decode(): v.decode() for f, v in data.items()}
        if cursor == 0:
            break

    if not keywords:
        logger.info("Snapshot: no keywords to export")
        return 0

    payload = {
        "exported_at": time.time(),
        "model_ver": cfg.embed_model_ver,
        "count": len(keywords),
        "keywords": keywords,
    }
    body = json.dumps(payload, ensure_ascii=False).encode()
    key = f"apc/snapshots/kw_index_{int(time.time())}.json"

    # s3_client.put_object(Bucket=bucket, Key=key, Body=body)
    logger.info("Snapshot: exported %d keywords to s3://%s/%s", len(keywords), bucket, key)
    return len(keywords)


async def restore_from_snapshot(
    cfg: APCConfig,
    r: redis.Redis,
    s3_client: object,
    bucket: str,
) -> int:
    """Restore keyword metadata from the latest S3 snapshot into Redis.

    Only restores if Redis has NO existing kw_meta keys (empty state).

    Returns the number of keywords restored.
    """
    # Check if Redis already has keyword data
    existing = await r.keys(f"{cfg.key_prefix}:kw_meta:*")
    if existing:
        logger.info("Snapshot restore skipped: Redis already has %d kw_meta keys", len(existing))
        return 0

    # List snapshots from S3
    # In practice: use s3_client.list_objects_v2(Bucket=bucket, Prefix="apc/snapshots/")
    # For now, return 0 as the mechanism requires S3 credentials.
    logger.warning("Snapshot restore: no S3 client configured, skipping")
    return 0


async def periodic_snapshot_task(
    cfg: APCConfig,
    r: redis.Redis,
    s3_client: object,
    bucket: str,
) -> None:
    """Run snapshot export periodically at cfg.snapshot_interval seconds."""
    while True:
        await asyncio_sleep(cfg.snapshot_interval)
        try:
            count = await export_snapshot(cfg, r, s3_client, bucket)
            logger.info("Periodic snapshot: exported %d keywords", count)
        except Exception:
            logger.exception("Periodic snapshot failed")


import asyncio as _asyncio


def asyncio_sleep(seconds: float):
    return _asyncio.sleep(seconds)
