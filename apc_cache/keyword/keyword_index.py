"""Keyword Vector Index Manager — in-memory keyword → embedding dict.

Blueprint §4.7:
- Full reload at startup (KEYS apc:kw_meta:* → filter by model_ver).
- Periodic incremental sync via ZRANGEBYSCORE on apc:kw_timeline (every 5s).
- Write hook: on_keyword_written() for immediate local update.
- search(): brute-force cosine similarity over local dict (< 1ms for < 2000 keywords).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np
import redis.asyncio as redis

from apc_cache.config import APCConfig

logger = logging.getLogger(__name__)


class KeywordIndexManager:
    """In-memory keyword → embedding index with periodic ZSET-based sync."""

    def __init__(self, cfg: APCConfig, r: redis.Redis) -> None:
        self._cfg = cfg
        self._r = r
        self._index: dict[str, np.ndarray] = {}  # keyword → embedding (384,)
        self._last_sync_ts: float = 0.0
        self._sync_task: Optional[asyncio.Task[None]] = None
        self._running = False

    # ── public API ─────────────────────────────────────────────

    async def start(self) -> None:
        """Full reload + start periodic background sync."""
        await self._full_reload()
        self._running = True
        self._sync_task = asyncio.create_task(self._periodic_sync())

    async def stop(self) -> None:
        """Cancel the background sync task."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Brute-force cosine similarity search over local index.

        Returns list of (keyword, score) sorted by score descending.
        """
        if not self._index:
            return []

        results: list[tuple[str, float]] = []
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        for kw, vec in self._index.items():
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            sim = float(np.dot(query_vec, vec) / (query_norm * vec_norm))
            results.append((kw, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def on_keyword_written(self, kw: str) -> None:
        """Write hook: immediately load the new keyword into local index.

        Called by template_gen after writing a new keyword to Redis.
        """
        meta_key = f"{self._cfg.key_prefix}:kw_meta:{kw}"
        data = await self._r.hgetall(meta_key)
        if not data:
            return

        model_ver = data.get(b"model_ver", b"").decode()
        if model_ver != self._cfg.embed_model_ver:
            return

        emb_bytes = data.get(b"embedding")
        if emb_bytes:
            self._index[kw] = np.frombuffer(emb_bytes, dtype=np.float32)
        else:
            # Phase 1 fallback: keyword exists but no embedding yet
            # In Phase 2, embedding is always provided on write.
            pass

    @property
    def size(self) -> int:
        return len(self._index)

    @property
    def last_sync_ts(self) -> float:
        return self._last_sync_ts

    # ── internal ───────────────────────────────────────────────

    async def _full_reload(self) -> None:
        """Load all keyword embeddings from Redis at startup.

        KEYS apc:kw_meta:* → filter model_ver → load into _index.
        """
        from apc_cache import metrics

        pattern = f"{self._cfg.key_prefix}:kw_meta:*"
        cursor = 0
        loaded = 0
        skipped = 0

        while True:
            cursor, keys = await self._r.scan(cursor, match=pattern, count=100)
            for k in keys:
                data = await self._r.hgetall(k)
                if not data:
                    continue

                model_ver = data.get(b"model_ver", b"").decode()
                if model_ver != self._cfg.embed_model_ver:
                    skipped += 1
                    continue

                emb_bytes = data.get(b"embedding")
                if not emb_bytes:
                    # Phase 1: no embedding yet — skip
                    skipped += 1
                    continue

                kw = k.decode().split(":")[-1]
                self._index[kw] = np.frombuffer(emb_bytes, dtype=np.float32)
                loaded += 1

            if cursor == 0:
                break

        self._last_sync_ts = time.time()
        metrics.kw_index_size.set(loaded)
        logger.info(
            "KeywordIndex full reload: loaded=%d skipped=%d model_ver=%s",
            loaded,
            skipped,
            self._cfg.embed_model_ver,
        )

    async def _periodic_sync(self) -> None:
        """Background task: incremental sync every kw_index_sync_interval seconds."""
        from apc_cache import metrics

        while self._running:
            try:
                await asyncio.sleep(self._cfg.kw_index_sync_interval)
                await self._incremental_sync()
                metrics.kw_index_size.set(len(self._index))
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("KeywordIndex periodic sync error")

    async def _incremental_sync(self) -> None:
        """ZRANGEBYSCORE apc:kw_timeline {last_sync_ts} +inf → load new keywords."""
        timeline_key = f"{self._cfg.key_prefix}:kw_timeline"

        members = await self._r.zrangebyscore(
            timeline_key,
            min=self._last_sync_ts,
            max="+inf",
            withscores=True,
        )

        new_count = 0
        for kw_bytes, score in members:
            kw = kw_bytes.decode()
            if kw not in self._index:
                await self.on_keyword_written(kw)
                new_count += 1

        self._last_sync_ts = time.time()
        if new_count:
            logger.debug("KeywordIndex sync: %d new keywords loaded", new_count)


# ── Global singleton for the lifespan ──────────────────────────

_index_manager: Optional[KeywordIndexManager] = None


def get_index_manager() -> Optional[KeywordIndexManager]:
    return _index_manager


def set_index_manager(mgr: KeywordIndexManager) -> None:
    global _index_manager
    _index_manager = mgr
