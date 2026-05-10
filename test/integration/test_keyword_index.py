"""Integration tests for KeywordIndexManager — full reload, incremental sync,
search, and write hooks.

Uses fakeredis for in-memory Redis.
"""

import time

import numpy as np
import pytest
import pytest_asyncio

from apc_cache.config import APCConfig
from apc_cache.keyword.keyword_index import KeywordIndexManager


@pytest_asyncio.fixture
async def index_mgr(default_cfg: APCConfig, fake_redis):
    """Create and start a KeywordIndexManager backed by fakeredis."""
    mgr = KeywordIndexManager(default_cfg, fake_redis)
    yield mgr
    # Cleanup
    if mgr._running:
        await mgr.stop()


@pytest_asyncio.fixture
async def index_mgr_with_data(default_cfg: APCConfig, fake_redis):
    """Create a manager and pre-populate Redis with keyword embeddings."""
    import struct

    prefix = default_cfg.key_prefix

    # Write two keyword embeddings into Redis
    emb1 = np.array([0.1] * 384, dtype=np.float32)
    emb2 = np.array([0.9] * 384, dtype=np.float32)

    await fake_redis.hset(
        f"{prefix}:kw_meta:test_kw_1",
        mapping={
            "embedding": emb1.tobytes(),
            "model_ver": default_cfg.embed_model_ver,
            "dim": "384",
            "created_at": str(time.time() - 100),
        },
    )
    await fake_redis.hset(
        f"{prefix}:kw_meta:test_kw_2",
        mapping={
            "embedding": emb2.tobytes(),
            "model_ver": default_cfg.embed_model_ver,
            "dim": "384",
            "created_at": str(time.time() - 50),
        },
    )
    await fake_redis.zadd(
        f"{prefix}:kw_timeline",
        {"test_kw_1": time.time() - 100, "test_kw_2": time.time() - 50},
    )

    mgr = KeywordIndexManager(default_cfg, fake_redis)
    yield mgr
    if mgr._running:
        await mgr.stop()


class TestFullReload:
    @pytest.mark.asyncio
    async def test_empty_redis(self, index_mgr):
        await index_mgr._full_reload()
        assert index_mgr.size == 0

    @pytest.mark.asyncio
    async def test_loads_embeddings(self, index_mgr_with_data):
        await index_mgr_with_data._full_reload()
        assert index_mgr_with_data.size == 2
        assert "test_kw_1" in index_mgr_with_data._index
        assert "test_kw_2" in index_mgr_with_data._index

    @pytest.mark.asyncio
    async def test_filters_by_model_ver(self, default_cfg, fake_redis):
        """Keywords with a different model_ver should be skipped."""
        import struct

        prefix = default_cfg.key_prefix
        emb = np.array([0.5] * 384, dtype=np.float32)

        await fake_redis.hset(
            f"{prefix}:kw_meta:old_kw",
            mapping={
                "embedding": emb.tobytes(),
                "model_ver": "v0",  # old version
                "dim": "384",
                "created_at": str(time.time()),
            },
        )
        await fake_redis.zadd(f"{prefix}:kw_timeline", {"old_kw": time.time()})

        mgr = KeywordIndexManager(default_cfg, fake_redis)
        await mgr._full_reload()
        try:
            assert mgr.size == 0  # old_kw filtered out
        finally:
            if mgr._running:
                await mgr.stop()


class TestSearch:
    @pytest.mark.asyncio
    async def test_empty_index(self, index_mgr):
        await index_mgr._full_reload()
        results = index_mgr.search(np.array([0.5] * 384, dtype=np.float32), top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_ordered_by_similarity(self, index_mgr_with_data):
        await index_mgr_with_data._full_reload()

        # Query closer to test_kw_2 (all 0.9) than test_kw_1 (all 0.1)
        query = np.array([0.85] * 384, dtype=np.float32)
        results = index_mgr_with_data.search(query, top_k=5)

        assert len(results) == 2
        assert results[0][0] == "test_kw_2"  # closer match first
        assert results[1][0] == "test_kw_1"
        assert results[0][1] > results[1][1]  # higher similarity

    @pytest.mark.asyncio
    async def test_respects_top_k(self, index_mgr_with_data):
        await index_mgr_with_data._full_reload()

        query = np.array([0.5] * 384, dtype=np.float32)
        results = index_mgr_with_data.search(query, top_k=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_zero_query_vector(self, index_mgr_with_data):
        await index_mgr_with_data._full_reload()
        query = np.zeros(384, dtype=np.float32)
        results = index_mgr_with_data.search(query, top_k=5)
        assert results == []  # zero norm → early return


class TestOnKeywordWritten:
    @pytest.mark.asyncio
    async def test_loads_new_keyword_on_hook(self, index_mgr, default_cfg, fake_redis):
        await index_mgr._full_reload()
        assert index_mgr.size == 0

        # Simulate a template_gen writing a new keyword
        emb = np.array([0.42] * 384, dtype=np.float32)
        prefix = default_cfg.key_prefix
        await fake_redis.hset(
            f"{prefix}:kw_meta:new_kw",
            mapping={
                "embedding": emb.tobytes(),
                "model_ver": default_cfg.embed_model_ver,
                "dim": "384",
                "created_at": str(time.time()),
            },
        )
        await fake_redis.zadd(f"{prefix}:kw_timeline", {"new_kw": time.time()})

        await index_mgr.on_keyword_written("new_kw")
        assert index_mgr.size == 1
        assert "new_kw" in index_mgr._index
        np.testing.assert_array_equal(index_mgr._index["new_kw"], emb)

    @pytest.mark.asyncio
    async def test_ignores_wrong_model_ver(self, index_mgr, default_cfg, fake_redis):
        await index_mgr._full_reload()

        emb = np.array([0.42] * 384, dtype=np.float32)
        await fake_redis.hset(
            f"{default_cfg.key_prefix}:kw_meta:other_kw",
            mapping={
                "embedding": emb.tobytes(),
                "model_ver": "v0",
                "dim": "384",
                "created_at": str(time.time()),
            },
        )
        await index_mgr.on_keyword_written("other_kw")
        assert index_mgr.size == 0  # not loaded, wrong model_ver


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_starts_background_sync(self, default_cfg, fake_redis):
        mgr = KeywordIndexManager(default_cfg, fake_redis)
        try:
            await mgr.start()
            assert mgr._running is True
            assert mgr._sync_task is not None
        finally:
            await mgr.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_sync(self, default_cfg, fake_redis):
        mgr = KeywordIndexManager(default_cfg, fake_redis)
        await mgr.start()
        await mgr.stop()
        assert mgr._running is False
