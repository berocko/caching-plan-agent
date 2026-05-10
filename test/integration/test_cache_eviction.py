"""Integration tests for cache eviction — LRU eviction and cascading cleanup.

Uses fakeredis with pre-seeded templates.
"""

import json
import time

import pytest

from apc_cache.cache.eviction import evict_lru, evict_stale_template
from apc_cache.config import APCConfig
from apc_cache.keyword.types import CtxFingerprint


@pytest_asyncio_mark = pytest.mark.asyncio  # shorthand


class TestEvictLRU:
    @pytest.mark.asyncio
    async def test_no_eviction_when_under_limit(self, small_cache_cfg, seeded_redis):
        """3 templates, cache_max_size=3 → no eviction needed."""
        evicted = await evict_lru(small_cache_cfg, seeded_redis)
        assert evicted == 0

        # All templates should still exist
        prefix = small_cache_cfg.key_prefix
        for tpl_id in [
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "11111111-2222-3333-4444-555555555555",
            "22222222-3333-4444-5555-666666666666",
        ]:
            assert await seeded_redis.exists(f"{prefix}:tpl:{tpl_id}")

    @pytest.mark.asyncio
    async def test_evicts_oldest_when_over_limit(self, small_cache_cfg, seeded_redis):
        """Add a 4th template → should evict oldest (by created_at)."""
        prefix = small_cache_cfg.key_prefix

        # Add a 4th template with the most recent created_at
        new_tpl_id = "99999999-8888-7777-6666-555555555555"
        new_tpl_data = {
            "template_id": new_tpl_id,
            "version": "v2.3",
            "schema_hash": "new_hash",
            "ctx_fingerprint": "{}",
            "task": "new task",
            "steps": "[]",
            "created_at": str(time.time()),  # now → newest
            "ttl_seconds": "86400",
        }
        await seeded_redis.hset(f"{prefix}:tpl:{new_tpl_id}", mapping=new_tpl_data)

        # The oldest template should be revenue_growth (created_at - 200)
        evicted = await evict_lru(small_cache_cfg, seeded_redis)
        assert evicted == 1

        # revenue_growth should be evicted (oldest)
        assert not await seeded_redis.exists(
            f"{prefix}:tpl:11111111-2222-3333-4444-555555555555"
        )

        # The newest template should survive
        assert await seeded_redis.exists(f"{prefix}:tpl:{new_tpl_id}")


class TestCascadingCleanup:
    @pytest.mark.asyncio
    async def test_evict_stale_removes_l1_refs(self, default_cfg, seeded_redis):
        """Evicting a template should also delete its L1 keys and reverse index."""
        prefix = default_cfg.key_prefix
        tpl_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

        # Manually create an L1 key referencing this template
        l1_key = f"{prefix}:l1:finance_agent:test_sig:test_cfphash"
        await seeded_redis.set(l1_key, tpl_id)
        await seeded_redis.sadd(f"{prefix}:tpl_refs:{tpl_id}", l1_key)

        # Evict
        await evict_stale_template(tpl_id, default_cfg, seeded_redis)

        # L1 key should be deleted
        assert not await seeded_redis.exists(l1_key)
        # Reverse index should be deleted
        assert not await seeded_redis.exists(f"{prefix}:tpl_refs:{tpl_id}")
        # Template should be deleted
        assert not await seeded_redis.exists(f"{prefix}:tpl:{tpl_id}")

    @pytest.mark.asyncio
    async def test_evict_stale_removes_from_tpl_idx(self, default_cfg, seeded_redis):
        """Template should be removed from all tpl_idx sets on eviction."""
        prefix = default_cfg.key_prefix
        tpl_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

        await evict_stale_template(tpl_id, default_cfg, seeded_redis)

        # tpl_id should no longer be in any tpl_idx set
        import redis.asyncio as redis

        idx_pattern = f"{prefix}:tpl_idx:*"
        cursor = 0
        while True:
            cursor, keys = await seeded_redis.scan(cursor, match=idx_pattern, count=100)
            for k in keys:
                members = await seeded_redis.smembers(k)
                member_ids = {m.decode() for m in members}
                assert tpl_id not in member_ids, f"tpl_id still in {k}"
            if cursor == 0:
                break

    @pytest.mark.asyncio
    async def test_evict_last_template_cleans_keyword(self, default_cfg, fake_redis):
        """When the last template for a keyword is evicted, keyword metadata is cleaned."""
        prefix = default_cfg.key_prefix

        # Create a single template for a keyword
        tpl_id = "single-tpl-kw-test"
        kw = "test_only_kw"
        agent = "test_agent"

        await fake_redis.hset(
            f"{prefix}:tpl:{tpl_id}",
            mapping={
                "template_id": tpl_id,
                "version": "v2.3",
                "schema_hash": "hash",
                "ctx_fingerprint": "{}",
                "task": "test",
                "steps": "[]",
                "created_at": str(time.time()),
                "ttl_seconds": "86400",
            },
        )
        await fake_redis.sadd(f"{prefix}:tpl_idx:{agent}:{kw}", tpl_id)
        await fake_redis.hset(
            f"{prefix}:kw_meta:{kw}",
            mapping={"model_ver": "v1", "dim": "384", "created_at": str(time.time())},
        )
        await fake_redis.zadd(f"{prefix}:kw_timeline", {kw: time.time()})

        # Verify setup
        assert await fake_redis.exists(f"{prefix}:tpl:{tpl_id}")
        assert await fake_redis.exists(f"{prefix}:kw_meta:{kw}")

        # Evict
        await evict_stale_template(tpl_id, default_cfg, fake_redis)

        # Keyword metadata should be cleaned (no more templates)
        assert not await fake_redis.exists(f"{prefix}:kw_meta:{kw}")
