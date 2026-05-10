"""End-to-end tests for the FastAPI application.

Tests full HTTP request/response lifecycle using httpx AsyncClient
with an in-memory Redis.

Covers:
- POST /agent/run (L1 hit, kw_cache hit, miss + large_planner + template_gen)
- GET /health
- GET /metrics/info
"""

from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from fakeredis import aioredis as fake_aioredis
from httpx import ASGITransport, AsyncClient


# ── We need to patch the module-level globals in main.py ──────────
# The FastAPI app relies on global cfg, redis_client, graph, kw_index_mgr
# set during lifespan. For e2e tests, we bypass lifespan and inject
# our own state.


@pytest_asyncio.fixture
async def api_client(default_cfg, fake_redis):
    """Create a FastAPI TestClient with populated module globals.

    We monkey-patch the module-level globals in main.py so that
    when the app's route handlers run, they use our fakeredis instance.
    """
    from apc_cache import main as main_module
    from apc_cache.graph import build_graph

    # Store originals to restore later
    orig_cfg = main_module.cfg
    orig_redis = main_module.redis_client
    orig_graph = main_module.graph
    orig_kw_index = main_module.kw_index_mgr

    # Inject test state
    main_module.cfg = default_cfg
    main_module.redis_client = fake_redis
    main_module.graph = build_graph(default_cfg, fake_redis)

    # Create a minimal kw_index_mgr stub for /metrics/info
    from apc_cache.keyword.keyword_index import KeywordIndexManager
    mgr = KeywordIndexManager(default_cfg, fake_redis)
    main_module.kw_index_mgr = mgr

    transport = ASGITransport(app=main_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    # Restore
    main_module.cfg = orig_cfg
    main_module.redis_client = orig_redis
    main_module.graph = orig_graph
    main_module.kw_index_mgr = orig_kw_index


# ══════════════════════════════════════════════════════════════════
# Health & info endpoints
# ══════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, api_client):
        response = await api_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestMetricsInfo:
    @pytest.mark.asyncio
    async def test_metrics_info_returns_data(self, api_client):
        response = await api_client.get("/metrics/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_ver" in data
        assert "kw_index_size" in data
        assert "cache_max_size" in data


# ══════════════════════════════════════════════════════════════════
# /agent/run — cache miss path
# ══════════════════════════════════════════════════════════════════


class TestAgentRunCacheMiss:
    @pytest.mark.asyncio
    async def test_cache_miss_returns_result(self, api_client):
        """A brand new query should go through the full miss path."""
        payload = {
            "query": "Calculate net profit margin for Q3 2024",
            "context": {"income_statement": {"revenue": 10000, "expenses": 7000}},
            "agent_id": "finance_agent",
            "tools": [{"name": "calculator"}],
            "tools_hash": "test_hash_001",
        }

        response = await api_client.post("/agent/run", json=payload)
        assert response.status_code == 200

        data = response.json()
        # Large_planner + template_gen path → miss, then writes to cache
        assert "cache_hit" in data
        assert "keyword" in data
        assert "final_output" in data
        assert data["iteration_count"] == 0

    @pytest.mark.asyncio
    async def test_miss_then_repeat_is_hit(self, api_client, default_cfg, fake_redis):
        """First request miss, second identical request should hit L1."""
        payload = {
            "query": "Analyze revenue growth",
            "context": {"balance_sheet": {"assets": 5000}},
            "agent_id": "finance_agent",
            "tools": [{"name": "calculator"}],
            "tools_hash": "hash_repeat",
        }

        # First request → miss, template_gen writes L1
        response1 = await api_client.post("/agent/run", json=payload)
        assert response1.status_code == 200

        # Second identical request → L1 hit
        response2 = await api_client.post("/agent/run", json=payload)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["cache_hit"] is True
        assert data2["cache_hit_layer"] == "L1"


# ══════════════════════════════════════════════════════════════════
# /agent/run — L1 hit path (pre-seeded)
# ══════════════════════════════════════════════════════════════════


class TestAgentRunL1Hit:
    @pytest.mark.asyncio
    async def test_pre_seeded_l1_key_hits(self, api_client, default_cfg, fake_redis):
        """Seed an L1 key → request should hit L1."""
        import json
        import time
        from apc_cache.fingerprint import compute_fingerprint, ctx_fp_hash
        from apc_cache.normalize import task_sig

        prefix = default_cfg.key_prefix
        query = "Calculate current ratio"
        agent_id = "finance_agent"
        tools_hash = "l1_test_hash"

        # Build context
        ctx = {"balance_sheet": {"current_assets": 1000, "current_liabilities": 500}}
        tools = [MagicMock(name="calculator")]
        tools[0].name = "calculator"

        ctx_fp = compute_fingerprint(ctx, tools, agent_id)
        ts = task_sig(query, agent_id, tools_hash)
        cfp_hash = ctx_fp_hash(ctx_fp)

        # Create a template
        tpl_id = "l1-hit-template-uuid"
        tpl_data = {
            "template_id": tpl_id,
            "version": "v2.3",
            "schema_hash": tools_hash,
            "ctx_fingerprint": json.dumps(ctx_fp.to_dict()),
            "task": "Calculate current ratio",
            "steps": json.dumps([{"description": "Step 1"}, {"description": "Step 2"}]),
            "created_at": str(time.time()),
            "ttl_seconds": "86400",
        }
        await fake_redis.hset(f"{prefix}:tpl:{tpl_id}", mapping=tpl_data)

        # Write L1 key
        l1_key = f"{prefix}:l1:{agent_id}:{ts}:{cfp_hash}"
        await fake_redis.set(l1_key, tpl_id)

        # Send request
        payload = {
            "query": query,
            "context": ctx,
            "agent_id": agent_id,
            "tools": [{"name": "calculator"}],
            "tools_hash": tools_hash,
        }

        response = await api_client.post("/agent/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["cache_hit"] is True
        assert data["cache_hit_layer"] == "L1"


# ══════════════════════════════════════════════════════════════════
# /agent/run — kw_cache path
# ══════════════════════════════════════════════════════════════════


class TestAgentRunKwCache:
    @pytest.mark.asyncio
    async def test_kw_cache_hit_goes_to_l2_l3(self, api_client, default_cfg, fake_redis):
        """Pre-populate kw_cache → L2/L3 hit on seeded template."""
        import json
        import time
        from apc_cache.normalize import qhash

        prefix = default_cfg.key_prefix
        query = "Calculate working capital from report"
        keyword = "working_capital_ratio"
        tools_hash = "kw_cache_test_hash"

        # Create a compatible template
        from apc_cache.keyword.types import CtxFingerprint
        ctx_fp = CtxFingerprint(
            context_type="financial_report",
            length_bucket="short",
            tools=frozenset(["calculator"]),
            agent_role="finance_agent",
            context_schema=frozenset(["balance_sheet", "income_statement"]),
        )
        tpl_id = "kw-cache-tpl-uuid"
        tpl_data = {
            "template_id": tpl_id,
            "version": "v2.3",
            "schema_hash": tools_hash,
            "ctx_fingerprint": json.dumps(ctx_fp.to_dict()),
            "task": "Calculate working capital ratio",
            "steps": json.dumps([{"description": "Step A"}]),
            "created_at": str(time.time()),
            "ttl_seconds": "86400",
        }
        await fake_redis.hset(f"{prefix}:tpl:{tpl_id}", mapping=tpl_data)
        await fake_redis.sadd(
            f"{prefix}:tpl_idx:finance_agent:{keyword}", tpl_id
        )

        # Write kw_cache entry
        qh = qhash(query)
        await fake_redis.setex(f"{prefix}:kw_cache:{qh}", 3600, keyword)

        # Send request
        payload = {
            "query": query,
            "context": {"balance_sheet": {"assets": 1000}, "income_statement": {"revenue": 500}},
            "agent_id": "finance_agent",
            "tools": [{"name": "calculator"}],
            "tools_hash": tools_hash,
        }

        response = await api_client.post("/agent/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["cache_hit"] is True
        assert data["cache_hit_layer"] == "L2_L3"
        assert data["keyword"] == keyword


# ══════════════════════════════════════════════════════════════════
# Error cases
# ══════════════════════════════════════════════════════════════════


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_empty_query_ok(self, api_client):
        """Empty query should not crash — just cache miss."""
        payload = {
            "query": "",
            "agent_id": "test_agent",
            "tools_hash": "",
        }
        response = await api_client.post("/agent/run", json=payload)
        assert response.status_code == 200  # should not 500

    @pytest.mark.asyncio
    async def test_no_tools_ok(self, api_client):
        """Missing tools/tools_hash should work (fall back to computed hash)."""
        payload = {
            "query": "Simple analysis",
            "agent_id": "default",
        }
        response = await api_client.post("/agent/run", json=payload)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_complex_context(self, api_client):
        """Large nested context should not crash."""
        payload = {
            "query": "Analyze deeply nested data",
            "context": {
                "level1": {
                    "level2": {
                        "level3": "deep value"
                    }
                }
            },
            "agent_id": "test_agent",
        }
        response = await api_client.post("/agent/run", json=payload)
        assert response.status_code == 200
