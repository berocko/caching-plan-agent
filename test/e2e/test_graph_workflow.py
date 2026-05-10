"""End-to-end tests for the full LangGraph workflow.

Validates every edge in the graph routing table (blueprint §5.2):
- entry → cache_lookup
- cache_lookup → small_planner (L1 hit)
- cache_lookup → large_planner (miss)
- small_planner → END (is_complete=True)
- large_planner → template_gen (is_complete=True)
- large_planner → actor (is_complete=False)
- actor → small_planner / large_planner (based on cache_hit)
- template_gen → END
"""

import json
import hashlib
import time
from unittest.mock import MagicMock

import pytest

from apc_cache.config import APCConfig
from apc_cache.fingerprint import compute_fingerprint, ctx_fp_hash
from apc_cache.graph import (
    _route_after_actor,
    _route_after_cache_lookup,
    _route_after_large_planner,
    _route_after_planner,
    build_graph,
)
from apc_cache.normalize import qhash, task_sig


class MockTool:
    def __init__(self, name: str):
        self.name = name


# ══════════════════════════════════════════════════════════════════
# Routing function unit tests (part of e2e since they define
# the graph topology)
# ══════════════════════════════════════════════════════════════════


class TestRoutingFunctions:
    def test_route_after_cache_lookup_hit(self):
        assert _route_after_cache_lookup({"cache_hit": True}) == "small_planner"

    def test_route_after_cache_lookup_miss(self):
        assert _route_after_cache_lookup({"cache_hit": False}) == "large_planner"

    def test_route_after_planner_complete(self):
        assert _route_after_planner({"is_complete": True}) == "END"

    def test_route_after_planner_incomplete(self):
        assert _route_after_planner({"is_complete": False}) == "actor"

    def test_route_after_large_planner_complete(self):
        assert _route_after_large_planner({"is_complete": True}) == "template_gen"

    def test_route_after_large_planner_incomplete(self):
        assert _route_after_large_planner({"is_complete": False}) == "actor"

    def test_route_after_actor_cache_hit(self):
        assert _route_after_actor({"cache_hit": True}) == "small_planner"

    def test_route_after_actor_cache_miss(self):
        assert _route_after_actor({"cache_hit": False}) == "large_planner"


# ══════════════════════════════════════════════════════════════════
# Full graph execution tests
# ══════════════════════════════════════════════════════════════════


class TestGraphCompiles:
    def test_build_graph_returns_compiled_graph(self, default_cfg, fake_redis):
        graph = build_graph(default_cfg, fake_redis)
        assert graph is not None
        # Should have .ainvoke method
        assert hasattr(graph, "ainvoke")


class TestGraphCacheMissPath:
    """cache_lookup → MISS → large_planner → template_gen → END"""

    @pytest.mark.asyncio
    async def test_full_miss_flow(self, default_cfg, fake_redis):
        graph = build_graph(default_cfg, fake_redis)

        initial_state: dict = {
            "query": "Calculate working capital ratio for Q3",
            "context": {"balance_sheet": {"assets": 1000}},
            "agent_id": "finance_agent",
            "tools": [MockTool("calculator")],
            "tools_hash": "abc123",
            "keyword": None,
            "cache_hit": False,
            "cache_hit_layer": "",
            "plan_template": None,
            "current_plan": None,
            "actor_responses": [],
            "execution_log": [],
            "iteration_count": 0,
            "final_output": None,
            "is_complete": False,
        }

        result = await graph.ainvoke(initial_state)

        # Miss path → large_planner generates + template_gen writes
        assert result["cache_hit"] is False  # cache_lookup says miss
        # large_planner generates a plan
        assert result["current_plan"] is not None
        # template_gen runs after large_planner
        assert result["plan_template"] is not None
        # is_complete set by large_planner
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_full_miss_persists_template(self, default_cfg, fake_redis):
        """template_gen should write template to Redis during miss flow."""
        prefix = default_cfg.key_prefix
        graph = build_graph(default_cfg, fake_redis)

        initial_state: dict = {
            "query": "Analyze year-over-year growth",
            "context": {"income_statement": {"revenue": 5000}},
            "agent_id": "growth_agent",
            "tools": [MockTool("calculator")],
            "tools_hash": "growth_hash",
            "keyword": None,
            "cache_hit": False,
            "cache_hit_layer": "",
            "plan_template": None,
            "current_plan": None,
            "actor_responses": [],
            "execution_log": [],
            "iteration_count": 0,
            "final_output": None,
            "is_complete": False,
        }

        await graph.ainvoke(initial_state)

        # Check that Redis now has a template
        tpl_keys = await fake_redis.keys(f"{prefix}:tpl:*")
        assert len(tpl_keys) >= 1, "template_gen should write at least one template"


class TestGraphL1HitPath:
    """cache_lookup → L1 HIT → small_planner → END"""

    @pytest.mark.asyncio
    async def test_l1_hit_flow(self, default_cfg, fake_redis):
        """Pre-seed an L1 key → the graph goes through small_planner."""
        import json
        import time

        prefix = default_cfg.key_prefix
        query = "Calculate current ratio"
        agent_id = "finance_agent"
        tools_hash = "graph_l1_test"

        ctx = {"balance_sheet": {"current_assets": 1000, "current_liabilities": 500}}
        tools = [MockTool("calculator")]

        # Build keys
        ctx_fp = compute_fingerprint(ctx, tools, agent_id)
        ts = task_sig(query, agent_id, tools_hash)
        cfp_hash = ctx_fp_hash(ctx_fp)

        # Create and seed template
        tpl_id = "graph-l1-hit"
        tpl_data = {
            "template_id": tpl_id,
            "version": "v2.3",
            "schema_hash": tools_hash,
            "ctx_fingerprint": json.dumps(ctx_fp.to_dict()),
            "task": "Calculate current ratio",
            "steps": json.dumps([
                {"description": "Get assets"},
                {"description": "Get liabilities"},
                {"description": "Divide"},
            ]),
            "created_at": str(time.time()),
            "ttl_seconds": "86400",
        }
        await fake_redis.hset(f"{prefix}:tpl:{tpl_id}", mapping=tpl_data)
        l1_key = f"{prefix}:l1:{agent_id}:{ts}:{cfp_hash}"
        await fake_redis.set(l1_key, tpl_id)

        graph = build_graph(default_cfg, fake_redis)

        initial_state: dict = {
            "query": query,
            "context": ctx,
            "agent_id": agent_id,
            "tools": tools,
            "tools_hash": tools_hash,
            "keyword": None,
            "cache_hit": False,
            "cache_hit_layer": "",
            "plan_template": None,
            "current_plan": None,
            "actor_responses": [],
            "execution_log": [],
            "iteration_count": 0,
            "final_output": None,
            "is_complete": False,
        }

        result = await graph.ainvoke(initial_state)

        assert result["cache_hit"] is True
        assert result["cache_hit_layer"] == "L1"
        assert result["plan_template"] is not None
        assert result["is_complete"] is True
        # small_planner should adapt the template
        assert "adapted from template" in result.get("current_plan", "")


class TestGraphKwCachePath:
    """cache_lookup → kw_cache HIT → L2/L3 → small_planner"""

    @pytest.mark.asyncio
    async def test_kw_cache_hit_l2_l3_flow(self, default_cfg, fake_redis):
        """Seed kw_cache + tpl_idx → L2/L3 hit path."""
        import json
        import time

        prefix = default_cfg.key_prefix
        query = "Calculate working capital from balance sheet"
        keyword = "working_capital_ratio"
        tools_hash = "graph_kw_test"

        # Seed kw_cache
        qh = qhash(query)
        await fake_redis.setex(f"{prefix}:kw_cache:{qh}", 3600, keyword)

        # Seed a matching template
        from apc_cache.keyword.types import CtxFingerprint
        ctx_fp = CtxFingerprint(
            context_type="financial_report",
            length_bucket="short",
            tools=frozenset(["calculator"]),
            agent_role="finance_agent",
            context_schema=frozenset(["balance_sheet"]),
        )
        tpl_id = "graph-kw-cache-hit"
        tpl_data = {
            "template_id": tpl_id,
            "version": "v2.3",
            "schema_hash": tools_hash,
            "ctx_fingerprint": json.dumps(ctx_fp.to_dict()),
            "task": "Working capital analysis",
            "steps": json.dumps([{"description": "Fetch"}, {"description": "Compute"}]),
            "created_at": str(time.time()),
            "ttl_seconds": "86400",
        }
        await fake_redis.hset(f"{prefix}:tpl:{tpl_id}", mapping=tpl_data)
        await fake_redis.sadd(f"{prefix}:tpl_idx:finance_agent:{keyword}", tpl_id)

        graph = build_graph(default_cfg, fake_redis)

        initial_state: dict = {
            "query": query,
            "context": {"balance_sheet": {"current_assets": 1000, "current_liabilities": 500}},
            "agent_id": "finance_agent",
            "tools": [MockTool("calculator")],
            "tools_hash": tools_hash,
            "keyword": None,
            "cache_hit": False,
            "cache_hit_layer": "",
            "plan_template": None,
            "current_plan": None,
            "actor_responses": [],
            "execution_log": [],
            "iteration_count": 0,
            "final_output": None,
            "is_complete": False,
        }

        result = await graph.ainvoke(initial_state)

        assert result["cache_hit"] is True
        assert result["cache_hit_layer"] == "L2_L3"
        assert result["keyword"] == keyword
        assert result["plan_template"] is not None
