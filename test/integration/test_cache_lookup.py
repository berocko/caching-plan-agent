"""Integration tests for cache_lookup — L1, kw_cache, and L2/L3 paths.

Uses fakeredis seeded with templates and keyword metadata.
"""

from unittest.mock import MagicMock

import pytest

from apc_cache.cache.lookup import cache_lookup, lookup_with_keyword
from apc_cache.config import APCConfig
from apc_cache.keyword.types import CtxFingerprint
from apc_cache.normalize import qhash, task_sig


# ── Helper: build a minimal Tool mock ────────────────────────────


class MockTool:
    def __init__(self, name: str):
        self.name = name


# ══════════════════════════════════════════════════════════════════
# L1 exact match tests
# ══════════════════════════════════════════════════════════════════


class TestL1ExactMatch:
    @pytest.mark.asyncio
    async def test_l1_hit_with_pre_seeded_key(self, default_cfg, fake_redis):
        """Pre-populate an L1 key pointing to an existing template."""
        from apc_cache.fingerprint import compute_fingerprint, ctx_fp_hash

        agent = "finance_agent"
        tools = [MockTool("calculator")]
        tools_hash = "abc123"
        query = "Calculate working capital ratio"
        ctx = {"balance_sheet": {"assets": 1000}, "income_statement": {"revenue": 500}}

        ts = task_sig(query, agent, tools_hash)
        ctx_fp = compute_fingerprint(ctx, tools, agent)
        cfp_hash = ctx_fp_hash(ctx_fp)
        l1_key = f"{default_cfg.key_prefix}:l1:{agent}:{ts}:{cfp_hash}"

        # Point L1 key to the template created in seeded_redis
        await fake_redis.set(l1_key, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

        state = {
            "query": query,
            "context": ctx,
            "agent_id": agent,
            "tools": tools,
            "tools_hash": tools_hash,
        }

        result = await cache_lookup(state, default_cfg, fake_redis)
        assert result["cache_hit"] is True
        assert result["cache_hit_layer"] == "L1"
        assert result["plan_template"] is not None

    @pytest.mark.asyncio
    async def test_l1_miss_when_key_absent(self, default_cfg, fake_redis):
        state = {
            "query": "Brand new query never seen before",
            "context": None,
            "agent_id": "finance_agent",
            "tools": [],
            "tools_hash": "no_hash",
        }
        result = await cache_lookup(state, default_cfg, fake_redis)
        # L1 miss and kw_cache miss → cache miss
        assert result["cache_hit"] is False
        assert result["keyword"] is None


# ══════════════════════════════════════════════════════════════════
# kw_cache tests
# ══════════════════════════════════════════════════════════════════


class TestKwCache:
    @pytest.mark.asyncio
    async def test_kw_cache_hit_goes_to_l2_l3(self, default_cfg, seeded_redis):
        """Pre-populate kw_cache → should hit L2/L3 on seeded data."""
        query = "Calculate working capital ratio from financial report"
        qh = qhash(query)
        kw_cache_key = f"{default_cfg.key_prefix}:kw_cache:{qh}"
        await seeded_redis.setex(kw_cache_key, 3600, "working_capital_ratio")

        state = {
            "query": query,
            "context": {"balance_sheet": {"assets": 1000}, "income_statement": {"revenue": 500}},
            "agent_id": "finance_agent",
            "tools": [MockTool("calculator")],
            "tools_hash": "abc123",
        }

        result = await cache_lookup(state, default_cfg, seeded_redis)
        # Should find the working_capital_ratio template in L2/L3
        assert result["cache_hit"] is True
        assert result["cache_hit_layer"] == "L2_L3"
        assert result["keyword"] == "working_capital_ratio"


# ══════════════════════════════════════════════════════════════════
# L2/L3 search via lookup_with_keyword
# ══════════════════════════════════════════════════════════════════


class TestL2L3Search:
    @pytest.mark.asyncio
    async def test_hit_with_compatible_ctx(self, default_cfg, seeded_redis):
        """Keyword exists with a matching template for this ctx."""
        state = {
            "query": "Calculate working capital ratio",
            "context": {"balance_sheet": {"assets": 1000}, "income_statement": {"revenue": 500}},
            "agent_id": "finance_agent",
            "tools": [MockTool("calculator")],
            "tools_hash": "abc123",
        }
        result = await lookup_with_keyword(
            state, "working_capital_ratio", default_cfg, seeded_redis
        )
        assert result["cache_hit"] is True
        assert result["cache_hit_layer"] == "L2_L3"
        assert result["keyword"] == "working_capital_ratio"
        assert result["plan_template"] is not None

    @pytest.mark.asyncio
    async def test_miss_when_ctx_incompatible(self, default_cfg, seeded_redis):
        """Keyword exists but all templates have incompatible ctx_fp."""
        state = {
            "query": "Working capital from long document",
            "context": "x" * 50000,  # long_document
            "agent_id": "finance_agent",
            "tools": [MockTool("text_parser")],  # different tools
            "tools_hash": "xyz789",
        }
        result = await lookup_with_keyword(
            state, "working_capital_ratio", default_cfg, seeded_redis
        )
        # Should miss because ctx is different (tools don't match)
        assert result["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_miss_when_keyword_not_in_index(self, default_cfg, fake_redis):
        """Keyword not in tpl_idx → cache miss."""
        state = {
            "query": "brand new concept query",
            "context": None,
            "agent_id": "finance_agent",
            "tools": [MockTool("calculator")],
            "tools_hash": "abc123",
        }
        result = await lookup_with_keyword(
            state, "nonexistent_keyword", default_cfg, fake_redis
        )
        assert result["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_l1_promotion_on_match(self, default_cfg, seeded_redis):
        """A successful L2/L3 match should promote the result to L1."""
        state = {
            "query": "Calculate working capital ratio",
            "context": {"balance_sheet": {"assets": 1000}, "income_statement": {"revenue": 500}},
            "agent_id": "finance_agent",
            "tools": [MockTool("calculator")],
            "tools_hash": "abc123",
        }
        await lookup_with_keyword(state, "working_capital_ratio", default_cfg, seeded_redis)

        # The L1 key should now exist
        from apc_cache.fingerprint import compute_fingerprint, ctx_fp_hash
        from apc_cache.normalize import task_sig

        ts = task_sig(state["query"], state["agent_id"], state["tools_hash"])
        ctx_fp = compute_fingerprint(state["context"], state["tools"], state["agent_id"])
        cfp_hash = ctx_fp_hash(ctx_fp)
        l1_key = f"{default_cfg.key_prefix}:l1:{state['agent_id']}:{ts}:{cfp_hash}"
        l1_value = await seeded_redis.get(l1_key)
        assert l1_value is not None
