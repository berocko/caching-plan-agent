"""Integration tests for template_gen — full write-back to all cache layers.

Uses fakeredis to verify that template_gen writes to:
- apc:tpl:{tpl_id} (template body)
- apc:tpl_idx:{agent}:{keyword} (L2 index)
- apc:l1:{agent}:{task_sig}:{ctx_fp_hash} (L1 promotion)
- apc:kw_meta:{keyword} + apc:kw_timeline (keyword index)
"""

import time
from unittest.mock import MagicMock

import pytest

from apc_cache.cache.template_gen import template_gen
from apc_cache.config import APCConfig


class MockTool:
    def __init__(self, name: str):
        self.name = name


class TestTemplateGen:
    @pytest.mark.asyncio
    async def test_writes_to_all_layers(self, default_cfg, fake_redis):
        """A full template_gen run should write to L1, L2, tpl, and kw_meta."""
        prefix = default_cfg.key_prefix

        state: dict = {
            "query": "Calculate working capital ratio from Q3 report",
            "context": {"balance_sheet": {"assets": 1000}},
            "agent_id": "finance_agent",
            "tools": [MockTool("calculator")],
            "tools_hash": "abc123",
            "keyword": "working_capital_ratio",
            "execution_log": [
                {"task": "Calculate working capital ratio"},
                {"step": "Fetch balance sheet data"},
                {"step": "Compute current assets / current liabilities"},
                {"step": "Format and return result"},
            ],
        }

        result = await template_gen(state, default_cfg, fake_redis)

        # 1. Plan template should be in the state
        assert result["plan_template"] is not None
        tpl_id = result["plan_template"].template_id

        # 2. Template body written to Redis
        assert await fake_redis.exists(f"{prefix}:tpl:{tpl_id}")

        # 3. L2 index populated
        idx_members = await fake_redis.smembers(
            f"{prefix}:tpl_idx:finance_agent:working_capital_ratio"
        )
        member_ids = {m.decode() for m in idx_members}
        assert tpl_id in member_ids

        # 4. L1 promotion key exists
        from apc_cache.fingerprint import compute_fingerprint, ctx_fp_hash
        from apc_cache.normalize import task_sig

        ts = task_sig(state["query"], state["agent_id"], state["tools_hash"])
        ctx_fp = compute_fingerprint(state["context"], state["tools"], state["agent_id"])
        cfp_hash = ctx_fp_hash(ctx_fp)
        l1_key = f"{prefix}:l1:finance_agent:{ts}:{cfp_hash}"
        l1_value = await fake_redis.get(l1_key)
        assert l1_value is not None
        assert l1_value.decode() == tpl_id

        # 5. Keyword metadata written
        assert await fake_redis.hexists(f"{prefix}:kw_meta:working_capital_ratio", "model_ver")

    @pytest.mark.asyncio
    async def test_resolves_keyword_alias(self, default_cfg, fake_redis):
        """If a keyword alias exists, the canonical form is used for indexing."""
        prefix = default_cfg.key_prefix

        # Set up an alias
        await fake_redis.hset(
            f"{prefix}:kw_alias", "wcr", "working_capital_ratio"
        )

        state: dict = {
            "query": "Calculate WCR",
            "context": {"balance_sheet": {"assets": 1000}},
            "agent_id": "finance_agent",
            "tools": [MockTool("calculator")],
            "tools_hash": "abc123",
            "keyword": "wcr",  # alias
            "execution_log": [
                {"task": "Calculate working capital ratio"},
                {"step": "Fetch data"},
            ],
        }

        result = await template_gen(state, default_cfg, fake_redis)
        tpl_id = result["plan_template"].template_id

        # Should be indexed under canonical keyword, not alias
        idx_members = await fake_redis.smembers(
            f"{prefix}:tpl_idx:finance_agent:working_capital_ratio"
        )
        member_ids = {m.decode() for m in idx_members}
        assert tpl_id in member_ids

    @pytest.mark.asyncio
    async def test_does_not_duplicate_keyword_metadata(self, default_cfg, fake_redis):
        """If a keyword already exists in kw_meta, do not overwrite."""
        prefix = default_cfg.key_prefix

        # Pre-create kw_meta
        original_ts = str(time.time() - 1000)
        await fake_redis.hset(
            f"{prefix}:kw_meta:existing_kw",
            mapping={
                "model_ver": default_cfg.embed_model_ver,
                "dim": "384",
                "created_at": original_ts,
            },
        )

        state: dict = {
            "query": "test query",
            "context": None,
            "agent_id": "test_agent",
            "tools": [],
            "tools_hash": "",
            "keyword": "existing_kw",
            "execution_log": [{"task": "test"}],
        }

        await template_gen(state, default_cfg, fake_redis)

        # created_at should NOT have been overwritten
        meta = await fake_redis.hgetall(f"{prefix}:kw_meta:existing_kw")
        assert meta[b"created_at"].decode() == original_ts

    @pytest.mark.asyncio
    async def test_template_includes_ctx_fingerprint(self, default_cfg, fake_redis):
        """The generated template should contain the ctx_fingerprint in its metadata."""
        import json

        state: dict = {
            "query": "Analyze financial report",
            "context": {"balance_sheet": {"assets": 1000}, "income_statement": {"revenue": 500}},
            "agent_id": "finance_agent",
            "tools": [MockTool("calculator")],
            "tools_hash": "abc123",
            "keyword": "financial_analysis",
            "execution_log": [{"task": "Analyze"}, {"step": "Review"}],
        }

        result = await template_gen(state, default_cfg, fake_redis)
        tpl = result["plan_template"]

        assert tpl.ctx_fingerprint.context_type == "financial_report"
        assert tpl.ctx_fingerprint.tools == frozenset(["calculator"])
        assert tpl.ctx_fingerprint.agent_role == "finance_agent"

        # Verify it's stored and retrievable
        prefix = default_cfg.key_prefix
        tpl_data = await fake_redis.hgetall(f"{prefix}:tpl:{tpl.template_id}")
        from apc_cache.keyword.types import PlanTemplate
        restored = PlanTemplate.from_redis_hash(tpl_data)
        assert restored.ctx_fingerprint.context_type == "financial_report"
