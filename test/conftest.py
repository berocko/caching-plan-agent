"""Shared fixtures for all test layers.

Strategy:
  - Unit tests: pure pytest, no fixtures needed beyond what each module defines.
  - Integration tests: fakeredis (in-memory Redis) to test Redis-dependent modules
    without a real Redis instance.
  - E2E tests: FastAPI TestClient + fakeredis for full request flow.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
import redis.asyncio as redis
from fakeredis import aioredis as fake_aioredis

from apc_cache.config import APCConfig

# ── Ensure tests never talk to real Redis ────────────────────────

os.environ.setdefault("APC_REDIS_URL", "redis://localhost:6379/0")


# ══════════════════════════════════════════════════════════════════
# Config fixtures
# ══════════════════════════════════════════════════════════════════


@pytest.fixture
def default_cfg() -> APCConfig:
    """Return an APCConfig with all defaults."""
    return APCConfig()


@pytest.fixture
def small_cache_cfg() -> APCConfig:
    """Config with small cache for eviction testing."""
    return APCConfig(cache_max_size=3, max_candidates_to_check=5)


@pytest.fixture
def drift_sensitive_cfg() -> APCConfig:
    """Config with tight drift thresholds."""
    return APCConfig(
        drift_sample_rate=0.05,
        drift_low_threshold=0.05,
        drift_high_threshold=0.15,
        kw_cache_ttl_min=60,
        kw_cache_ttl_max=300,
    )


# ══════════════════════════════════════════════════════════════════
# Redis fixtures — fakeredis (integration + e2e)
# ══════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def fake_redis() -> AsyncGenerator[redis.Redis, None]:
    """Return an in-memory Redis client backed by fakeredis.

    Each test gets a clean database.
    """
    r = fake_aioredis.FakeRedis(decode_responses=False)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def seeded_redis(
    fake_redis: redis.Redis, default_cfg: APCConfig
) -> AsyncGenerator[redis.Redis, None]:
    """A fakeredis pre-seeded with templates and keyword metadata.

    Seeds:
      - 2 keywords: "working_capital_ratio", "revenue_growth"
      - 3 templates across keywords
      - L1 keys for one template
    """
    import json
    import time

    prefix = default_cfg.key_prefix
    agent = "finance_agent"

    # Template 1: working_capital_ratio, financial_report context
    import hashlib
    from apc_cache.keyword.types import CtxFingerprint

    ctx_fp = CtxFingerprint(
        context_type="financial_report",
        length_bucket="short",
        tools=frozenset(["calculator"]),
        agent_role="finance_agent",
        context_schema=frozenset(["balance_sheet", "income_statement"]),
    )

    tpl1_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    tpl1_data = {
        "template_id": tpl1_id,
        "version": "v2.3",
        "schema_hash": "abc123",
        "ctx_fingerprint": json.dumps(ctx_fp.to_dict()),
        "task": "Calculate working capital ratio",
        "steps": json.dumps([
            {"description": "Fetch balance sheet"},
            {"description": "Compute current ratio"},
        ]),
        "created_at": str(time.time() - 100),
        "ttl_seconds": "86400",
    }
    await fake_redis.hset(f"{prefix}:tpl:{tpl1_id}", mapping=tpl1_data)
    await fake_redis.sadd(f"{prefix}:tpl_idx:{agent}:working_capital_ratio", tpl1_id)

    # Template 2: revenue_growth, same tools
    tpl2_id = "11111111-2222-3333-4444-555555555555"
    ctx_fp2 = CtxFingerprint(
        context_type="financial_report",
        length_bucket="medium",
        tools=frozenset(["calculator"]),
        agent_role="finance_agent",
        context_schema=frozenset(["revenue", "expenses"]),
    )
    tpl2_data = {
        "template_id": tpl2_id,
        "version": "v2.3",
        "schema_hash": "abc123",
        "ctx_fingerprint": json.dumps(ctx_fp2.to_dict()),
        "task": "Analyze revenue growth",
        "steps": json.dumps([
            {"description": "Fetch revenue data"},
            {"description": "Calculate YoY growth"},
        ]),
        "created_at": str(time.time() - 200),
        "ttl_seconds": "86400",
    }
    await fake_redis.hset(f"{prefix}:tpl:{tpl2_id}", mapping=tpl2_data)
    await fake_redis.sadd(f"{prefix}:tpl_idx:{agent}:revenue_growth", tpl2_id)

    # Template 3: also under working_capital_ratio but different ctx
    tpl3_id = "22222222-3333-4444-5555-666666666666"
    ctx_fp3 = CtxFingerprint(
        context_type="tabular_data",
        length_bucket="short",
        tools=frozenset(["spreadsheet_parser"]),
        agent_role="finance_agent",
        context_schema=frozenset(["rows", "columns"]),
    )
    tpl3_data = {
        "template_id": tpl3_id,
        "version": "v2.3",
        "schema_hash": "def456",
        "ctx_fingerprint": json.dumps(ctx_fp3.to_dict()),
        "task": "Working capital from spreadsheet",
        "steps": json.dumps([
            {"description": "Parse spreadsheet"},
            {"description": "Extract WC components"},
        ]),
        "created_at": str(time.time() - 50),
        "ttl_seconds": "86400",
    }
    await fake_redis.hset(f"{prefix}:tpl:{tpl3_id}", mapping=tpl3_data)
    await fake_redis.sadd(f"{prefix}:tpl_idx:{agent}:working_capital_ratio", tpl3_id)

    # Keyword metadata (without embeddings for Phase 1)
    await fake_redis.hset(
        f"{prefix}:kw_meta:working_capital_ratio",
        mapping={"model_ver": "v1", "dim": "384", "created_at": str(time.time() - 100)},
    )
    await fake_redis.zadd(
        f"{prefix}:kw_timeline",
        {"working_capital_ratio": time.time() - 100},
    )
    await fake_redis.hset(
        f"{prefix}:kw_meta:revenue_growth",
        mapping={"model_ver": "v1", "dim": "384", "created_at": str(time.time() - 200)},
    )
    await fake_redis.zadd(
        f"{prefix}:kw_timeline",
        {"revenue_growth": time.time() - 200},
    )

    yield fake_redis


# ══════════════════════════════════════════════════════════════════
# Mock LLM fixture (for llm_normalize tests)
# ══════════════════════════════════════════════════════════════════


class MockLLMResponse:
    """Simulates a LangChain AIMessage."""
    def __init__(self, text: str) -> None:
        self.content = text


@pytest.fixture
def mock_llm():
    """Return a mock LLM callable for use in llm_normalize tests.

    Usage in tests:
        llm = mock_llm
        llm.ainvoke.return_value = MockLLMResponse("working_capital_ratio")
    """
    llm = MagicMock()
    return llm


# ══════════════════════════════════════════════════════════════════
# Mock KeywordIndexManager (for candidates tests)
# ══════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_kw_index():
    """Return a mock KeywordIndexManager with configurable search()."""
    index = MagicMock()
    index.search.return_value = []
    return index


# ══════════════════════════════════════════════════════════════════
# Helper functions used across test files
# ══════════════════════════════════════════════════════════════════


@pytest.fixture
def make_state():
    """Factory for building AgentState dicts with defaults."""
    def _make(**overrides: Any) -> dict[str, Any]:
        state: dict[str, Any] = {
            "query": "Calculate working capital ratio",
            "context": {"balance_sheet": {"assets": 1000}, "income_statement": {"revenue": 500}},
            "agent_id": "finance_agent",
            "tools": [MagicMock(name="calculator")],
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
        # Fix tool names for the mock
        state["tools"][0].name = "calculator"
        state.update(overrides)
        return state
    return _make


@pytest.fixture
def ctx_fp_finance():
    """Pre-built CtxFingerprint for financial_report / short / calculator."""
    from apc_cache.keyword.types import CtxFingerprint
    return CtxFingerprint(
        context_type="financial_report",
        length_bucket="short",
        tools=frozenset(["calculator"]),
        agent_role="finance_agent",
        context_schema=frozenset(["balance_sheet", "income_statement"]),
    )
