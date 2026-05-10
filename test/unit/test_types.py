"""Tests for apc_cache.keyword.types — CtxFingerprint, PlanTemplate,
CandidateResult, adjacent_buckets.
"""

import json
import time

import pytest

from apc_cache.config import CURRENT_VERSION
from apc_cache.keyword.types import CandidateResult, CtxFingerprint, PlanTemplate, adjacent_buckets


class TestCtxFingerprint:
    def test_default_values(self):
        fp = CtxFingerprint("unknown", "short", frozenset(), "agent")
        assert fp.context_type == "unknown"
        assert fp.length_bucket == "short"
        assert fp.tools == frozenset()
        assert fp.agent_role == "agent"
        assert fp.context_schema == frozenset()
        assert fp.query_lang == "en"

    def test_to_dict_sort_keys(self):
        fp = CtxFingerprint(
            "financial_report",
            "short",
            frozenset(["calc", "fetch"]),
            "finance_agent",
            context_schema=frozenset(["a", "b"]),
        )
        d = fp.to_dict(sort_keys=True)
        # Keys should be alphabetically sorted
        keys = list(d.keys())
        assert keys == sorted(keys)
        assert d["tools"] == ["calc", "fetch"]

    def test_to_dict_includes_query_lang_when_not_en(self):
        fp = CtxFingerprint("x", "short", frozenset(), "agent", query_lang="zh")
        d = fp.to_dict()
        assert d["query_lang"] == "zh"

    def test_to_dict_omits_query_lang_when_en(self):
        fp = CtxFingerprint("x", "short", frozenset(), "agent", query_lang="en")
        d = fp.to_dict()
        assert "query_lang" not in d

    def test_roundtrip_via_json(self):
        fp = CtxFingerprint(
            "financial_report",
            "medium",
            frozenset(["a", "b"]),
            "agent",
            context_schema=frozenset(["k1", "k2"]),
        )
        d = fp.to_dict()
        raw = json.dumps(d, sort_keys=True)
        back = json.loads(raw)
        assert back["context_type"] == "financial_report"
        assert back["length_bucket"] == "medium"
        assert set(back["tools"]) == {"a", "b"}
        assert set(back["context_schema"]) == {"k1", "k2"}


class TestCandidateResult:
    def test_empty_candidates(self):
        cr = CandidateResult(items=[], action="shortcut_new")
        assert cr.items == []
        assert cr.action == "shortcut_new"

    def test_with_candidates(self):
        items = [("kw1", 0.95), ("kw2", 0.80)]
        cr = CandidateResult(items=items, action="shortcut_reuse")
        assert cr.items == items
        assert cr.action == "shortcut_reuse"

    def test_ask_llm_action(self):
        cr = CandidateResult(items=[("kw", 0.70)], action="ask_llm")
        assert cr.action == "ask_llm"


class TestPlanTemplate:
    @pytest.fixture
    def ctx_fp(self):
        return CtxFingerprint(
            "financial_report", "short", frozenset(["calc"]), "agent",
            context_schema=frozenset(["bs"]),
        )

    @pytest.fixture
    def valid_template(self, ctx_fp):
        return PlanTemplate(
            template_id="test-id",
            version=CURRENT_VERSION,
            schema_hash="abc123",
            ctx_fingerprint=ctx_fp,
            task="Test task",
            steps=[{"description": "Step 1"}],
            created_at=time.time(),
            ttl_seconds=86400,
        )

    def test_is_valid_current(self, valid_template):
        assert valid_template.is_valid(tools_hash="abc123") is True

    def test_is_valid_wrong_version(self, valid_template):
        valid_template.version = "v1.0"
        assert valid_template.is_valid() is False

    def test_is_valid_wrong_tools_hash(self, valid_template):
        assert valid_template.is_valid(tools_hash="wrong_hash") is False

    def test_is_valid_empty_tools_hash_skipped(self, valid_template):
        """When tools_hash="" the schema_hash check is skipped."""
        valid_template.schema_hash = "whatever"
        assert valid_template.is_valid(tools_hash="") is True

    def test_is_valid_expired(self, valid_template):
        valid_template.created_at = time.time() - 100000
        valid_template.ttl_seconds = 60
        assert valid_template.is_valid() is False

    def test_to_redis_hash(self, valid_template):
        h = valid_template.to_redis_hash()
        assert h["template_id"] == "test-id"
        assert h["version"] == CURRENT_VERSION
        assert "ctx_fingerprint" in h
        assert "steps" in h

    def test_from_redis_hash_roundtrip(self, valid_template):
        """Serialize via to_redis_hash → wire encode → from_redis_hash."""
        h = valid_template.to_redis_hash()
        wire = {k.encode(): v.encode() for k, v in h.items()}
        back = PlanTemplate.from_redis_hash(wire)
        assert back.template_id == valid_template.template_id
        assert back.version == valid_template.version
        assert back.task == valid_template.task
        assert back.steps == valid_template.steps
        # ctx_fingerprint comparison
        assert back.ctx_fingerprint.context_type == valid_template.ctx_fingerprint.context_type
        assert back.ctx_fingerprint.tools == valid_template.ctx_fingerprint.tools

    def test_from_redis_hash_no_query_lang(self, valid_template):
        """Backwards compat: if query_lang is missing, default to 'en'."""
        h = valid_template.to_redis_hash()
        d = json.loads(h["ctx_fingerprint"])
        d.pop("query_lang", None)
        h["ctx_fingerprint"] = json.dumps(d)
        wire = {k.encode(): v.encode() for k, v in h.items()}
        back = PlanTemplate.from_redis_hash(wire)
        assert back.ctx_fingerprint.query_lang == "en"


class TestAdjacentBuckets:
    def test_adjacent_buckets_table(self):
        # All valid pairs
        assert adjacent_buckets("short", "short") is True
        assert adjacent_buckets("short", "medium") is True
        assert adjacent_buckets("medium", "long") is True
        # Invalid
        assert adjacent_buckets("short", "long") is False
        # Unknown
        assert adjacent_buckets("short", "huge") is False
