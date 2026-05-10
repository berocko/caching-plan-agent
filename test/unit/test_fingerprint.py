"""Tests for apc_cache.fingerprint — context classification, fingerprinting,
compatibility checking.
"""

import pytest

from apc_cache.fingerprint import (
    _bucket_length,
    _classify_context,
    _extract_schema,
    compute_fingerprint,
    ctx_compatible,
    ctx_fp_hash,
)
from apc_cache.keyword.types import CtxFingerprint, adjacent_buckets


# ── Helper: minimal tool mock ────────────────────────────────────


class MockTool:
    def __init__(self, name: str):
        self.name = name


# ══════════════════════════════════════════════════════════════════
# _classify_context
# ══════════════════════════════════════════════════════════════════


class TestClassifyContext:
    def test_financial_report_by_keys(self):
        ctx = {"balance_sheet": {}, "income_statement": {}}
        assert _classify_context(ctx) == "financial_report"

    def test_financial_report_case_insensitive(self):
        ctx = {"Balance_Sheet": {}, "Cash_Flow": {}}
        assert _classify_context(ctx) == "financial_report"

    def test_tabular_data_by_keys(self):
        ctx = {"rows": [], "columns": []}
        assert _classify_context(ctx) == "tabular_data"

    def test_generic_dict_is_structured_json(self):
        ctx = {"foo": "bar", "baz": 42}
        assert _classify_context(ctx) == "structured_json"

    def test_long_string_is_long_document(self):
        ctx = "x" * 10001
        assert _classify_context(ctx) == "long_document"

    def test_short_string_is_short_text(self):
        ctx = "hello world"
        assert _classify_context(ctx) == "short_text"

    def test_string_boundary_exactly_10000(self):
        ctx = "x" * 10000
        assert _classify_context(ctx) == "short_text"

    def test_list_of_dicts_is_tabular(self):
        ctx = [{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}]
        assert _classify_context(ctx) == "tabular_data"

    def test_empty_list_is_structured_json(self):
        assert _classify_context([]) == "structured_json"

    def test_none_is_unknown(self):
        assert _classify_context(None) == "unknown"

    def test_int_is_unknown(self):
        assert _classify_context(42) == "unknown"


# ══════════════════════════════════════════════════════════════════
# _bucket_length
# ══════════════════════════════════════════════════════════════════


class TestBucketLength:
    def test_short_text(self):
        assert _bucket_length("short") == "short"

    def test_medium_text(self):
        # ~2500 tokens = 10000 chars
        assert _bucket_length("x" * 4000) == "medium"

    def test_long_text(self):
        assert _bucket_length("x" * 50000) == "long"

    def test_short_dict(self):
        assert _bucket_length({"a": 1}) == "short"


# ══════════════════════════════════════════════════════════════════
# _extract_schema
# ══════════════════════════════════════════════════════════════════


class TestExtractSchema:
    def test_flat_dict(self):
        schema = _extract_schema({"name": "a", "age": 30})
        assert schema == frozenset(["name", "age"])

    def test_nested_two_levels(self):
        ctx = {
            "user": {"name": "Alice", "email": "a@b.com"},
            "order": {"id": 1},
        }
        schema = _extract_schema(ctx)
        assert "user.name" in schema
        assert "user.email" in schema
        assert "order.id" in schema

    def test_max_depth_respected(self):
        """Depth 3 keys should NOT appear."""
        ctx = {"a": {"b": {"c": 1}}}
        schema = _extract_schema(ctx)
        assert "a.b" in schema
        assert "a.b.c" not in schema

    def test_non_dict_returns_empty(self):
        assert _extract_schema("hello") == frozenset()
        assert _extract_schema(42) == frozenset()
        assert _extract_schema(None) == frozenset()


# ══════════════════════════════════════════════════════════════════
# compute_fingerprint
# ══════════════════════════════════════════════════════════════════


class TestComputeFingerprint:
    def test_full_fingerprint(self):
        tools = [MockTool("calculator"), MockTool("data_fetcher")]
        ctx = {"balance_sheet": {"assets": 1000}}
        fp = compute_fingerprint(ctx, tools, "finance_agent")
        assert fp.context_type == "financial_report"
        assert fp.length_bucket == "short"
        assert fp.tools == frozenset(["calculator", "data_fetcher"])
        assert fp.agent_role == "finance_agent"
        assert "balance_sheet" in fp.context_schema

    def test_default_query_lang(self):
        fp = compute_fingerprint("text", [], "agent")
        assert fp.query_lang == "en"

    def test_explicit_query_lang(self):
        fp = compute_fingerprint("text", [], "agent", query_lang="zh")
        assert fp.query_lang == "zh"


# ══════════════════════════════════════════════════════════════════
# ctx_fp_hash
# ══════════════════════════════════════════════════════════════════


class TestCtxFpHash:
    def test_deterministic(self):
        fp = CtxFingerprint("financial_report", "short", frozenset(["a"]), "agent")
        assert ctx_fp_hash(fp) == ctx_fp_hash(fp)

    def test_length(self):
        fp = CtxFingerprint("financial_report", "short", frozenset(), "agent")
        assert len(ctx_fp_hash(fp)) == 12

    def test_different_schema_different_hash(self):
        fp1 = CtxFingerprint("x", "short", frozenset(["a"]), "r", frozenset(["k1"]))
        fp2 = CtxFingerprint("x", "short", frozenset(["a"]), "r", frozenset(["k2"]))
        assert ctx_fp_hash(fp1) != ctx_fp_hash(fp2)


# ══════════════════════════════════════════════════════════════════
# ctx_compatible — blueprint §4.3
# ══════════════════════════════════════════════════════════════════


class TestCtxCompatible:
    @pytest.fixture
    def base_fp(self):
        return CtxFingerprint(
            context_type="financial_report",
            length_bucket="short",
            tools=frozenset(["calculator"]),
            agent_role="finance_agent",
            context_schema=frozenset(["balance_sheet", "income_statement"]),
        )

    def test_exact_match(self, base_fp):
        assert ctx_compatible(base_fp, base_fp) is True

    def test_type_mismatch(self, base_fp):
        other = CtxFingerprint("tabular_data", "short", frozenset(["calculator"]), "finance_agent")
        assert ctx_compatible(base_fp, other) is False

    def test_tools_mismatch(self, base_fp):
        other = CtxFingerprint(
            "financial_report", "short", frozenset(["data_fetcher"]), "finance_agent"
        )
        assert ctx_compatible(base_fp, other) is False

    def test_agent_role_mismatch(self, base_fp):
        other = CtxFingerprint("financial_report", "short", frozenset(["calculator"]), "sales_agent")
        assert ctx_compatible(base_fp, other) is False

    def test_adjacent_bucket_medium(self, base_fp):
        """short ↔ medium is allowed."""
        other = CtxFingerprint(
            "financial_report", "medium", frozenset(["calculator"]), "finance_agent"
        )
        assert ctx_compatible(base_fp, other) is True

    def test_cross_bucket_short_long(self, base_fp):
        """short ↔ long is NOT allowed."""
        other = CtxFingerprint(
            "financial_report", "long", frozenset(["calculator"]), "finance_agent"
        )
        assert ctx_compatible(base_fp, other) is False

    def test_schema_subset_ok(self, base_fp):
        """tpl schema ⊆ query schema → compatible."""
        query_fp = CtxFingerprint(
            "financial_report",
            "short",
            frozenset(["calculator"]),
            "finance_agent",
            context_schema=frozenset(["balance_sheet", "income_statement", "cash_flow"]),
        )
        assert ctx_compatible(base_fp, query_fp) is True

    def test_schema_not_subset(self, base_fp):
        """tpl needs fields the query doesn't have → not compatible."""
        query_fp = CtxFingerprint(
            "financial_report",
            "short",
            frozenset(["calculator"]),
            "finance_agent",
            context_schema=frozenset(["balance_sheet"]),
        )
        assert ctx_compatible(base_fp, query_fp) is False

    def test_empty_template_schema_always_compatible(self):
        """If template has no schema requirement, any query is fine."""
        tpl_fp = CtxFingerprint(
            "short_text", "short", frozenset(["calc"]), "agent", context_schema=frozenset()
        )
        query_fp = CtxFingerprint(
            "short_text", "short", frozenset(["calc"]), "agent", context_schema=frozenset()
        )
        assert ctx_compatible(tpl_fp, query_fp) is True


# ══════════════════════════════════════════════════════════════════
# adjacent_buckets
# ══════════════════════════════════════════════════════════════════


class TestAdjacentBuckets:
    def test_same_bucket(self):
        assert adjacent_buckets("short", "short") is True

    def test_adjacent_short_medium(self):
        assert adjacent_buckets("short", "medium") is True

    def test_adjacent_medium_long(self):
        assert adjacent_buckets("medium", "long") is True

    def test_cross_short_long(self):
        assert adjacent_buckets("short", "long") is False

    def test_unknown_bucket(self):
        assert adjacent_buckets("short", "unknown_bucket") is False

    def test_symmetric(self):
        assert adjacent_buckets("medium", "short") == adjacent_buckets("short", "medium")
