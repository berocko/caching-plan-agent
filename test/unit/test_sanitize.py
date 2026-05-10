"""Tests for apc_cache.keyword.sanitize — keyword cleaning and entity detection."""

import pytest

from apc_cache.keyword.sanitize import contains_query_entities, sanitize_keyword


class TestSanitizeKeyword:
    def test_lowercase(self):
        assert sanitize_keyword("Working Capital Ratio") == "working capital ratio"

    def test_truncate_long_keyword(self):
        long_kw = "a" * 100
        result = sanitize_keyword(long_kw)
        assert result is not None
        assert len(result) <= 64

    def test_removes_punctuation(self):
        assert sanitize_keyword("hello, world!") == "hello world"

    def test_preserves_hyphens_and_underscores(self):
        assert sanitize_keyword("working-capital_ratio") == "working-capital_ratio"

    def test_short_keyword_rejected(self):
        """Keywords < 3 chars after cleaning → None."""
        assert sanitize_keyword("ab") is None
        assert sanitize_keyword("a") is None
        assert sanitize_keyword("") is None

    def test_empty_string(self):
        assert sanitize_keyword("") is None

    def test_whitespace_only(self):
        assert sanitize_keyword("   ") is None

    def test_merge_internal_whitespace(self):
        assert sanitize_keyword("hello    world") == "hello world"

    def test_strips_surrounding_whitespace(self):
        assert sanitize_keyword("  hello  ") == "hello"

    def test_numbers_preserved(self):
        assert sanitize_keyword("ratio 2023") == "ratio 2023"

    def test_mixed_special_chars(self):
        result = sanitize_keyword("Hello! @World #2023")
        # !, @, # removed; numbers stay
        assert result == "hello world 2023"

    def test_min_boundary_exactly_3(self):
        assert sanitize_keyword("abc") == "abc"

    def test_max_boundary_exactly_64(self):
        kw = "x" * 64
        assert sanitize_keyword(kw) == kw


class TestContainsQueryEntities:
    def test_year_in_keyword(self):
        assert contains_query_entities("report_2023", "Generate a report for 2023") is True

    def test_no_year_in_keyword(self):
        assert contains_query_entities("financial_report", "Generate a report for 2023") is False

    def test_number_in_keyword(self):
        assert contains_query_entities("id_12345", "Look up record with ID 12345") is True

    def test_proper_noun_in_keyword(self):
        assert contains_query_entities("apple_analysis", "Analyze Apple Inc financials") is True

    def test_simple_generic_keyword(self):
        assert contains_query_entities("working_capital", "Calculate working capital ratio") is False

    def test_short_number_not_matched(self):
        """Only ≥ 4-digit numbers trigger entity detection."""
        assert contains_query_entities("ratio_100", "Calculate ratio for item 100") is False

    def test_case_insensitive_entity_match(self):
        assert contains_query_entities("APPLE_REPORT", "analyze Apple Inc earnings") is True
