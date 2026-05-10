"""Tests for template_gen internal helpers — _rule_filter, _classify_ttl,
and sanitize_keyword integration.
"""

import pytest

from apc_cache.cache.template_gen import _classify_ttl, _rule_filter


class TestRuleFilter:
    def test_empty_log(self):
        result = _rule_filter([])
        assert result == {"task": "", "steps": []}

    def test_task_extraction(self):
        log = [{"task": "Calculate working capital ratio"}]
        result = _rule_filter(log)
        assert result["task"] == "Calculate working capital ratio"

    def test_step_extraction(self):
        log = [
            {"step": "Fetch data"},
            {"step": "Compute ratio"},
        ]
        result = _rule_filter(log)
        assert len(result["steps"]) == 2
        assert result["steps"][0]["description"] == "Fetch data"
        assert result["steps"][1]["description"] == "Compute ratio"

    def test_action_extraction(self):
        log = [
            {"action": "Fetch balance sheet"},
            {"action": "Extract WC components"},
        ]
        result = _rule_filter(log)
        assert len(result["steps"]) == 2
        assert result["steps"][0]["description"] == "Fetch balance sheet"

    def test_mixed_entries(self):
        log = [
            {"task": "Analyze revenue"},
            {"step": "Step A"},
            {"action": "Action B"},
            "not a dict",  # should be skipped
            42,  # should be skipped
        ]
        result = _rule_filter(log)
        assert result["task"] == "Analyze revenue"
        assert len(result["steps"]) == 2  # step + action

    def test_non_dict_entries_skipped(self):
        log = ["string", 123, None]
        result = _rule_filter(log)
        assert result == {"task": "", "steps": []}


class TestClassifyTTL:
    def test_default_ttl(self):
        assert _classify_ttl("Calculate working capital ratio") == 86400

    def test_temporal_hint_today(self):
        assert _classify_ttl("Calculate today working capital") == 3600

    def test_temporal_hint_current(self):
        assert _classify_ttl("Current working capital ratio") == 3600

    def test_temporal_hint_now(self):
        assert _classify_ttl("Now calculate the ratio") == 3600

    def test_temporal_hint_latest(self):
        assert _classify_ttl("Latest financial report analysis") == 3600

    def test_temporal_hint_recent(self):
        assert _classify_ttl("Analyze recent transactions") == 3600

    def test_temporal_hint_this_week(self):
        assert _classify_ttl("This week report") == 3600

    def test_temporal_hint_this_month(self):
        assert _classify_ttl("This month financials") == 3600

    def test_year_in_query(self):
        assert _classify_ttl("Q3 2024 financial report") == 3600

    def test_year_boundary_1900(self):
        assert _classify_ttl("Historical 1999 data") == 3600

    def test_year_boundary_2099(self):
        assert _classify_ttl("Forecast 2099") == 3600

    def test_no_year_match_for_short_numbers(self):
        """Numbers like 202 (not a year) should not trigger temporal TTL."""
        assert _classify_ttl("Calculate ratio for item 202") == 86400

    def test_case_insensitive(self):
        assert _classify_ttl("TODAY financials") == 3600
