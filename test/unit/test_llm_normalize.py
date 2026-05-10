"""Tests for apc_cache.keyword.llm_normalize — keyword normalization and fallback chain.

Blueprint §4.6: LLM normalization with 4-layer fallback chain.
Tests the pure helper functions directly; normalize_keyword is async-tested in integration.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from apc_cache.config import APCConfig
from apc_cache.keyword.llm_normalize import (
    _detect_intent,
    _fallback_extract_keyword,
    _format_candidates,
    _parse_llm_keyword,
    normalize_keyword,
)
from apc_cache.keyword.sanitize import sanitize_keyword


class MockAIMessage:
    def __init__(self, content: str):
        self.content = content


# ══════════════════════════════════════════════════════════════════
# _format_candidates
# ══════════════════════════════════════════════════════════════════


class TestFormatCandidates:
    def test_normal_list(self):
        candidates = [("kw1", 0.95), ("kw2", 0.80)]
        text = _format_candidates(candidates)
        assert "1. kw1  (similarity: 0.950)" in text
        assert "2. kw2  (similarity: 0.800)" in text

    def test_empty_list(self):
        assert "(no existing candidates)" in _format_candidates([])


# ══════════════════════════════════════════════════════════════════
# _parse_llm_keyword
# ══════════════════════════════════════════════════════════════════


class TestParseLLMKeyword:
    def test_simple_string(self):
        assert _parse_llm_keyword("working_capital_ratio") == "working_capital_ratio"

    def test_ai_message(self):
        msg = MockAIMessage("working_capital_ratio")
        assert _parse_llm_keyword(msg) == "working_capital_ratio"

    def test_strips_quotes(self):
        assert _parse_llm_keyword('"working_capital_ratio"') == "working_capital_ratio"
        assert _parse_llm_keyword("'revenue_growth'") == "revenue_growth"
        assert _parse_llm_keyword("`current_ratio`") == "current_ratio"

    def test_takes_first_line_only(self):
        text = "working_capital_ratio\nThis keyword was chosen because..."
        assert _parse_llm_keyword(text) == "working_capital_ratio"

    def test_strips_whitespace(self):
        assert _parse_llm_keyword("  hello world  ") == "hello world"


# ══════════════════════════════════════════════════════════════════
# _detect_intent
# ══════════════════════════════════════════════════════════════════


class TestDetectIntent:
    @pytest.fixture
    def candidate_set(self):
        return {"working_capital_ratio", "revenue_growth", "current_ratio"}

    def test_exact_match_is_reuse(self, candidate_set):
        msg = MockAIMessage("working_capital_ratio")
        assert _detect_intent(msg, candidate_set) == "reuse"

    def test_string_match_is_reuse(self, candidate_set):
        assert _detect_intent("working_capital_ratio", candidate_set) == "reuse"

    def test_new_keyword_is_new(self, candidate_set):
        msg = MockAIMessage("profit_margin_analysis")
        assert _detect_intent(msg, candidate_set) == "new"

    def test_reuse_hint_in_text(self, candidate_set):
        msg = MockAIMessage("I will reuse the existing keyword: working_capital_ratio")
        assert _detect_intent(msg, candidate_set) == "reuse"

    def test_match_hint_in_text(self, candidate_set):
        msg = MockAIMessage("This query matches existing keywords")
        assert _detect_intent(msg, candidate_set) == "reuse"


# ══════════════════════════════════════════════════════════════════
# _fallback_extract_keyword
# ══════════════════════════════════════════════════════════════════


class TestFallbackExtractKeyword:
    def test_simple_query(self):
        result = _fallback_extract_keyword("Calculate working capital ratio")
        assert sanitize_keyword(result) is not None
        assert "calculate" not in result.lower()  # stopword removed

    def test_all_stopwords_returns_default(self):
        result = _fallback_extract_keyword("the is a for in")
        assert result == "general_query"

    def test_empty_query(self):
        result = _fallback_extract_keyword("")
        assert result == "general_query"

    def test_truncates_to_8_words(self):
        """Max 8 words in fallback keyword."""
        long_query = "calculate the total revenue growth margin for the last fiscal year in north america region"
        result = _fallback_extract_keyword(long_query)
        assert len(result.split()) <= 8

    def test_with_punctuation(self):
        result = _fallback_extract_keyword(
            "Calculate the working-capital ratio for Q3 2024!"
        )
        assert sanitize_keyword(result) is not None


# ══════════════════════════════════════════════════════════════════
# normalize_keyword — async tests
# ══════════════════════════════════════════════════════════════════


class TestNormalizeKeywordAsync:
    @pytest.fixture
    def cfg(self) -> APCConfig:
        return APCConfig(
            normalize_llm_timeout=2.0,
            max_retry_on_entity=2,
        )

    @pytest.mark.asyncio
    async def test_llm_reuses_candidate(self, cfg, mock_llm):
        candidates = [("working_capital_ratio", 0.92), ("revenue_growth", 0.70)]
        mock_llm.ainvoke.return_value = MockAIMessage("working_capital_ratio")

        result = await normalize_keyword(
            query="Calculate working capital ratio",
            candidates=candidates,
            llm=mock_llm,
            cfg=cfg,
        )
        assert result == "working_capital_ratio"

    @pytest.mark.asyncio
    async def test_llm_creates_new_keyword(self, cfg, mock_llm):
        candidates = [("revenue_growth", 0.60)]
        mock_llm.ainvoke.return_value = MockAIMessage("profit_margin_analysis")

        result = await normalize_keyword(
            query="Analyze profit margins for Q3",
            candidates=candidates,
            llm=mock_llm,
            cfg=cfg,
        )
        assert sanitize_keyword(result) is not None

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_top_candidate(self, cfg, mock_llm):
        candidates = [("working_capital_ratio", 0.80)]
        mock_llm.ainvoke.side_effect = asyncio.TimeoutError()

        result = await normalize_keyword(
            query="Calculate working capital",
            candidates=candidates,
            llm=mock_llm,
            cfg=cfg,
        )
        assert result == "working_capital_ratio"

    @pytest.mark.asyncio
    async def test_timeout_no_good_candidate(self, cfg, mock_llm):
        candidates = [("irrelevant_kw", 0.40)]
        mock_llm.ainvoke.side_effect = asyncio.TimeoutError()

        result = await normalize_keyword(
            query="Calculate working capital ratio",
            candidates=candidates,
            llm=mock_llm,
            cfg=cfg,
        )
        # Falls back to fallback_extract_keyword
        assert result is not None
        assert sanitize_keyword(result) is not None

    @pytest.mark.asyncio
    async def test_hallucination_detected(self, cfg, mock_llm):
        """LLM claims to reuse but returns a keyword not in the candidate list."""
        candidates = [("working_capital_ratio", 0.92)]
        # LLM says "reuse the existing keyword" but returns something else
        mock_llm.ainvoke.return_value = MockAIMessage(
            "I'll reuse the existing keyword: revenue_growth"
        )

        result = await normalize_keyword(
            query="Calculate working capital",
            candidates=candidates,
            llm=mock_llm,
            cfg=cfg,
        )
        # Falls back to fallback
        assert sanitize_keyword(result) is not None

    @pytest.mark.asyncio
    async def test_entity_injection_retry(self, cfg, mock_llm):
        """Keyword contains a proper noun from the query → retry."""
        candidates = [("financial_analysis", 0.70)]
        # First response contains entity
        mock_llm.ainvoke.side_effect = [
            MockAIMessage("apple_analysis"),  # entity injection
            MockAIMessage("financial_analysis"),  # cleaner retry
        ]

        result = await normalize_keyword(
            query="Analyze Apple Inc financials",
            candidates=candidates,
            llm=mock_llm,
            cfg=cfg,
        )
        assert "apple" not in result.lower()
