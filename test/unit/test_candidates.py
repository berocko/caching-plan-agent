"""Tests for apc_cache.keyword.candidates — shortcut decision logic.

Blueprint §4.5: build_candidates with DIST_HIGH=0.85, DIST_LOW=0.50.
"""

import numpy as np
import pytest

from apc_cache.config import APCConfig
from apc_cache.keyword.candidates import build_candidates
from apc_cache.keyword.types import CandidateResult


class TestBuildCandidates:
    @pytest.fixture
    def cfg(self) -> APCConfig:
        return APCConfig(
            candidate_dist_high=0.85,
            candidate_dist_low=0.50,
            candidate_top_k=5,
        )

    @pytest.fixture
    def query_vec(self) -> np.ndarray:
        return np.array([0.1] * 384, dtype=np.float32)

    def test_empty_index_shortcut_new(self, cfg, query_vec, mock_kw_index):
        mock_kw_index.search.return_value = []
        result = build_candidates(query_vec, mock_kw_index, cfg)
        assert result.action == "shortcut_new"
        assert result.items == []

    def test_all_below_low_threshold_shortcut_new(self, cfg, query_vec, mock_kw_index):
        mock_kw_index.search.return_value = [
            ("kw1", 0.49),
            ("kw2", 0.30),
        ]
        result = build_candidates(query_vec, mock_kw_index, cfg)
        assert result.action == "shortcut_new"
        assert result.items == []

    def test_single_strong_candidate_reuse(self, cfg, query_vec, mock_kw_index):
        mock_kw_index.search.return_value = [("working_capital", 0.92)]
        result = build_candidates(query_vec, mock_kw_index, cfg)
        assert result.action == "shortcut_reuse"
        assert result.items == [("working_capital", 0.92)]

    def test_two_candidates_clear_leader_reuse(self, cfg, query_vec, mock_kw_index):
        mock_kw_index.search.return_value = [
            ("working_capital", 0.91),
            ("revenue_growth", 0.70),
        ]
        result = build_candidates(query_vec, mock_kw_index, cfg)
        assert result.action == "shortcut_reuse"
        assert len(result.items) == 1
        assert result.items[0][0] == "working_capital"

    def test_leader_not_enough_gap_ask_llm(self, cfg, query_vec, mock_kw_index):
        """Top score ≥ 0.85 but gap ≤ 0.15 → ask_llm."""
        mock_kw_index.search.return_value = [
            ("kw1", 0.88),
            ("kw2", 0.75),
        ]
        result = build_candidates(query_vec, mock_kw_index, cfg)
        assert result.action == "ask_llm"

    def test_leader_not_high_enough_ask_llm(self, cfg, query_vec, mock_kw_index):
        """Top score < 0.85 → ask_llm even with big gap."""
        mock_kw_index.search.return_value = [
            ("kw1", 0.80),
            ("kw2", 0.40),
        ]
        result = build_candidates(query_vec, mock_kw_index, cfg)
        assert result.action == "ask_llm"

    def test_returns_at_most_top_3_for_ask_llm(self, cfg, query_vec, mock_kw_index):
        mock_kw_index.search.return_value = [
            ("kw1", 0.80),
            ("kw2", 0.75),
            ("kw3", 0.70),
            ("kw4", 0.65),
            ("kw5", 0.60),
        ]
        result = build_candidates(query_vec, mock_kw_index, cfg)
        assert result.action == "ask_llm"
        assert len(result.items) == 3

    def test_filtered_by_dist_low(self, cfg, query_vec, mock_kw_index):
        """Only candidates ≥ DIST_LOW (0.50) are kept."""
        mock_kw_index.search.return_value = [
            ("kw1", 0.90),
            ("kw2", 0.48),  # below threshold
            ("kw3", 0.30),  # below threshold
        ]
        result = build_candidates(query_vec, mock_kw_index, cfg)
        assert result.items == [("kw1", 0.90)]
        assert result.action == "shortcut_reuse"

    def test_custom_thresholds(self, query_vec, mock_kw_index):
        cfg = APCConfig(candidate_dist_high=0.90, candidate_dist_low=0.60)
        mock_kw_index.search.return_value = [
            ("kw1", 0.88),  # < 0.90 → not high enough for single reuse
            ("kw2", 0.55),  # < 0.60 → filtered out
        ]
        result = build_candidates(query_vec, mock_kw_index, cfg)
        # kw2 filtered out, kw1 is alone but < 0.90 → ask_llm
        assert result.action == "ask_llm"
        assert len(result.items) == 1

    def test_edge_case_exactly_at_high_threshold(self, cfg, query_vec, mock_kw_index):
        mock_kw_index.search.return_value = [("kw1", 0.85)]
        result = build_candidates(query_vec, mock_kw_index, cfg)
        assert result.action == "shortcut_reuse"

    def test_edge_case_exactly_at_low_threshold(self, cfg, query_vec, mock_kw_index):
        mock_kw_index.search.return_value = [("kw1", 0.50), ("kw2", 0.70)]
        result = build_candidates(query_vec, mock_kw_index, cfg)
        # kw1 is exactly at low threshold → included; gap not checked since kw1 < high
        assert result.action == "ask_llm"
