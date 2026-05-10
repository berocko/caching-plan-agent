"""Tests for apc_cache.keyword.kw_cache — drift tracking and adaptive TTL."""

import pytest

from apc_cache.config import APCConfig
from apc_cache.keyword.kw_cache import (
    _drift_window,
    _MAX_DRIFT_WINDOW,
    adaptive_ttl,
    compute_drift_rate,
    record_drift_result,
)


@pytest.fixture(autouse=True)
def clear_drift_window():
    """Each test starts with a clean drift window."""
    _drift_window.clear()
    yield
    _drift_window.clear()


class TestRecordDrift:
    def test_basic_record(self):
        record_drift_result(True)
        assert len(_drift_window) == 1
        assert _drift_window[0] is True

    def test_window_capped(self):
        for i in range(_MAX_DRIFT_WINDOW + 50):
            record_drift_result(i % 2 == 0)
        assert len(_drift_window) == _MAX_DRIFT_WINDOW

    def test_fifo_order(self):
        record_drift_result(True)
        record_drift_result(False)
        record_drift_result(True)
        for _ in range(_MAX_DRIFT_WINDOW - 1):
            record_drift_result(False)
        # First entry (True) should have been evicted
        assert _drift_window[0] is False  # the second entry is now first


class TestComputeDriftRate:
    def test_empty_window(self):
        assert compute_drift_rate() == 0.0

    def test_all_stable(self):
        for _ in range(10):
            record_drift_result(True)
        assert compute_drift_rate() == 0.0

    def test_all_drifted(self):
        for _ in range(10):
            record_drift_result(False)
        assert compute_drift_rate() == 1.0

    def test_half_drifted(self):
        for _ in range(5):
            record_drift_result(True)
        for _ in range(5):
            record_drift_result(False)
        assert compute_drift_rate() == 0.5


class TestAdaptiveTTL:
    @pytest.fixture
    def cfg(self) -> APCConfig:
        return APCConfig(
            kw_cache_ttl_min=300,
            kw_cache_ttl_max=3600,
            drift_low_threshold=0.05,
            drift_high_threshold=0.15,
        )

    def test_zero_drift_returns_max(self, cfg):
        """< 5% drift → max TTL."""
        for _ in range(95):
            record_drift_result(True)
        for _ in range(5):
            record_drift_result(False)  # 5% drift rate
        # 5% is not < 5%, so it goes to mid tier
        # Actually let's test properly
        _drift_window.clear()
        # 0% drift
        for _ in range(100):
            record_drift_result(True)
        assert adaptive_ttl(cfg) == 3600

    def test_low_drift_returns_max(self, cfg):
        """~4% drift → still max."""
        for _ in range(96):
            record_drift_result(True)
        for _ in range(4):
            record_drift_result(False)
        assert adaptive_ttl(cfg) == 3600

    def test_mid_drift_returns_1200(self, cfg):
        """10% drift → 1200s."""
        for _ in range(90):
            record_drift_result(True)
        for _ in range(10):
            record_drift_result(False)
        assert adaptive_ttl(cfg) == 1200

    def test_high_drift_returns_min(self, cfg):
        """20% drift → min TTL."""
        for _ in range(80):
            record_drift_result(True)
        for _ in range(20):
            record_drift_result(False)
        assert adaptive_ttl(cfg) == 300

    def test_boundary_exactly_5_percent(self, cfg):
        """Exactly 5% drift → mid tier (not < low threshold)."""
        for _ in range(95):
            record_drift_result(True)
        for _ in range(5):
            record_drift_result(False)
        # 5% is not < cfg.drift_low_threshold (0.05), but is < cfg.drift_high_threshold (0.15)
        assert adaptive_ttl(cfg) == 1200

    def test_boundary_exactly_15_percent(self, cfg):
        """Exactly 15% drift → min tier."""
        for _ in range(85):
            record_drift_result(True)
        for _ in range(15):
            record_drift_result(False)
        # 15% is not < cfg.drift_high_threshold (0.15)
        assert adaptive_ttl(cfg) == 300
