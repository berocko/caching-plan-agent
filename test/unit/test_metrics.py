"""Tests for apc_cache.metrics — ensure all metrics are registered and
accept correct label values.

These are structural validation tests, not functional assertions on values.
"""

import pytest
from prometheus_client import Counter, Gauge, Histogram

from apc_cache import metrics


def _collect_value(metric, **labels):
    """Return the current value of a metric for the given labels."""
    if labels:
        return metric.labels(**labels)._value.get()
    return metric._value.get()


class TestMetricsExist:
    """Validate each metric is the expected type."""

    def test_cache_hit_total_is_counter_with_layer_label(self):
        assert isinstance(metrics.cache_hit_total, Counter)
        # labels should accept "layer"
        metrics.cache_hit_total.labels(layer="L1")
        metrics.cache_hit_total.labels(layer="L2_L3")

    def test_cache_miss_total_is_counter(self):
        assert isinstance(metrics.cache_miss_total, Counter)

    def test_kw_cache_hit_total_is_counter(self):
        assert isinstance(metrics.kw_cache_hit_total, Counter)

    def test_kw_cache_drift_total_has_direction_label(self):
        assert isinstance(metrics.kw_cache_drift_total, Counter)
        metrics.kw_cache_drift_total.labels(direction="stable")
        metrics.kw_cache_drift_total.labels(direction="changed")

    def test_kw_normalization_action_has_action_label(self):
        assert isinstance(metrics.kw_normalization_action, Counter)
        for action in ("shortcut_reuse", "shortcut_new", "ask_llm"):
            metrics.kw_normalization_action.labels(action=action)

    def test_kw_normalization_latency_ms_is_histogram(self):
        assert isinstance(metrics.kw_normalization_latency_ms, Histogram)

    def test_kw_normalization_fallback_total_has_reason_label(self):
        assert isinstance(metrics.kw_normalization_fallback_total, Counter)
        for reason in ("timeout", "bad_format", "hallucination", "entity"):
            metrics.kw_normalization_fallback_total.labels(reason=reason)

    def test_kw_index_size_is_gauge(self):
        assert isinstance(metrics.kw_index_size, Gauge)

    def test_kw_near_duplicate_total_is_counter(self):
        assert isinstance(metrics.kw_near_duplicate_total, Counter)

    def test_embed_model_version_is_gauge_with_label(self):
        assert isinstance(metrics.embed_model_version, Gauge)
        metrics.embed_model_version.labels(model_ver="v1")
