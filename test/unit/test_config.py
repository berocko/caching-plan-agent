"""Tests for apc_cache.config — configuration management."""

import os

import pytest

from apc_cache.config import APCConfig


class TestAPCConfig:
    def test_default_values(self):
        cfg = APCConfig()
        assert cfg.cache_max_size == 100
        assert cfg.max_tpl_per_kw == 32
        assert cfg.max_candidates_to_check == 10
        assert cfg.kw_cache_ttl_min == 300
        assert cfg.kw_cache_ttl_max == 3600
        assert cfg.drift_sample_rate == 0.05
        assert cfg.embed_model_name == "all-MiniLM-L6-v2"
        assert cfg.embed_model_ver == "v1"
        assert cfg.candidate_dist_high == 0.85
        assert cfg.candidate_dist_low == 0.50
        assert cfg.key_prefix == "apc"
        assert cfg.redis_url == "redis://localhost:6379/0"

    def test_from_env_defaults(self):
        """Without env vars set, from_env() returns defaults."""
        cfg = APCConfig.from_env()
        assert cfg.cache_max_size == 100

    def test_from_env_override(self, monkeypatch):
        monkeypatch.setenv("APC_CACHE_MAX_SIZE", "50")
        monkeypatch.setenv("APC_EMBED_MODEL_VER", "v2")
        monkeypatch.setenv("APC_DRIFT_SAMPLE_RATE", "0.10")
        monkeypatch.setenv("APC_DECISION_LOG_ENABLED", "false")
        monkeypatch.setenv("APC_SNAPSHOT_ENABLED", "true")

        cfg = APCConfig.from_env()
        assert cfg.cache_max_size == 50
        assert cfg.embed_model_ver == "v2"
        assert cfg.drift_sample_rate == 0.10
        assert cfg.decision_log_enabled is False
        assert cfg.snapshot_enabled is True

    def test_key_prefix_custom(self):
        cfg = APCConfig(key_prefix="mycache")
        assert cfg.key_prefix == "mycache"

    def test_key_prefix_from_env(self, monkeypatch):
        monkeypatch.setenv("APC_KEY_PREFIX", "customcache")
        cfg = APCConfig.from_env()
        assert cfg.key_prefix == "customcache"

    def test_boolean_parsing(self, monkeypatch):
        # Various ways to say false
        monkeypatch.setenv("APC_DECISION_LOG_ENABLED", "False")
        cfg = APCConfig.from_env()
        assert cfg.decision_log_enabled is False

        monkeypatch.setenv("APC_DECISION_LOG_ENABLED", "0")
        cfg2 = APCConfig.from_env()
        assert cfg2.decision_log_enabled is True  # only "false" string

    def test_slots_no_extra_attrs(self):
        cfg = APCConfig()
        with pytest.raises(AttributeError):
            cfg.nonexistent = 42  # type: ignore
