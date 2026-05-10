"""APC Keyword Cache Subsystem — configuration.

All config values are sourced from env vars. Those marked with ✓ support
hot-reload via admin API or Redis config key reads.
"""

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class APCConfig:
    # ── Cache sizing ──────────────────────────────────────────
    cache_max_size: int = 100  # ✗ restart required
    max_tpl_per_kw: int = 32  # ✗ restart
    max_candidates_to_check: int = 10  # ✗ restart

    # ── kw_cache TTL (adaptive range) ─────────────────────────
    kw_cache_ttl_min: int = 300  # ✓ hot-reload
    kw_cache_ttl_max: int = 3600  # ✓ hot-reload

    # ── Drift sampling ────────────────────────────────────────
    drift_sample_rate: float = 0.05  # ✓ hot-reload
    drift_high_threshold: float = 0.15  # ✓ hot-reload
    drift_low_threshold: float = 0.05  # ✓ hot-reload

    # ── Embedding ─────────────────────────────────────────────
    embed_model_name: str = "all-MiniLM-L6-v2"  # ✗ restart
    embed_model_ver: str = "v1"  # ✗ restart

    # ── Candidate thresholds ──────────────────────────────────
    candidate_dist_high: float = 0.85  # ✓ hot-reload
    candidate_dist_low: float = 0.50  # ✓ hot-reload
    candidate_top_k: int = 5  # ✗ restart

    # ── LLM normalization ─────────────────────────────────────
    normalize_llm_timeout: float = 2.0  # ✗ restart
    max_retry_on_entity: int = 2  # ✗ restart

    # ── Keyword index sync ────────────────────────────────────
    kw_index_sync_interval: int = 5  # ✗ restart

    # ── Decision log ──────────────────────────────────────────
    decision_log_enabled: bool = True  # ✗ restart

    # ── Snapshot ──────────────────────────────────────────────
    snapshot_enabled: bool = False  # ✗ restart
    snapshot_interval: int = 3600  # ✗ restart

    # ── Key prefix ────────────────────────────────────────────
    key_prefix: str = "apc"

    # ── Redis ─────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    @classmethod
    def from_env(cls) -> "APCConfig":
        return cls(
            cache_max_size=int(os.getenv("APC_CACHE_MAX_SIZE", "100")),
            max_tpl_per_kw=int(os.getenv("APC_MAX_TPL_PER_KW", "32")),
            max_candidates_to_check=int(os.getenv("APC_MAX_CANDIDATES_TO_CHECK", "10")),
            kw_cache_ttl_min=int(os.getenv("APC_KW_CACHE_TTL_MIN", "300")),
            kw_cache_ttl_max=int(os.getenv("APC_KW_CACHE_TTL_MAX", "3600")),
            drift_sample_rate=float(os.getenv("APC_DRIFT_SAMPLE_RATE", "0.05")),
            drift_high_threshold=float(os.getenv("APC_DRIFT_HIGH_THRESHOLD", "0.15")),
            drift_low_threshold=float(os.getenv("APC_DRIFT_LOW_THRESHOLD", "0.05")),
            embed_model_name=os.getenv("APC_EMBED_MODEL_NAME", "all-MiniLM-L6-v2"),
            embed_model_ver=os.getenv("APC_EMBED_MODEL_VER", "v1"),
            candidate_dist_high=float(os.getenv("APC_CANDIDATE_DIST_HIGH", "0.85")),
            candidate_dist_low=float(os.getenv("APC_CANDIDATE_DIST_LOW", "0.50")),
            candidate_top_k=int(os.getenv("APC_CANDIDATE_TOP_K", "5")),
            normalize_llm_timeout=float(os.getenv("APC_NORMALIZE_LLM_TIMEOUT", "2.0")),
            max_retry_on_entity=int(os.getenv("APC_MAX_RETRY_ON_ENTITY", "2")),
            kw_index_sync_interval=int(os.getenv("APC_KW_INDEX_SYNC_INTERVAL", "5")),
            decision_log_enabled=os.getenv("APC_DECISION_LOG_ENABLED", "True").lower() != "false",
            snapshot_enabled=os.getenv("APC_SNAPSHOT_ENABLED", "False").lower() == "true",
            snapshot_interval=int(os.getenv("APC_SNAPSHOT_INTERVAL", "3600")),
            key_prefix=os.getenv("APC_KEY_PREFIX", "apc"),
            redis_url=os.getenv("APC_REDIS_URL", "redis://localhost:6379/0"),
        )


# Module-level defaults used when a config instance isn't available.
CURRENT_VERSION = "v2.3"
