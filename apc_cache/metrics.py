"""Prometheus metrics for the APC Keyword Cache Subsystem.

Blueprint §8: All metric exposure points.

Usage:
    from apc_cache import metrics
    metrics.cache_hit_total.labels(layer="L1").inc()
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Cache hit/miss ─────────────────────────────────────────────

cache_hit_total = Counter(
    "apc_cache_hit_total",
    "Total cache hits by layer.",
    ["layer"],  # "L1" | "L2_L3"
)

cache_miss_total = Counter(
    "apc_cache_miss_total",
    "Total cache misses.",
)

# ── kw_cache ───────────────────────────────────────────────────

kw_cache_hit_total = Counter(
    "apc_kw_cache_hit_total",
    "Keyword cache hit count.",
)

kw_cache_drift_total = Counter(
    "apc_kw_cache_drift_total",
    "Drift detection results from side-band sampling.",
    ["direction"],  # "stable" | "changed"
)

# ── Keyword normalization ──────────────────────────────────────

kw_normalization_action = Counter(
    "apc_kw_normalization_action_total",
    "Normalization action distribution.",
    ["action"],  # "shortcut_reuse" | "shortcut_new" | "ask_llm"
)

kw_normalization_latency_ms = Histogram(
    "apc_kw_normalization_latency_ms",
    "Keyword normalization latency in milliseconds.",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

kw_normalization_fallback_total = Counter(
    "apc_kw_normalization_fallback_total",
    "Fallback reasons.",
    ["reason"],  # "timeout" | "bad_format" | "hallucination" | "entity"
)

# ── Keyword index ──────────────────────────────────────────────

kw_index_size = Gauge(
    "apc_kw_index_size",
    "Number of keywords in the local index.",
)

kw_near_duplicate_total = Counter(
    "apc_kw_near_duplicate_total",
    "Near-duplicate keywords detected (cosine > 0.95).",
)

# ── Embedding model ────────────────────────────────────────────

embed_model_version = Gauge(
    "apc_embed_model_version",
    "Current embedding model version (constant gauge).",
    ["model_ver"],
)
