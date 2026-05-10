"""Cache lookup node — the five exit paths.

Blueprint §3: L1 → kw_cache → vector-normalize → L2/L3 → MISS
Blueprint §5.1: cache_lookup node, all requests enter here.

This module implements the cache_lookup LangGraph node. It reads
from an AgentState dict, populates cache_hit / keyword / plan_template,
and returns the updated state.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Optional

import redis.asyncio as redis

from apc_cache.config import APCConfig
from apc_cache.fingerprint import compute_fingerprint, ctx_compatible, ctx_fp_hash
from apc_cache.keyword.types import PlanTemplate
from apc_cache.normalize import qhash, task_sig

# ── public API ──────────────────────────────────────────────────


async def cache_lookup(state: dict[str, Any], cfg: APCConfig, r: redis.Redis) -> dict[str, Any]:
    """Execute the full cache lookup pipeline.

    Populates state with: cache_hit, cache_hit_layer, keyword,
    plan_template (if hit), keyword_candidates (if L2/L3 path taken).

    Returns the updated state dict.
    """
    query: str = state["query"]
    ctx: Any = state.get("context")
    agent_id: str = state["agent_id"]
    tools: list[Any] = state.get("tools", [])
    tools_hash: str = state.get("tools_hash", "")

    # ── 1.1–1.4  Compute L1 key ───────────────────────────
    ts = task_sig(query, agent_id, tools_hash)
    ctx_fp = compute_fingerprint(ctx, tools, agent_id)
    cfp_hash = ctx_fp_hash(ctx_fp)
    l1_key = f"{cfg.key_prefix}:l1:{agent_id}:{ts}:{cfp_hash}"

    # ── 1.5  L1 GET ───────────────────────────────────────
    tpl_id = await r.get(l1_key)
    if tpl_id is not None:
        tpl_id_str = tpl_id.decode()
        tpl_data = await r.hgetall(f"{cfg.key_prefix}:tpl:{tpl_id_str}")
        if tpl_data:
            template = PlanTemplate.from_redis_hash(tpl_data)
            if template.is_valid(tools_hash=tools_hash):
                from apc_cache import metrics

                metrics.cache_hit_total.labels(layer="L1").inc()
                state["cache_hit"] = True
                state["cache_hit_layer"] = "L1"
                state["plan_template"] = template
                state["keyword"] = None  # not needed for L1
                return state

    # ── 2.1–2.2  kw_cache ─────────────────────────────────
    qh = qhash(query)
    kw_cache_key = f"{cfg.key_prefix}:kw_cache:{qh}"
    cached_kw = await r.get(kw_cache_key)
    if cached_kw is not None:
        keyword = cached_kw.decode()
        from apc_cache import metrics

        metrics.kw_cache_hit_total.inc()

        # Side-band drift check (5 % of hits) — fire-and-forget
        if random.random() < cfg.drift_sample_rate:
            import asyncio

            asyncio.create_task(_check_drift(query, keyword, cfg, r))

        return await _l2_l3_search(state, keyword, agent_id, ctx_fp, cfg, r)

    # ── 3.x  Vector normalization path ← deferred to Phase 2 ──
    # In Phase 1, fall back to LLM-generated keyword (large_planner path).
    # Phase 2 injects: embed → candidates → normalize_keyword.
    state["cache_hit"] = False
    state["keyword"] = None
    from apc_cache import metrics

    metrics.cache_miss_total.inc()
    return state


async def lookup_with_keyword(
    state: dict[str, Any],
    keyword: str,
    cfg: APCConfig,
    r: redis.Redis,
) -> dict[str, Any]:
    """Entry point for Phase 2: after keyword is resolved (via kw_cache or normalization),
    continue to L2/L3 search.

    This is called by the vector normalization pipeline when it has a keyword.
    """
    agent_id: str = state["agent_id"]
    query: str = state["query"]
    ctx: Any = state.get("context")
    tools = state.get("tools", [])
    ctx_fp = compute_fingerprint(ctx, tools, agent_id)

    return await _l2_l3_search(state, keyword, agent_id, ctx_fp, cfg, r)


# ── internal ────────────────────────────────────────────────────


async def _l2_l3_search(
    state: dict[str, Any],
    keyword: str,
    agent_id: str,
    ctx_fp: Any,  # CtxFingerprint
    cfg: APCConfig,
    r: redis.Redis,
) -> dict[str, Any]:
    """L2 index lookup + L3 fingerprint filtering.

    Steps (blueprint §3, path ③):
    - SMEMBERS apc:tpl_idx:{agent}:{keyword}
    - Iterate up to MAX_CANDIDATES_TO_CHECK
    - validate() + ctx_compatible()
    - First match → promote to L1 → return hit
    - No match → return miss
    """
    import asyncio

    from apc_cache.fingerprint import ctx_compatible

    # ── 3.5  L2 index lookup ──────────────────────────────
    idx_key = f"{cfg.key_prefix}:tpl_idx:{agent_id}:{keyword}"
    candidate_ids = await r.smembers(idx_key)
    if not candidate_ids:
        state["cache_hit"] = False
        state["keyword"] = keyword
        from apc_cache import metrics

        metrics.cache_miss_total.inc()
        return state

    # ── 3.6  L3 candidate walk ────────────────────────────
    # Decode and limit
    cids: list[str] = sorted(cid.decode() for cid in candidate_ids)
    cids = cids[: cfg.max_candidates_to_check]

    tools_hash = state.get("tools_hash", "")
    query = state.get("query", "")

    for cid in cids:
        tpl_data = await r.hgetall(f"{cfg.key_prefix}:tpl:{cid}")
        if not tpl_data:
            continue

        template = PlanTemplate.from_redis_hash(tpl_data)
        if not template.is_valid(tools_hash=tools_hash):
            # stale — schedule async eviction
            asyncio.create_task(_async_evict_stale(cid, r, cfg))
            continue

        if ctx_compatible(template.ctx_fingerprint, ctx_fp):
            # ── 3.7  Promote to L1 ────────────────────────
            ts = task_sig(query, agent_id, tools_hash)
            cfp_hash = ctx_fp_hash(ctx_fp)
            l1_key = f"{cfg.key_prefix}:l1:{agent_id}:{ts}:{cfp_hash}"
            await r.set(l1_key, cid, ex=86400)
            await r.sadd(f"{cfg.key_prefix}:tpl_refs:{cid}", l1_key)

            from apc_cache import metrics

            metrics.cache_hit_total.labels(layer="L2_L3").inc()
            state["cache_hit"] = True
            state["cache_hit_layer"] = "L2_L3"
            state["plan_template"] = template
            state["keyword"] = keyword
            return state

    # ── 3.8  No match ─────────────────────────────────────
    state["cache_hit"] = False
    state["keyword"] = keyword
    from apc_cache import metrics

    metrics.cache_miss_total.inc()
    return state


async def _check_drift(
    query: str,
    cached_keyword: str,
    cfg: APCConfig,
    r: redis.Redis,
) -> None:
    """Side-band drift check: re-run normalization and compare.

    Blueprint §3.2, step 2.3: 5 % sample rate.
    Records drift metric for kw_cache adaptive TTL.
    """
    from apc_cache import metrics

    # In Phase 1, we don't have the full normalization pipeline.
    # Record as "stable" since we can't detect drift yet.
    # Phase 2 replaces this with actual normalization + comparison.
    metrics.kw_cache_drift_total.labels(direction="stable").inc()


async def _async_evict_stale(tpl_id: str, r: redis.Redis, cfg: APCConfig) -> None:
    """Background eviction of a stale/mismatched template."""
    try:
        refs_key = f"{cfg.key_prefix}:tpl_refs:{tpl_id}"
        l1_keys = await r.smembers(refs_key)
        if l1_keys:
            await r.delete(*l1_keys)
        await r.delete(refs_key, f"{cfg.key_prefix}:tpl:{tpl_id}")
    except Exception:
        pass  # best-effort, do not block the main path
