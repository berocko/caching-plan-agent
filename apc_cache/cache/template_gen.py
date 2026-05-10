"""Template generation node — write-back to all cache layers.

Blueprint §4.8: Called after large_planner produces execution_log.
Writes template to L2 index, keyword vector index, and promotes to L1.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import redis.asyncio as redis

from apc_cache.config import APCConfig, CURRENT_VERSION
from apc_cache.fingerprint import compute_fingerprint, ctx_fp_hash
from apc_cache.keyword.sanitize import sanitize_keyword
from apc_cache.keyword.types import PlanTemplate
from apc_cache.normalize import task_sig


async def template_gen(state: dict[str, Any], cfg: APCConfig, r: redis.Redis) -> dict[str, Any]:
    """Generate template from execution_log and write to all cache layers.

    Blueprint §4.8, path ⑤:
    1. rule_filter(execution_log) → extract plan structure
    2. llm_filter → generalize (GPT-4o-mini)
    3-4. Write template → HSET
    5-6. Maintain keyword index
    7. Promote to L1
    8-9. Async: decision log + LRU eviction
    """
    from apc_cache import metrics

    execution_log: list[dict[str, Any]] = state.get("execution_log", [])
    keyword: str = state.get("keyword", "")
    agent_id: str = state["agent_id"]
    tools_hash: str = state.get("tools_hash", "")
    tools: list[Any] = state.get("tools", [])
    query: str = state["query"]
    ctx: Any = state.get("context")
    ctx_fp = compute_fingerprint(ctx, tools, agent_id)

    # ── 5.1  rule_filter ──────────────────────────────────
    plan_structure = _rule_filter(execution_log)

    # ── 5.2  llm_filter (generalization) ──────────────────
    # In Phase 1, rule_filter alone provides the template structure.
    # Phase 2/3 can add LLM generalization via lightweight_llm.
    generalized = plan_structure

    # ── 5.3–5.4  Build & write template ───────────────────
    tpl_id = str(uuid.uuid4())
    ttl = _classify_ttl(query)

    template = PlanTemplate(
        template_id=tpl_id,
        version=CURRENT_VERSION,
        schema_hash=tools_hash,
        ctx_fingerprint=ctx_fp,
        task=generalized.get("task", query),
        steps=generalized.get("steps", []),
        created_at=time.time(),
        ttl_seconds=ttl,
    )

    tpl_key = f"{cfg.key_prefix}:tpl:{tpl_id}"
    await r.hset(tpl_key, mapping=template.to_redis_hash())

    # ── 5.5  L2 index ─────────────────────────────────────
    kw = await _resolve_alias(keyword, r, cfg)
    kw = sanitize_keyword(kw) or kw
    idx_key = f"{cfg.key_prefix}:tpl_idx:{agent_id}:{kw}"
    await r.sadd(idx_key, tpl_id)

    # ── 5.6  Keyword vector index maintenance ─────────────
    await _upsert_keyword(kw, cfg, r)

    # ── 5.7  Promote to L1 ────────────────────────────────
    ts = task_sig(query, agent_id, tools_hash)
    cfp_hash = ctx_fp_hash(ctx_fp)
    l1_key = f"{cfg.key_prefix}:l1:{agent_id}:{ts}:{cfp_hash}"
    await r.set(l1_key, tpl_id, ex=86400)
    await r.sadd(f"{cfg.key_prefix}:tpl_refs:{tpl_id}", l1_key)

    # ── 5.8–5.9  Async side effects ───────────────────────
    if cfg.decision_log_enabled:
        asyncio.create_task(_log_decision(tpl_id, kw, state, cfg, r))
    asyncio.create_task(_enforce_lru_if_needed(cfg, r))

    metrics.cache_miss_total.inc()  # this path means we had a miss earlier
    state["plan_template"] = template
    return state


# ── internal helpers ────────────────────────────────────────────


async def _resolve_alias(keyword: str, r: redis.Redis, cfg: APCConfig) -> str:
    """Check kw_alias table: if keyword has an alias, use canonical form."""
    alias_key = f"{cfg.key_prefix}:kw_alias"
    canonical = await r.hget(alias_key, keyword)
    if canonical:
        return canonical.decode()
    return keyword


async def _upsert_keyword(keyword: str, cfg: APCConfig, r: redis.Redis) -> None:
    """Write keyword metadata + timeline entry.

    In Phase 1: writes kw_meta (without embedding) + kw_timeline.
    Phase 2: adds embedding vector, duplicate checks, near-duplicate detection.
    """
    from apc_cache import metrics

    # Check if keyword already exists
    meta_key = f"{cfg.key_prefix}:kw_meta:{keyword}"
    existing = await r.hexists(meta_key, "model_ver")
    if existing:
        return

    now = time.time()
    await r.hset(
        meta_key,
        mapping={
            "model_ver": cfg.embed_model_ver,
            "dim": "384",
            "created_at": str(now),
        },
    )
    await r.zadd(f"{cfg.key_prefix}:kw_timeline", {keyword: now})
    metrics.kw_index_size.inc()


def _rule_filter(execution_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract plan structure from execution log without LLM.

    Blueprint §4.8 step 1: a deterministic extraction that captures
    the task description and step sequence from the actor's log.
    """
    if not execution_log:
        return {"task": "", "steps": []}

    steps: list[dict[str, Any]] = []
    task = ""

    for entry in execution_log:
        if isinstance(entry, dict):
            if "task" in entry:
                task = entry["task"]
            if "step" in entry:
                steps.append({"description": str(entry["step"])})
            elif "action" in entry:
                steps.append({"description": str(entry["action"])})

    return {"task": task, "steps": steps}


def _classify_ttl(query: str) -> int:
    """Assign TTL based on query temporality heuristics.

    Time-sensitive queries (containing "today", "current", "now", dates)
    get shorter TTL (3600s). Otherwise default to 86400s.
    """
    import re

    low = query.lower()
    temporal_hints = {"today", "current", "now", "latest", "recent", "this week", "this month"}
    if any(hint in low for hint in temporal_hints):
        return 3600
    if re.search(r"\b(19|20)\d{2}\b", low):
        return 3600
    return 86400


async def _log_decision(tpl_id: str, keyword: str, state: dict[str, Any], cfg: APCConfig, r: redis.Redis) -> None:
    """Async decision log writer. Phase 3 fills in PG schema.

    Phase 1: no-op placeholder. Phase 3 adds actual PG write.
    """
    pass


async def _enforce_lru_if_needed(cfg: APCConfig, r: redis.Redis) -> None:
    """Check template count and trigger LRU eviction if over limit.

    Phase 1: simple count check. Phase 3 adds Lua-based LRU script.
    """
    # Count templates by scanning for apc:tpl:* keys
    keys = await r.keys(f"{cfg.key_prefix}:tpl:*")
    if len(keys) > cfg.cache_max_size:
        from apc_cache.cache.eviction import evict_lru

        await evict_lru(cfg, r)
