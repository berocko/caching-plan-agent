"""Context fingerprinting — structured, NOT content-based hashing.

Blueprint §4.2, §6.1-6.3: CtxFingerprint is a 5-tuple (context_type,
length_bucket, tools, agent_role, context_schema). The structured
approach ensures two different companies' financial reports hit the
same template, which a content hash would never achieve.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from apc_cache.keyword.types import CtxFingerprint, adjacent_buckets

# Re-export for convenience
__all__ = ["CtxFingerprint", "adjacent_buckets", "compute_fingerprint", "ctx_compatible", "ctx_fp_hash"]


def _classify_context(ctx: Any) -> str:
    """Classify the shape of context data."""
    if isinstance(ctx, dict):
        keys_lower = {k.lower() for k in ctx}
        if keys_lower & {"balance_sheet", "income_statement", "cash_flow"}:
            return "financial_report"
        if keys_lower & {"rows", "columns"}:
            return "tabular_data"
        return "structured_json"
    if isinstance(ctx, str):
        return "long_document" if len(ctx) > 10000 else "short_text"
    if isinstance(ctx, list):
        return "tabular_data" if ctx and isinstance(ctx[0], dict) else "structured_json"
    return "unknown"


def _bucket_length(ctx: Any) -> str:
    """Estimate token count and bucket."""
    import json as _json

    raw = ctx if isinstance(ctx, str) else _json.dumps(ctx, ensure_ascii=False)
    tokens = len(raw) // 4  # rough token estimate
    if tokens < 1000:
        return "short"
    if tokens <= 10000:
        return "medium"
    return "long"


def _extract_schema(obj: Any, max_depth: int = 2, _depth: int = 0) -> frozenset[str]:
    """Recursively extract key names up to max_depth, joining with '.'.

    Non-dict input → empty frozenset.
    """
    if not isinstance(obj, dict) or _depth >= max_depth:
        return frozenset()

    keys: set[str] = set()
    for k, v in obj.items():
        keys.add(k)
        if isinstance(v, dict):
            for sub in _extract_schema(v, max_depth, _depth + 1):
                keys.add(f"{k}.{sub}")
    return frozenset(keys)


def compute_fingerprint(
    context: Any,
    tools: list[Any],
    agent_role: str,
    query_lang: str = "en",
) -> CtxFingerprint:
    """Build a structured CtxFingerprint from request context."""
    tool_names = frozenset(t.name for t in tools)
    return CtxFingerprint(
        context_type=_classify_context(context),
        length_bucket=_bucket_length(context),
        tools=tool_names,
        agent_role=agent_role,
        context_schema=_extract_schema(context),
        query_lang=query_lang,
    )


def ctx_fp_hash(ctx_fp: CtxFingerprint) -> str:
    """ctx_fp_hash = sha256(json(ctx_fp, sort_keys=True))[:12]"""
    d = ctx_fp.to_dict(sort_keys=True)
    raw = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def ctx_compatible(tpl_ctx: CtxFingerprint, query_ctx: CtxFingerprint) -> bool:
    """Determine whether *tpl_ctx* is compatible with *query_ctx*.

    Short-circuit AND (blueprint §4.3):
    1. context_type must match
    2. tools must match
    3. agent_role must match
    4. length_bucket must be adjacent (not cross-bucket)
    5. tpl's schema must be a subset of query's schema (if non-empty)
    """
    if tpl_ctx.context_type != query_ctx.context_type:
        return False
    if tpl_ctx.tools != query_ctx.tools:
        return False
    if tpl_ctx.agent_role != query_ctx.agent_role:
        return False
    if not adjacent_buckets(tpl_ctx.length_bucket, query_ctx.length_bucket):
        return False
    if tpl_ctx.context_schema and not (tpl_ctx.context_schema <= query_ctx.context_schema):
        return False
    return True
