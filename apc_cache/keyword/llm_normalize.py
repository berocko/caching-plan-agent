"""LLM-based keyword normalization with fallback chain.

Blueprint §4.6:
- Primary: LLM chooses reuse/new from candidate list.
- Fallback 1 (timeout): use top candidate if score > 0.70, else pure generation.
- Fallback 2 (bad format): sanitize, then pure generation if empty.
- Fallback 3 (hallucination): LLM claimed reuse but kw not in candidates → fallback.
- Fallback 4 (entity injection): retry with stricter prompt → sanitize.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from apc_cache.config import APCConfig
from apc_cache.keyword.sanitize import contains_query_entities, sanitize_keyword

logger = logging.getLogger(__name__)

NORMALIZE_PROMPT = """You are a keyword normalizer, not a generator. Your job is to decide whether the user's query can be represented by one of the existing keywords, or if a new keyword is needed.

Existing keyword candidates (sorted by relevance):
{candidates_text}

User query: {query}

Rules:
1. If the query's meaning matches an existing keyword, return that keyword EXACTLY as written.
2. If no existing keyword matches, generate a NEW keyword:
   - Lowercase English only
   - No more than 8 words
   - No punctuation (underscores and hyphens are allowed)
   - Generic, no proper nouns, numbers, or dates
   - Describe the TASK, not the input data

Respond with ONLY the keyword, nothing else."""

STRICTER_NORMALIZE_PROMPT = """You are a keyword normalizer. Your output must be generic and contain NO proper nouns, company names, person names, dates, years, or specific numbers from the query.

Existing keyword candidates:
{candidates_text}

User query: {query}

Return ONLY one keyword (lowercase, max 8 words, no punctuation except underscores/hyphens).
If the query matches an existing candidate's meaning, return that candidate EXACTLY.
Otherwise, create a new GENERIC keyword describing the analytical task, not the data.

Keyword:"""


async def normalize_keyword(
    query: str,
    candidates: list[tuple[str, float]],
    llm: Any,
    cfg: APCConfig,
) -> str:
    """Normalize a query into a canonical keyword.

    Args:
        query: The raw user query.
        candidates: List of (keyword, score) from vector search.
        llm: A LangChain-compatible chat model (e.g., ChatOpenAI).
        cfg: Configuration.

    Returns:
        A normalized keyword string.
    """
    from apc_cache import metrics

    t0 = asyncio.get_event_loop().time()

    try:
        # ── Call LLM ──────────────────────────────────────
        candidates_text = _format_candidates(candidates)
        prompt = NORMALIZE_PROMPT.format(
            candidates_text=candidates_text,
            query=query,
        )
        raw = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=cfg.normalize_llm_timeout,
        )
        kw = _parse_llm_keyword(raw)

        # ── Anti-hallucination check ──────────────────────
        candidate_strings = {c[0] for c in candidates}
        intent = _detect_intent(raw, candidate_strings)
        if intent == "reuse" and kw not in candidate_strings:
            logger.warning("LLM hallucinated: claimed reuse but kw=%r not in candidates", kw)
            metrics.kw_normalization_fallback_total.labels(reason="hallucination").inc()
            metrics.kw_normalization_action.labels(action="shortcut_new").inc()
            return _fallback_extract_keyword(query)

        # ── Anti-entity injection check ──────────────────
        if contains_query_entities(kw, query):
            logger.warning("Entity injection detected in kw=%r, retrying", kw)
            retry_kw = await _retry_stricter(query, candidates, llm, cfg)
            if retry_kw and not contains_query_entities(retry_kw, query):
                kw = retry_kw
            else:
                kw = sanitize_keyword(kw) or _fallback_extract_keyword(query)
                if contains_query_entities(kw, query) and retry_kw is None:
                    metrics.kw_normalization_fallback_total.labels(reason="entity").inc()

        # ── Final sanitize ───────────────────────────────
        final_kw = sanitize_keyword(kw)
        if not final_kw:
            metrics.kw_normalization_fallback_total.labels(reason="bad_format").inc()
            final_kw = _fallback_extract_keyword(query)

        elapsed_ms = (asyncio.get_event_loop().time() - t0) * 1000
        metrics.kw_normalization_latency_ms.observe(elapsed_ms)

        return final_kw

    except (asyncio.TimeoutError, Exception) as exc:
        elapsed_ms = (asyncio.get_event_loop().time() - t0) * 1000
        metrics.kw_normalization_latency_ms.observe(elapsed_ms)

        if isinstance(exc, asyncio.TimeoutError):
            metrics.kw_normalization_fallback_total.labels(reason="timeout").inc()
        else:
            metrics.kw_normalization_fallback_total.labels(reason="bad_format").inc()

        logger.warning("LLM normalization failed: %s", exc)

        # Fallback: use top candidate if score > 0.70
        if candidates and candidates[0][1] > 0.70:
            return candidates[0][0]

        return _fallback_extract_keyword(query)


# ── internal helpers ────────────────────────────────────────────


def _format_candidates(candidates: list[tuple[str, float]]) -> str:
    lines = []
    for i, (kw, score) in enumerate(candidates, 1):
        lines.append(f"  {i}. {kw}  (similarity: {score:.3f})")
    return "\n".join(lines) if lines else "(no existing candidates)"


def _parse_llm_keyword(raw: Any) -> str:
    """Extract the first meaningful line from LLM output."""
    if hasattr(raw, "content"):
        text = raw.content
    else:
        text = str(raw)

    text = text.strip()
    line = text.split("\n")[0].strip()
    # Strip quotes
    line = line.strip('"').strip("'").strip("`")
    return line.strip()


def _detect_intent(raw: Any, candidate_strings: set[str]) -> str:
    """Detect whether LLM intended to reuse a candidate or create a new one."""
    text = raw.content if hasattr(raw, "content") else str(raw)
    parsed = _parse_llm_keyword(raw)
    if parsed in candidate_strings:
        return "reuse"
    # Check if LLM explicitly said to use an existing one
    text_lower = text.lower()
    if "existing" in text_lower or "reuse" in text_lower or "match" in text_lower:
        return "reuse"
    return "new"


async def _retry_stricter(
    query: str,
    candidates: list[tuple[str, float]],
    llm: Any,
    cfg: APCConfig,
) -> Optional[str]:
    """Retry with a stricter prompt that forbids entity inclusion."""
    for _ in range(cfg.max_retry_on_entity):
        try:
            candidates_text = _format_candidates(candidates)
            prompt = STRICTER_NORMALIZE_PROMPT.format(
                candidates_text=candidates_text,
                query=query,
            )
            raw = await asyncio.wait_for(
                llm.ainvoke(prompt),
                timeout=cfg.normalize_llm_timeout,
            )
            kw = _parse_llm_keyword(raw)
            if not contains_query_entities(kw, query):
                return sanitize_keyword(kw) or None
        except Exception:
            break
    return None


def _fallback_extract_keyword(query: str) -> str:
    """Pure rule-based keyword extraction — no LLM, no embedding.

    Blueprint: fallback when everything else fails.
    Strips common stopwords and returns the longest remaining phrase.
    """
    import re

    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "not", "no", "this", "that", "these",
        "those", "it", "its", "i", "me", "my", "we", "our", "you",
        "your", "he", "she", "they", "what", "which", "who", "whom",
        "how", "when", "where", "why", "please", "help",
    }

    cleaned = re.sub(r"[^\w\s]", " ", query.lower())
    words = [w for w in cleaned.split() if w not in stopwords]
    if not words:
        return "general_query"

    # Take up to 8 meaningful words
    kw = " ".join(words[:8])
    return sanitize_keyword(kw) or "general_query"
