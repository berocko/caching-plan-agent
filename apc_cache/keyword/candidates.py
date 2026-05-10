"""Candidate building from vector search results.

Blueprint §4.5: Take top-k results from keyword_index.search() and
apply the shortcut decision logic (reuse / new / ask_llm).
"""

from __future__ import annotations

from apc_cache.config import APCConfig
from apc_cache.keyword.types import CandidateResult


def build_candidates(
    query_vec: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    kw_index: "KeywordIndexManager",  # type: ignore[name-defined]  # noqa: F821
    cfg: APCConfig,
) -> CandidateResult:
    """Run vector search and apply shortcut logic.

    Steps (blueprint §4.5):
    1. raw = kw_index.search(query_vec, top_k)
    2. filtered = [(kw, score) for ... if score >= DIST_LOW]
    3. empty → shortcut_new
    4. single with score ≥ DIST_HIGH → shortcut_reuse
    5. top with big lead → shortcut_reuse
    6. Otherwise → ask_llm (top-3 candidates)
    """
    top_k = cfg.candidate_top_k
    raw = kw_index.search(query_vec, top_k)

    dist_high = cfg.candidate_dist_high
    dist_low = cfg.candidate_dist_low

    filtered = [(kw, score) for kw, score in raw if score >= dist_low]

    # No valid candidates → force new keyword
    if not filtered:
        return CandidateResult(items=[], action="shortcut_new")

    # Single strong candidate → reuse
    if len(filtered) == 1 and filtered[0][1] >= dist_high:
        return CandidateResult(items=filtered, action="shortcut_reuse")

    # Clear leader with big gap → reuse top-1
    if (
        len(filtered) >= 2
        and filtered[0][1] >= dist_high
        and (filtered[0][1] - filtered[1][1]) > 0.15
    ):
        return CandidateResult(items=[filtered[0]], action="shortcut_reuse")

    # Ambiguous zone → ask LLM with top-3
    return CandidateResult(items=filtered[:3], action="ask_llm")
