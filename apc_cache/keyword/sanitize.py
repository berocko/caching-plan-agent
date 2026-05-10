"""Keyword sanitizer — whitelist filter, length cap, empty rejection.

Blueprint §4.6 step 6: sanitize_keyword()
Final §4.5: sanitize rules
"""

import re
from typing import Optional

_MAX_LEN = 64
_MIN_LEN = 3
_VALID_RE = re.compile(r"[^a-z0-9\s\-_]")


def sanitize_keyword(raw: str) -> Optional[str]:
    """Clean and validate a keyword string.

    Returns None if the result is shorter than _MIN_LEN after cleaning.
    """
    if not raw:
        return None

    kw = raw.strip()
    kw = kw[: _MAX_LEN - 1] if len(kw) > _MAX_LEN else kw
    kw = kw.lower()
    kw = _VALID_RE.sub("", kw)
    kw = re.sub(r"\s+", " ", kw).strip()

    if len(kw) < _MIN_LEN:
        return None

    return kw


def contains_query_entities(keyword: str, query: str) -> bool:
    """Check if the keyword embeds proper nouns, numbers, or dates from the query.

    This is an anti-entity-injection guard: normalized keywords should be
    generic descriptors, not data-specific tags (blueprint §4.6 step 5).
    """
    import re as _re

    query_lower = query.lower()
    kw_lower = keyword.lower()

    potential_entities: list[str] = []

    years = _re.findall(r"\b(19|20)\d{2}\b", query_lower)
    potential_entities.extend(years)

    numbers = _re.findall(r"\b\d{4,}\b", query_lower)
    potential_entities.extend(numbers)

    proper_pattern = _re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", query)
    potential_entities.extend(e.lower() for e in proper_pattern)

    for entity in potential_entities:
        if entity in kw_lower:
            return True

    return False
