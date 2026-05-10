"""Query normalization for L1 cache key computation.

Blueprint §4.1: Lossless surface normalisation only — no semantic transforms.
- Unicode NFKC
- Lowercase
- Merge whitespace

Deliberately NOT doing: stopword removal, synonym replacement, number placeholders.
Those belong to the keyword layer (L2), not L1.
"""

import hashlib
import re
import unicodedata


def normalize(query: str) -> str:
    """Return a losslessly normalised copy of *query* for L1 key derivation."""
    q = unicodedata.normalize("NFKC", query)
    q = q.lower()
    q = re.sub(r"\s+", " ", q).strip()
    return q


def task_sig(query: str, agent_id: str, tools_hash: str) -> str:
    """task_sig = sha256(normalize(query) | agent_id | tools_hash)[:16]"""
    nq = normalize(query)
    raw = f"{nq}|{agent_id}|{tools_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def qhash(query: str) -> str:
    """qhash = sha256(normalize(query))[:16]  (used for kw_cache key)"""
    nq = normalize(query)
    return hashlib.sha256(nq.encode()).hexdigest()[:16]


def kw_lock_hash(keyword: str) -> str:
    """Return the lock hash for a keyword: sha256(keyword)[:16]"""
    return hashlib.sha256(keyword.encode()).hexdigest()[:16]
