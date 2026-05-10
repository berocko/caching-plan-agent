"""Multi-language embedding model support.

Blueprint Phase 4: When switching from English-only (all-MiniLM-L6-v2) to
multilingual (paraphrase-multilingual-MiniLM-L12-v2), keyword embeddings
and query embeddings must use the same model_ver.

This module provides the model registry and the detect/switch utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbedModelSpec:
    name: str  # HuggingFace model name
    ver: str  # model_ver tag
    dim: int  # embedding dimension
    multilingual: bool
    description: str


# ── Model registry ──────────────────────────────────────────────

MODEL_REGISTRY: dict[str, EmbedModelSpec] = {
    "miniLM-en": EmbedModelSpec(
        name="all-MiniLM-L6-v2",
        ver="v1",
        dim=384,
        multilingual=False,
        description="English, 384-dim, ~80MB, CPU < 5ms",
    ),
    "miniLM-multi": EmbedModelSpec(
        name="paraphrase-multilingual-MiniLM-L12-v2",
        ver="v2",
        dim=384,
        multilingual=True,
        description="Multilingual (50+ languages), 384-dim, ~420MB, CPU ~10ms",
    ),
}

# Default model
DEFAULT_MODEL_KEY = "miniLM-en"


def get_model_spec(key: str = DEFAULT_MODEL_KEY) -> EmbedModelSpec:
    """Get the spec for a registered model by key."""
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {key}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


# ── Language detection helper ───────────────────────────────────

_LANG_PATTERNS: dict[str, str] = {
    "zh": r"[一-鿿]",
    "ja": r"[぀-ゟ゠-ヿ]",
    "ko": r"[가-힯]",
    "ar": r"[؀-ۿ]",
    "ru": r"[Ѐ-ӿ]",
}


def detect_query_lang(query: str) -> str:
    """Crude script-based language detection.

    Returns ISO 639-1 code or "en" as default.
    """
    import re

    for lang, pattern in _LANG_PATTERNS.items():
        if re.search(pattern, query):
            return lang
    return "en"


def is_multilingual_query(query: str) -> bool:
    """Return True if the query contains non-Latin scripts."""
    return detect_query_lang(query) != "en"
