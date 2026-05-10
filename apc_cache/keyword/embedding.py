"""Async-safe embedding via sentence-transformers.

Blueprint §4.4:
- Global singleton SentenceTransformer, loaded at lifespan.
- ThreadPoolExecutor(max_workers=2) + asyncio.Semaphore(4).
- CPU inference < 5ms for all-MiniLM-L6-v2, 384-dim.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_model: Optional[object] = None  # SentenceTransformer
_executor: Optional[ThreadPoolExecutor] = None
_semaphore: Optional[asyncio.Semaphore] = None
_model_name: str = ""
_model_ver: str = ""


def init_embedding(model_name: str, model_ver: str) -> None:
    """Load the SentenceTransformer model and create executor + semaphore.

    Must be called once during lifespan startup, before any encode() call.
    """
    global _model, _executor, _semaphore, _model_name, _model_ver

    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s (ver=%s)", model_name, model_ver)
    _model = SentenceTransformer(model_name)
    _executor = ThreadPoolExecutor(max_workers=2)
    _semaphore = asyncio.Semaphore(4)
    _model_name = model_name
    _model_ver = model_ver

    # Warmup
    _model.encode("warmup", show_progress_bar=False)
    logger.info("Embedding model loaded and warmed up.")


async def encode(text: str) -> np.ndarray:
    """Encode a single text to a (384,) float32 ndarray.

    Must call init_embedding() first.
    """
    assert _model is not None, "init_embedding() must be called before encode()"
    assert _executor is not None
    assert _semaphore is not None

    async with _semaphore:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: _model.encode(text, show_progress_bar=False),
        )
        return np.asarray(result, dtype=np.float32)


def encode_sync(text: str) -> np.ndarray:
    """Synchronous encode for non-async contexts (e.g. reindex scripts)."""
    assert _model is not None, "init_embedding() must be called first"
    result = _model.encode(text, show_progress_bar=False)
    return np.asarray(result, dtype=np.float32)


def get_model_ver() -> str:
    return _model_ver


def shutdown_embedding() -> None:
    """Shutdown executor. Called during lifespan shutdown."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
