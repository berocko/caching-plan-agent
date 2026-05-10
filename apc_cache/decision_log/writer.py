"""Async decision log writer.

Blueprint §11.3: Asynchronously write normalization decisions to PostgreSQL.
Does NOT block the main request path.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from apc_cache.decision_log.models import Base, KWNormalizationLog

logger = logging.getLogger(__name__)

_engine: Optional[object] = None  # AsyncEngine
_session_factory: Optional[object] = None  # async_sessionmaker


def init_decision_log(dsn: str) -> None:
    """Initialise the async engine and session factory.

    Args:
        dsn: PostgreSQL connection string, e.g.
             "postgresql+asyncpg://user:pass@localhost:5432/apc"
    """
    global _engine, _session_factory
    _engine = create_async_engine(dsn, echo=False, pool_size=5, max_overflow=10)
    _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def create_tables() -> None:
    """Create all tables if they don't exist."""
    assert _engine is not None, "init_decision_log() must be called first"
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def write_normalization_log(
    query_hash: str,
    query_vec: Optional[list[float]],
    candidates: list[dict],
    action: str,
    chosen_kw: str,
    chosen_rank: Optional[int],
    model_ver: str,
    latency_ms: int,
) -> None:
    """Write a single normalization decision record.

    Fire-and-forget: exceptions are logged but never raised to the caller.
    """
    if _session_factory is None:
        return

    try:
        async with _session_factory() as session:
            log = KWNormalizationLog(
                query_hash=query_hash,
                query_vec=json.dumps(query_vec) if query_vec else None,
                candidates=candidates,
                action=action,
                chosen_kw=chosen_kw,
                chosen_rank=chosen_rank,
                model_ver=model_ver,
                latency_ms=latency_ms,
            )
            session.add(log)
            await session.commit()
    except Exception:
        logger.exception("Failed to write decision log")


async def shutdown_decision_log() -> None:
    """Close the engine connection pool."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
