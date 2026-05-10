"""Decision log database models.

Blueprint §11.3: kw_normalization_log table for training data collection.
Uses SQLAlchemy ORM. The pgvector extension is required for the vector column.
"""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, BIGINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class KWNormalizationLog(Base):
    __tablename__ = "kw_normalization_log"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_vec: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON-serialised float list
    candidates: Mapped[dict] = mapped_column(JSONB, nullable=False)
    action: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # 'shortcut_reuse' | 'shortcut_new' | 'ask_llm_reuse' | 'ask_llm_new'
    chosen_kw: Mapped[str] = mapped_column(String(256), nullable=False)
    chosen_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_ver: Mapped[str] = mapped_column(String(32), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<KWNormLog id={self.id} action={self.action} "
            f"chosen_kw={self.chosen_kw} latency={self.latency_ms}ms>"
        )
