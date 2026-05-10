"""Core types shared across the keyword subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apc_cache.config import CURRENT_VERSION


@dataclass(slots=True)
class CtxFingerprint:
    """Structured context fingerprint — NOT a content hash.

    Design rationale (blueprint §4.2, §6.1):
    - context_type classifies the shape of the input data.
    - length_bucket buckets token counts to allow adjacency matching.
    - tools is a frozenset of tool names available to the agent.
    - agent_role names the agent/tenant.
    - context_schema captures top-level and second-level key names (max_depth=2).
    """

    context_type: str  # "financial_report" | "tabular_data" | "long_document" | ...
    length_bucket: str  # "short" | "medium" | "long"
    tools: frozenset[str]
    agent_role: str
    context_schema: frozenset[str] = field(default_factory=frozenset)
    query_lang: str = "en"  # ISO 639-1; added in Phase 4 (multi-language)

    def to_dict(self, sort_keys: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "context_type": self.context_type,
            "length_bucket": self.length_bucket,
            "tools": sorted(self.tools),
            "agent_role": self.agent_role,
            "context_schema": sorted(self.context_schema),
        }
        if self.query_lang != "en":
            d["query_lang"] = self.query_lang
        if sort_keys:
            d = dict(sorted(d.items()))
        return d


_LENGTH_ORDER = {"short": 0, "medium": 1, "long": 2}


def adjacent_buckets(a: str, b: str) -> bool:
    """Return True when length buckets are the same or adjacent (short↔medium, medium↔long)."""
    ai = _LENGTH_ORDER.get(a)
    bi = _LENGTH_ORDER.get(b)
    if ai is None or bi is None:
        return False
    return abs(ai - bi) <= 1


# ── CandidateResult ────────────────────────────────────────────


@dataclass(slots=True)
class CandidateResult:
    """Output of build_candidates — vector recall + shortcut decision.

    items: list of (keyword, score) tuples, sorted by score desc.
    action: "shortcut_reuse" | "shortcut_new" | "ask_llm"
    """

    items: list[tuple[str, float]]
    action: str  # "shortcut_reuse" | "shortcut_new" | "ask_llm"


# ── Template wrapper ───────────────────────────────────────────


@dataclass(slots=True)
class PlanTemplate:
    """Template stored in apc:tpl:<tpl_id> (Hash)."""

    template_id: str
    version: str  # e.g. "v2.3"
    schema_hash: str  # tools_hash
    ctx_fingerprint: CtxFingerprint
    task: str
    steps: list[dict[str, Any]]
    created_at: float
    ttl_seconds: int

    def is_valid(self, current_version: str = CURRENT_VERSION, tools_hash: str = "") -> bool:
        """Check version, schema_hash, and TTL.

        When tools_hash is "" (caller didn't check yet), skip schema_hash validation.
        """
        import time

        if self.version != current_version:
            return False
        if tools_hash and self.schema_hash != tools_hash:
            return False
        if self.created_at + self.ttl_seconds <= time.time():
            return False
        return True

    @classmethod
    def from_redis_hash(cls, data: dict[bytes, bytes]) -> "PlanTemplate":
        import json

        def _decode(v: bytes) -> str:
            return v.decode("utf-8")

        ctx_raw = json.loads(_decode(data[b"ctx_fingerprint"]))
        ctx_fp = CtxFingerprint(
            context_type=ctx_raw["context_type"],
            length_bucket=ctx_raw["length_bucket"],
            tools=frozenset(ctx_raw["tools"]),
            agent_role=ctx_raw["agent_role"],
            context_schema=frozenset(ctx_raw.get("context_schema", [])),
            query_lang=ctx_raw.get("query_lang", "en"),
        )
        return cls(
            template_id=_decode(data[b"template_id"]),
            version=_decode(data[b"version"]),
            schema_hash=_decode(data[b"schema_hash"]),
            ctx_fingerprint=ctx_fp,
            task=_decode(data[b"task"]),
            steps=json.loads(_decode(data[b"steps"])),
            created_at=float(_decode(data[b"created_at"])),
            ttl_seconds=int(_decode(data[b"ttl_seconds"])),
        )

    def to_redis_hash(self) -> dict[str, str]:
        import json

        return {
            "template_id": self.template_id,
            "version": self.version,
            "schema_hash": self.schema_hash,
            "ctx_fingerprint": json.dumps(self.ctx_fingerprint.to_dict()),
            "task": self.task,
            "steps": json.dumps(self.steps, ensure_ascii=False),
            "created_at": str(self.created_at),
            "ttl_seconds": str(self.ttl_seconds),
        }
