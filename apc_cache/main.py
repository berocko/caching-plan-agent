"""FastAPI application entry point for the APC Keyword Cache Subsystem.

Blueprint §6: Lifespan management, graph mounting, and health endpoints.
Blueprint §1: POST /agent/run as the main entry point.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from apc_cache.config import APCConfig
from apc_cache.graph import build_graph
from apc_cache.keyword.embedding import init_embedding, shutdown_embedding
from apc_cache.keyword.keyword_index import KeywordIndexManager, set_index_manager
from apc_cache.metrics import embed_model_version

logger = logging.getLogger(__name__)

# ── Global state ─────────────────────────────────────────────────

cfg: APCConfig | None = None
redis_client: redis.Redis | None = None
graph: Any | None = None
kw_index_mgr: KeywordIndexManager | None = None


# ── Lifespan (blueprint §6) ──────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cfg, redis_client, graph, kw_index_mgr

    # ── Startup (blueprint §6.1) ──────────────────────────
    cfg = APCConfig.from_env()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info("APC Keyword Cache starting (model_ver=%s)", cfg.embed_model_ver)

    # 1. Redis connection pool
    redis_client = redis.from_url(cfg.redis_url, decode_responses=False)

    # 2. Embedding model
    init_embedding(cfg.embed_model_name, cfg.embed_model_ver)
    embed_model_version.labels(model_ver=cfg.embed_model_ver).set(1)

    # 3. Keyword index manager → full reload + periodic sync
    kw_index_mgr = KeywordIndexManager(cfg, redis_client)
    await kw_index_mgr.start()
    set_index_manager(kw_index_mgr)

    # 4. Check model_ver consistency
    from apc_cache.ops.reindex import check_model_ver_consistency

    ver_dist = await check_model_ver_consistency(cfg, redis_client)
    total_kw = sum(ver_dist.values())
    if total_kw > 0:
        current = ver_dist.get(cfg.embed_model_ver, 0)
        ratio = 1.0 - (current / total_kw)
        if ratio > 0.10:
            logger.warning(
                "Model version skew: %.1f%% keywords not on current ver %s. Dist: %s",
                ratio * 100,
                cfg.embed_model_ver,
                ver_dist,
            )

    # 5. Build LangGraph graph
    graph = build_graph(cfg, redis_client)

    logger.info("APC Keyword Cache started successfully")

    yield  # ── App running ───────────────────────────────

    # ── Shutdown (blueprint §6.2) ─────────────────────────
    logger.info("APC Keyword Cache shutting down")

    # 1. Stop keyword index sync
    if kw_index_mgr:
        await kw_index_mgr.stop()

    # 2. Close Redis
    if redis_client:
        await redis_client.aclose()

    # 3. Shutdown embedding executor
    shutdown_embedding()

    logger.info("APC Keyword Cache stopped")


# ── FastAPI app ──────────────────────────────────────────────────

app = FastAPI(
    title="APC Keyword Cache Subsystem",
    version="2.3.0",
    lifespan=lifespan,
)


# ── Request/Response models ──────────────────────────────────────


class ToolDef(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    query: str
    context: Any = None
    agent_id: str = "default"
    tools: list[ToolDef] = Field(default_factory=list)
    tools_hash: str = ""


class AgentRunResponse(BaseModel):
    cache_hit: bool
    cache_hit_layer: str | None = None
    keyword: str | None = None
    final_output: str | None = None
    iteration_count: int = 0


# ── Routes (blueprint §1) ──────────────────────────────────────


@app.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(req: AgentRunRequest) -> AgentRunResponse:
    """Main entry point: execute the agent with caching.

    Fills agent_id, tools, tools_hash into the AgentState,
    then runs the LangGraph graph.
    """
    assert graph is not None, "Graph not initialised"
    assert cfg is not None, "Config not initialised"

    tools_hash = req.tools_hash or _compute_tools_hash(req.tools)

    initial_state: dict[str, Any] = {
        "query": req.query,
        "context": req.context,
        "agent_id": req.agent_id,
        "tools": req.tools,
        "tools_hash": tools_hash,
        "keyword": None,
        "cache_hit": False,
        "cache_hit_layer": "",
        "plan_template": None,
        "current_plan": None,
        "actor_responses": [],
        "execution_log": [],
        "iteration_count": 0,
        "final_output": None,
        "is_complete": False,
    }

    result = await graph.ainvoke(initial_state)

    return AgentRunResponse(
        cache_hit=result.get("cache_hit", False),
        cache_hit_layer=result.get("cache_hit_layer"),
        keyword=result.get("keyword"),
        final_output=result.get("final_output"),
        iteration_count=result.get("iteration_count", 0),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics/info")
async def metrics_info() -> dict[str, Any]:
    """Expose cache subsystem info for debugging."""
    assert cfg is not None
    assert kw_index_mgr is not None

    return {
        "model_ver": cfg.embed_model_ver,
        "model_name": cfg.embed_model_name,
        "kw_index_size": kw_index_mgr.size,
        "cache_max_size": cfg.cache_max_size,
        "last_sync_ts": kw_index_mgr.last_sync_ts,
    }


# ── helpers ──────────────────────────────────────────────────────


def _compute_tools_hash(tools: list[Any]) -> str:
    import hashlib
    import json

    names = sorted(t.name for t in tools)
    raw = json.dumps(names, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
