"""LangGraph state graph for the APC Keyword Cache Subsystem.

Blueprint §5: Node list, edge routing, and AgentState schema.

Nodes:
- cache_lookup: All requests enter here. Populates cache_hit, keyword, plan_template.
- small_planner: Adapt a cached template for the current query (cache_hit=True).
- large_planner: Generate a new plan from scratch (cache_hit=False).
- actor: Execute the plan steps.
- template_gen: After large_planner completes, write template to cache layers.

Edges (blueprint §5.2):
  entry → cache_lookup
  cache_lookup → cache_hit=True  → small_planner
  cache_lookup → cache_hit=False → large_planner
  small_planner → is_complete=True  → END
  small_planner → is_complete=False → actor
  large_planner → is_complete=True  → template_gen
  large_planner → is_complete=False → actor
  actor → cache_hit=True  → small_planner
  actor → cache_hit=False → large_planner
  template_gen → END
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from apc_cache.cache.lookup import cache_lookup, lookup_with_keyword
from apc_cache.cache.template_gen import template_gen
from apc_cache.config import APCConfig

logger = logging.getLogger(__name__)

# ── AgentState type (blueprint §5.3) ────────────────────────────

AGENT_STATE_FIELDS = {
    "query": str,
    "context": object,  # Any
    "agent_id": str,
    "tools": list,
    "tools_hash": str,
    "keyword": str | None,
    "cache_hit": bool,
    "cache_hit_layer": str,
    "plan_template": object | None,
    "current_plan": str | None,
    "actor_responses": list,
    "execution_log": list,
    "iteration_count": int,
    "final_output": str | None,
    "is_complete": bool,
}


# ── Node implementations ────────────────────────────────────────


def _make_cache_lookup_node(cfg: APCConfig, r: Redis):
    async def node(state: dict[str, Any]) -> dict[str, Any]:
        return await cache_lookup(state, cfg, r)

    return node


def _make_template_gen_node(cfg: APCConfig, r: Redis):
    async def node(state: dict[str, Any]) -> dict[str, Any]:
        return await template_gen(state, cfg, r)

    return node


def _make_small_planner_node():
    """Placeholder small_planner — adapts a cached template.

    In the full system this uses a local LM (LLaMa-3.1-8B) to adapt
    the template steps to the specific query/context.
    """

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        template = state.get("plan_template")
        query = state.get("query", "")

        if template is not None:
            # Adapt template steps to current query
            steps = template.steps
            plan_text = "\n".join(
                f"{i+1}. {s.get('description', str(s))}" for i, s in enumerate(steps)
            )
            state["current_plan"] = f"# Plan (adapted from template)\n\n{plan_text}"
        else:
            state["current_plan"] = f"# Plan for: {query}\n\n(no template available)"

        state["is_complete"] = True
        state["final_output"] = state["current_plan"]
        return state

    return node


def _make_large_planner_node():
    """Placeholder large_planner — generates a plan from scratch.

    In the full system this calls GPT-4o to produce a full plan
    and records the keyword annotation.
    """

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        keyword = state.get("keyword", "")

        plan = f"# Generated plan for: {query}\n\nKeyword: {keyword}\n\n1. Analyse input\n2. Execute computation\n3. Format output"

        state["current_plan"] = plan
        state["execution_log"] = [
            {"task": query, "action": f"Generated plan with keyword: {keyword}"}
        ]
        state["is_complete"] = True
        state["final_output"] = plan
        return state

    return node


def _make_actor_node():
    """Placeholder actor — executes plan steps.

    In the full system this uses a local LM to run tool calls.
    """

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("current_plan", "")
        responses = state.get("actor_responses", [])

        responses.append(f"Executed: {plan[:100]}...")
        state["actor_responses"] = responses
        state["is_complete"] = True
        return state

    return node


# ── Conditional edge functions ──────────────────────────────────


def _route_after_cache_lookup(state: dict[str, Any]) -> Literal["small_planner", "large_planner"]:
    if state.get("cache_hit", False):
        return "small_planner"
    return "large_planner"


def _route_after_planner(state: dict[str, Any]) -> Literal["actor", "END"]:
    if state.get("is_complete", False):
        return "END"
    return "actor"


def _route_after_large_planner(state: dict[str, Any]) -> Literal["actor", "template_gen"]:
    if state.get("is_complete", False):
        return "template_gen"
    return "actor"


def _route_after_actor(state: dict[str, Any]) -> Literal["small_planner", "large_planner"]:
    if state.get("cache_hit", False):
        return "small_planner"
    return "large_planner"


# ── Graph builder ───────────────────────────────────────────────


def build_graph(cfg: APCConfig, r: Redis) -> CompiledStateGraph:
    """Construct the full LangGraph StateGraph.

    Returns a compiled graph ready to be invoked via .ainvoke().
    """
    graph = StateGraph(dict)

    # Nodes
    graph.add_node("cache_lookup", _make_cache_lookup_node(cfg, r))
    graph.add_node("small_planner", _make_small_planner_node())
    graph.add_node("large_planner", _make_large_planner_node())
    graph.add_node("actor", _make_actor_node())
    graph.add_node("template_gen", _make_template_gen_node(cfg, r))

    # Entry
    graph.set_entry_point("cache_lookup")

    # Conditional edges
    graph.add_conditional_edges(
        "cache_lookup",
        _route_after_cache_lookup,
        {"small_planner": "small_planner", "large_planner": "large_planner"},
    )

    graph.add_conditional_edges(
        "small_planner",
        _route_after_planner,
        {"actor": "actor", "END": END},
    )

    graph.add_conditional_edges(
        "large_planner",
        _route_after_large_planner,
        {"actor": "actor", "template_gen": "template_gen"},
    )

    graph.add_conditional_edges(
        "actor",
        _route_after_actor,
        {"small_planner": "small_planner", "large_planner": "large_planner"},
    )

    # Terminal
    graph.add_edge("template_gen", END)

    return graph.compile()
