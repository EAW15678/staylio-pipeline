"""
TS-17 — LangGraph Pipeline Graph
Agent Orchestration — All 7 Agents

Defines the complete directed execution graph for the Staylio intake pipeline.

EXECUTION ORDER:
  Sequential:  Agent 1 → (Agents 2+3+4 parallel) → Agent 5 → Agent 6 → Agent 7
  Parallel:    Agents 2, 3, and 4 run concurrently after Agent 1 completes

DEPENDENCY RULES (from TS-17):
  - Agent 1 must complete before any other agent (all agents need the KB)
  - Agents 2, 3, 4 run in parallel — they are independent of each other
  - Agent 5 waits for Agents 2 AND 3 (needs content + enhanced photos)
  - Agent 5 does NOT wait for Agent 4 — local guide is loaded at page build time
  - Agent 6 waits for Agent 5 (needs live page URL for UTM links)
  - Agent 7 runs after Agent 5 (page must be deployed for tracking to start)
    Agent 7 does NOT block Agent 6

HOSTING (from TS-17):
  LangGraph runs as a persistent Python service on Railway or EC2.
  NOT inside Vercel — serverless 300s limit insufficient for full pipeline.
  The intake portal fires a thin POST trigger and returns immediately.
  Pipeline processes asynchronously and writes status to PostgreSQL.
  Dashboard polls PostgreSQL every 30 seconds for progress.

FAILURE RECOVERY:
  - Agent-level failures are non-fatal where possible (page goes live without
    perfect social content; local guide is empty rather than blocking page)
  - Pipeline status is written at every step so AMs can see exactly where
    a failure occurred
  - Retry policy: Agents 2 and 3 retry once automatically on failure
    (LLM rate limits, transient API errors)
  - Human escalation: Agent 2 quality gate failure routes to AM review queue
    Agent 3 provenance flag routes to AM review queue

FIX NOTES (v6 → v7):
  InvalidUpdateError fix — when Agents 2, 3, and 4 run in parallel they each
  return a state dict. LangGraph merges all three dicts and raises
  InvalidUpdateError if any key receives more than one value in the same step.
  Fix: annotate every PipelineState field with a reducer function.
  - Scalar fields (str, bool, dict): _last() — last-write-wins
  - List fields (errors, meta_campaigns): _concat() — accumulate across agents
"""

import logging
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END

from agents.agent1.agent import agent1_node
from agents.agent2.agent import agent2_node
from agents.agent3.agent import agent3_node
from agents.agent4.agent import agent4_node
from agents.agent5.agent import agent5_node
from agents.agent6.agent import agent6_node
from agents.agent7.agent import agent7_node

logger = logging.getLogger(__name__)


# ── Reducer helpers ───────────────────────────────────────────────────────────
# LangGraph requires every field that more than one parallel node might write
# to be annotated with a reducer.  We use two simple reducers:
#
#   _last    — last-write-wins (correct for scalars: str, bool, dict)
#   _concat  — list concatenation (correct for error/event accumulator lists)

def _last(a, b):
    """Return the most-recent value.  Used for scalar and dict fields."""
    return b


def _concat(a, b):
    """Concatenate two lists.  None-safe.  Used for errors and meta_campaigns."""
    return (a or []) + (b or [])


# ── Pipeline State ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """
    Shared state passed between all agents in the pipeline.
    LangGraph manages checkpointing and recovery of this state.

    Every field is annotated with a reducer so that LangGraph can merge
    the concurrent updates produced by the parallel fan-out of Agents 2/3/4
    without raising InvalidUpdateError.
    """
    # Identity
    property_id:          Annotated[str,   _last]
    client_id:            Annotated[str,   _last]

    # Agent 1 outputs
    knowledge_base:       Annotated[dict,  _last]
    knowledge_base_ready: Annotated[bool,  _last]
    agent1_complete:      Annotated[bool,  _last]

    # Parallel agent signals
    agent2_ready:         Annotated[bool,  _last]
    agent3_ready:         Annotated[bool,  _last]
    agent4_ready:         Annotated[bool,  _last]

    # Agent 2 outputs
    content_package:      Annotated[dict,  _last]
    agent2_complete:      Annotated[bool,  _last]
    agent2_needs_review:  Annotated[bool,  _last]

    # Agent 3 outputs
    visual_media_package: Annotated[dict,  _last]
    agent3_complete:      Annotated[bool,  _last]

    # Agent 4 outputs
    local_guide:          Annotated[dict,  _last]
    agent4_complete:      Annotated[bool,  _last]

    # Agent 5 outputs
    page_url:             Annotated[str,   _last]
    page_slug:            Annotated[str,   _last]
    agent5_complete:      Annotated[bool,  _last]
    agent6_ready:         Annotated[bool,  _last]

    # Agent 6 outputs
    social_calendar:      Annotated[dict,  _last]
    meta_campaigns:       Annotated[list,  _concat]
    spark_cluster_id:     Annotated[str,   _last]
    agent6_complete:      Annotated[bool,  _last]
    agent7_ready:         Annotated[bool,  _last]

    # Agent 7 outputs
    attribution_tier:     Annotated[str,   _last]
    pixel_snippet:        Annotated[str,   _last]
    agent7_complete:      Annotated[bool,  _last]
    pipeline_complete:    Annotated[bool,  _last]

    # Cross-agent — errors from any agent are accumulated
    errors:               Annotated[list,  _concat]


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_agent1(state: PipelineState) -> str:
    """After Agent 1: if successful, fan out to parallel agents."""
    if not state.get("agent1_complete"):
        logger.error(f"[Pipeline] Agent 1 failed for property {state['property_id']}")
        return "pipeline_failed"
    return "parallel_agents"


def route_after_parallel(state: PipelineState) -> str:
    """
    After Agents 2+3+4: check readiness for Agent 5.
    Agent 5 requires Agent 2 (content) AND Agent 3 (photos).
    Agent 4 (local guide) is loaded at page build time — not blocking.
    """
    if not state.get("agent2_complete") and not state.get("agent2_needs_review"):
        logger.warning(f"[Pipeline] Agent 2 incomplete — page will launch without content")
    if not state.get("agent3_complete"):
        logger.warning(f"[Pipeline] Agent 3 incomplete — page will launch without enhanced photos")
    return "agent5"


def route_after_agent5(state: PipelineState) -> str:
    """After Agent 5: if page deployed, launch Agent 6 and Agent 7 in parallel."""
    if not state.get("agent5_complete"):
        logger.error(f"[Pipeline] Agent 5 failed — page not deployed for {state['property_id']}")
        return "pipeline_failed"
    return "post_launch_agents"


# ── Graph Construction ────────────────────────────────────────────────────────

def build_pipeline_graph() -> StateGraph:
    """
    Build and compile the complete LangGraph pipeline.
    Returns a compiled graph ready for execution.
    """
    graph = StateGraph(PipelineState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("agent3", agent3_node)
    graph.add_node("agent4", agent4_node)
    graph.add_node("agent5", agent5_node)
    graph.add_node("agent6", agent6_node)
    graph.add_node("agent7", agent7_node)
    graph.add_node("pipeline_failed", _handle_pipeline_failure)

    # ── Entry point ────────────────────────────────────────────────────────
    graph.set_entry_point("agent1")

    # ── Agent 1 → parallel fan-out ────────────────────────────────────────
    graph.add_conditional_edges(
        "agent1",
        route_after_agent1,
        {
            "parallel_agents": "agent2",
            "pipeline_failed": "pipeline_failed",
        },
    )

    graph.add_edge("agent1", "agent3")
    graph.add_edge("agent1", "agent4")

    # ── Parallel agents → Agent 5 ─────────────────────────────────────────
    graph.add_edge("agent2", "agent5")
    graph.add_edge("agent3", "agent5")
    graph.add_edge("agent4", "agent5")

    # ── Agent 5 → post-launch agents ──────────────────────────────────────
    graph.add_conditional_edges(
        "agent5",
        route_after_agent5,
        {
            "post_launch_agents": "agent6",
            "pipeline_failed": "pipeline_failed",
        },
    )

    graph.add_edge("agent5", "agent7")

    # ── Terminal edges ────────────────────────────────────────────────────
    graph.add_edge("agent6", END)
    graph.add_edge("agent7", END)
    graph.add_edge("pipeline_failed", END)

    return graph.compile()


def _handle_pipeline_failure(state: PipelineState) -> PipelineState:
    """Terminal node for unrecoverable pipeline failures."""
    property_id = state.get("property_id", "unknown")
    errors = state.get("errors", [])
    logger.error(
        f"[Pipeline] Pipeline failed for property {property_id}. "
        f"Errors: {errors}"
    )
    return {**state, "pipeline_complete": False}


# ── Pipeline execution entry point ────────────────────────────────────────────

def run_intake_pipeline(property_id: str, client_id: str) -> dict:
    """
    Execute the full 7-agent pipeline for a new property intake.

    Args:
        property_id: UUID of the new property (from intake submission)
        client_id:   UUID of the PMC or IO client

    Returns:
        Final pipeline state dict
    """
    logger.info(f"[Pipeline] Starting intake pipeline for property {property_id}")

    initial_state: PipelineState = {
        "property_id": property_id,
        "client_id": client_id,
        "knowledge_base": {},
        "knowledge_base_ready": False,
        "agent1_complete": False,
        "agent2_ready": False,
        "agent3_ready": False,
        "agent4_ready": False,
        "content_package": {},
        "agent2_complete": False,
        "agent2_needs_review": False,
        "visual_media_package": {},
        "agent3_complete": False,
        "local_guide": {},
        "agent4_complete": False,
        "page_url": "",
        "page_slug": "",
        "agent5_complete": False,
        "agent6_ready": False,
        "social_calendar": {},
        "meta_campaigns": [],
        "spark_cluster_id": "",
        "agent6_complete": False,
        "agent7_ready": False,
        "attribution_tier": "",
        "pixel_snippet": "",
        "agent7_complete": False,
        "pipeline_complete": False,
        "errors": [],
    }

    pipeline = build_pipeline_graph()
    final_state = pipeline.invoke(initial_state)

    if final_state.get("pipeline_complete"):
        logger.info(
            f"[Pipeline] Complete for property {property_id}. "
            f"Page: {final_state.get('page_url')}. "
            f"Errors: {len(final_state.get('errors', []))}"
        )
    else:
        logger.error(f"[Pipeline] Did not complete for property {property_id}")

    return final_state
