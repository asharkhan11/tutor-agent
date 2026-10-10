"""
app/agents/teach_agent.py — Concept teaching agent.

Graph:
  START → retrieve_concept → explain → enrich → END

The agent retrieves relevant content, generates a structured explanation,
then enriches it with examples, analogies, and related concepts.
"""
from __future__ import annotations
from typing import TypedDict

from langgraph.graph import StateGraph, END, START

from app.rag.retriever import retrieve_nodes, build_context_string, nodes_to_source_refs
from app.core.llm_factory import get_resilient_llm
from app.core.observability import observe
from app.models.schemas import TeachRequest, TeachResponse, SourceNode
from app.core.logging import get_logger

log = get_logger(__name__)

# ── Graph state ────────────────────────────────────────────────────────────────

class TeachState(TypedDict):
    concept: str
    doc_ids: list[str] | None
    detail_level: str
    include_examples: bool
    include_analogies: bool
    session_id: str
    user_id: str | None
    context: str
    sources: list[SourceNode]
    explanation: str
    key_points: list[str]
    examples: list[str]
    analogies: list[str]
    related_concepts: list[str]


# ── Node functions ─────────────────────────────────────────────────────────────

async def _node_retrieve_concept(state: TeachState) -> TeachState:
    nodes = await retrieve_nodes(
        query=state["concept"],
        doc_ids=state["doc_ids"],
        top_k=8,
    )
    context = build_context_string(nodes)
    sources = nodes_to_source_refs(nodes)
    return {**state, "context": context, "sources": sources}


async def _node_explain(state: TeachState) -> TeachState:
    llm = get_resilient_llm()
    level_desc = {
        "brief": "a concise 3-4 sentence overview",
        "medium": "a clear explanation with key points (150-250 words)",
        "detailed": "a thorough explanation with depth and nuance (300-500 words)",
    }.get(state["detail_level"], "a clear explanation")

    prompt = f"""You are a brilliant tutor. Explain the concept to a student using the context provided.

Concept: {state["concept"]}

Source material:
{state["context"][:3000]}

Instructions:
- Provide {level_desc}
- Start with the core definition or idea
- Build up to deeper understanding
- Use clear, accessible language
- Structure your answer well

Explanation:"""

    resp = await llm.acomplete(prompt)
    explanation = resp.text.strip()
    return {**state, "explanation": explanation}


async def _node_enrich(state: TeachState) -> TeachState:
    """Extract key points, examples, analogies, and related concepts."""
    llm = get_resilient_llm()
    explanation = state["explanation"]
    concept = state["concept"]

    examples_hint = "['2-3 concrete examples from real world or the text']" if state["include_examples"] else "[]"
    analogies_hint = "['1-2 simple analogies to everyday things']" if state["include_analogies"] else "[]"

    prompt = f"""Based on this explanation of "{concept}":

{explanation}

Extract and generate (respond in JSON only, no markdown):
{{
  "key_points": ["3-5 bullet-point takeaways"],
  "examples": {examples_hint},
  "analogies": {analogies_hint},
  "related_concepts": ["3-5 related topics the student should also study"]
}}"""

    resp = await llm.acomplete(prompt)
    try:
        import json, re
        text = re.sub(r"```(?:json)?", "", resp.text).strip().rstrip("`").strip()
        data = json.loads(text)
    except Exception:
        data = {}

    return {
        **state,
        "key_points": data.get("key_points", []),
        "examples": data.get("examples", []) if state["include_examples"] else [],
        "analogies": data.get("analogies", []) if state["include_analogies"] else [],
        "related_concepts": data.get("related_concepts", []),
    }


# ── Build graph ────────────────────────────────────────────────────────────────

def _build_teach_graph():
    g = StateGraph(TeachState)
    g.add_node("retrieve_concept", _node_retrieve_concept)
    g.add_node("explain", _node_explain)
    g.add_node("enrich", _node_enrich)
    g.add_edge(START, "retrieve_concept")
    g.add_edge("retrieve_concept", "explain")
    g.add_edge("explain", "enrich")
    g.add_edge("enrich", END)
    return g.compile()


_teach_graph = None


def get_teach_graph():
    global _teach_graph
    if _teach_graph is None:
        _teach_graph = _build_teach_graph()
    return _teach_graph


# ── Public API ─────────────────────────────────────────────────────────────────

@observe(name="teach_agent")
async def run_teach(request: TeachRequest) -> TeachResponse:
    graph = get_teach_graph()
    state: TeachState = {
        "concept": request.concept,
        "doc_ids": request.doc_ids,
        "detail_level": request.detail_level,
        "include_examples": request.include_examples,
        "include_analogies": request.include_analogies,
        "session_id": request.session_id,
        "user_id": request.user_id,
        "context": "",
        "sources": [],
        "explanation": "",
        "key_points": [],
        "examples": [],
        "analogies": [],
        "related_concepts": [],
    }
    final = await graph.ainvoke(state)
    return TeachResponse(
        concept=request.concept,
        explanation=final["explanation"],
        key_points=final["key_points"],
        examples=final["examples"],
        analogies=final["analogies"],
        related_concepts=final["related_concepts"],
        sources=final["sources"],
        session_id=request.session_id,
    )
