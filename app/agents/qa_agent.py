"""
app/agents/qa_agent.py — LangGraph Q&A agent.

Graph:
  START → retrieve → grade_relevance → (relevant) answer
                                     → (irrelevant) rephrase_and_retry → retrieve → answer
         → answer → END

This gives the agent one self-correcting loop before answering.
"""
from __future__ import annotations
from typing import TypedDict, Annotated, Sequence
import operator

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.rag.retriever import retrieve_nodes, build_context_string, nodes_to_source_refs
from app.core.llm_factory import get_resilient_llm
from app.core.observability import observe
from app.models.schemas import QAResponse, SourceNode
from app.core.logging import get_logger

log = get_logger(__name__)
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_resilient_llm()
    return _llm


# ── Graph state ────────────────────────────────────────────────────────────────

class QAState(TypedDict):
    question: str
    doc_ids: list[str] | None
    top_k: int
    session_id: str
    user_id: str | None
    nodes: list[dict]
    context: str
    answer: str
    sources: list[SourceNode]
    retried: bool
    tokens_used: int


# ── Node functions ─────────────────────────────────────────────────────────────

async def _node_retrieve(state: QAState) -> QAState:
    log.info("QA: retrieving", question=state["question"][:80])
    nodes = await retrieve_nodes(
        query=state["question"],
        doc_ids=state["doc_ids"],
        top_k=state["top_k"],
    )
    context = build_context_string(nodes)
    return {**state, "nodes": nodes, "context": context}


async def _node_grade_relevance(state: QAState) -> str:
    """Return 'relevant' or 'rephrase' edge label."""
    if not state["nodes"]:
        return "rephrase"
    # Quick LLM grade: is the context useful for the question?
    llm = _get_llm()
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Retrieved context snippet: {state['context'][:600]}\n\n"
        "Is this context relevant enough to answer the question? "
        "Reply with only 'yes' or 'no'."
    )
    resp = await llm.acomplete(prompt)
    verdict = resp.text.strip().lower()
    log.debug("Relevance grade", verdict=verdict)
    return "relevant" if "yes" in verdict else "rephrase"


async def _node_rephrase(state: QAState) -> QAState:
    """Rephrase the question and re-retrieve once."""
    llm = _get_llm()
    prompt = (
        f"The following question didn't return good results from a document search.\n"
        f"Rephrase it to be more specific and keyword-rich:\n{state['question']}\n\n"
        "Rephrased question (one sentence only):"
    )
    resp = await llm.acomplete(prompt)
    new_q = resp.text.strip()
    log.info("Rephrased question", original=state["question"][:60], rephrased=new_q[:60])

    nodes = await retrieve_nodes(
        query=new_q,
        doc_ids=state["doc_ids"],
        top_k=state["top_k"],
    )
    context = build_context_string(nodes)
    return {**state, "nodes": nodes, "context": context, "retried": True}


async def _node_answer(state: QAState) -> QAState:
    llm = _get_llm()
    context = state["context"]
    question = state["question"]

    system = (
        "You are an expert tutor helping a student understand their study material.\n"
        "Use ONLY the provided context to answer. If the context doesn't contain enough "
        "information, say so honestly. Be clear, concise, and educational.\n"
        "Format your answer with:\n"
        "- Direct answer first\n"
        "- Explanation with key points\n"
        "- Examples if helpful\n"
    )
    prompt = f"{system}\n\nContext:\n{context}\n\nStudent Question: {question}\n\nAnswer:"

    resp = await llm.acomplete(prompt)
    answer = resp.text.strip()
    sources = nodes_to_source_refs(state["nodes"])

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "tokens_used": len(prompt.split()) + len(answer.split()),  # rough estimate
    }


# ── Build graph ────────────────────────────────────────────────────────────────

def _build_qa_graph():
    g = StateGraph(QAState)

    g.add_node("retrieve", _node_retrieve)
    g.add_node("rephrase", _node_rephrase)
    g.add_node("answer", _node_answer)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges(
        "retrieve",
        _node_grade_relevance,
        {"relevant": "answer", "rephrase": "rephrase"},
    )
    g.add_edge("rephrase", "answer")
    g.add_edge("answer", END)

    return g.compile()


_qa_graph = None


def get_qa_graph():
    global _qa_graph
    if _qa_graph is None:
        _qa_graph = _build_qa_graph()
    return _qa_graph


# ── Public API ─────────────────────────────────────────────────────────────────

@observe(name="qa_agent")
async def run_qa(
    question: str,
    doc_ids: list[str] | None = None,
    top_k: int = 5,
    session_id: str = "",
    user_id: str | None = None,
) -> QAResponse:
    graph = get_qa_graph()
    state: QAState = {
        "question": question,
        "doc_ids": doc_ids,
        "top_k": top_k,
        "session_id": session_id,
        "user_id": user_id,
        "nodes": [],
        "context": "",
        "answer": "",
        "sources": [],
        "retried": False,
        "tokens_used": 0,
    }
    final_state = await graph.ainvoke(state)
    return QAResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        session_id=session_id,
        tokens_used=final_state.get("tokens_used"),
    )
