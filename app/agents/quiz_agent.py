"""
app/agents/quiz_agent.py — Quiz & test-paper generation via LangGraph.

Graph:
  START → retrieve_content → plan_questions → generate_questions → validate → END

Each question type (MCQ / short / long / true-false / fill-blank) has its own
prompt template. The validate node removes malformed questions.
"""
from __future__ import annotations
import json
import re
import uuid
from typing import TypedDict

from langgraph.graph import StateGraph, END, START

from app.rag.retriever import retrieve_nodes, build_context_string
from app.core.llm_factory import get_resilient_llm
from app.core.observability import observe
from app.models.schemas import (
    QuizRequest, QuizResponse, QuizQuestion, QuizQuestion,
    MCQOption, QuestionType, DifficultyLevel, TestPaperRequest,
    TestPaperResponse, TestSection,
)
from app.core.logging import get_logger

log = get_logger(__name__)


# ── Prompt templates ───────────────────────────────────────────────────────────

_MCQ_TMPL = """
Generate {n} multiple-choice questions (MCQ) at {difficulty} difficulty from the context below.

Context:
{context}

Topic filter: {topic}

For EACH question output STRICT JSON:
{{
  "type": "mcq",
  "question": "...",
  "options": [{{"key": "A", "text": "..."}}, {{"key": "B", "text": "..."}}, {{"key": "C", "text": "..."}}, {{"key": "D", "text": "..."}}],
  "answer": "A",
  "explanation": "...",
  "difficulty": "{difficulty}"
}}

Output a JSON array of objects. Nothing else.
"""

_SHORT_TMPL = """
Generate {n} short-answer questions at {difficulty} difficulty from the context below.

Context:
{context}

Topic filter: {topic}

For EACH question output STRICT JSON:
{{
  "type": "short",
  "question": "...",
  "options": null,
  "answer": "2-3 sentence expected answer",
  "explanation": "Why this answer is correct",
  "difficulty": "{difficulty}"
}}

Output a JSON array. Nothing else.
"""

_LONG_TMPL = """
Generate {n} long-answer / essay questions at {difficulty} difficulty from the context below.

Context:
{context}

Topic filter: {topic}

For EACH question output STRICT JSON:
{{
  "type": "long",
  "question": "...",
  "options": null,
  "answer": "Comprehensive model answer (4-6 sentences)",
  "explanation": "Key points that should be covered",
  "difficulty": "{difficulty}"
}}

Output a JSON array. Nothing else.
"""

_TF_TMPL = """
Generate {n} true/false questions at {difficulty} difficulty from the context below.

Context:
{context}

Topic filter: {topic}

For EACH question output STRICT JSON:
{{
  "type": "true_false",
  "question": "Statement that is either true or false",
  "options": [{{"key": "A", "text": "True"}}, {{"key": "B", "text": "False"}}],
  "answer": "A",
  "explanation": "...",
  "difficulty": "{difficulty}"
}}

Output a JSON array. Nothing else.
"""

_FILL_TMPL = """
Generate {n} fill-in-the-blank questions at {difficulty} difficulty from the context below.
Use ___ to mark the blank.

Context:
{context}

Topic filter: {topic}

For EACH question output STRICT JSON:
{{
  "type": "fill_blank",
  "question": "The ___ is responsible for ...",
  "options": null,
  "answer": "word or phrase that fills the blank",
  "explanation": "...",
  "difficulty": "{difficulty}"
}}

Output a JSON array. Nothing else.
"""

_TMPL_MAP = {
    QuestionType.MCQ: _MCQ_TMPL,
    QuestionType.SHORT: _SHORT_TMPL,
    QuestionType.LONG: _LONG_TMPL,
    QuestionType.TRUE_FALSE: _TF_TMPL,
    QuestionType.FILL_BLANK: _FILL_TMPL,
}


# ── Graph state ────────────────────────────────────────────────────────────────

class QuizState(TypedDict):
    request: dict          # serialised QuizRequest
    context: str
    raw_questions: list[dict]
    valid_questions: list[QuizQuestion]


# ── Node functions ─────────────────────────────────────────────────────────────

async def _node_retrieve_content(state: QuizState) -> QuizState:
    req = state["request"]
    query = req.get("topic") or req.get("chapter") or "key concepts"
    nodes = await retrieve_nodes(
        query=query,
        doc_ids=req["doc_ids"],
        top_k=15,  # fetch more for quiz context
    )
    context = build_context_string(nodes)
    return {**state, "context": context}


async def _node_generate_questions(state: QuizState) -> QuizState:
    req = state["request"]
    llm = get_resilient_llm()
    context = state["context"]
    topic = req.get("topic") or req.get("chapter") or "all topics"
    difficulty = req.get("difficulty", "mixed")
    q_types: list[str] = req.get("question_types", ["mcq"])
    total_n: int = req.get("num_questions", 10)

    # Distribute questions across types
    per_type = max(1, total_n // len(q_types))
    remainder = total_n - per_type * len(q_types)

    all_raw: list[dict] = []
    for i, qt in enumerate(q_types):
        n = per_type + (1 if i < remainder else 0)
        eff_difficulty = difficulty if difficulty != "mixed" else ["easy", "medium", "hard"][i % 3]
        tmpl = _TMPL_MAP.get(QuestionType(qt), _MCQ_TMPL)
        prompt = tmpl.format(
            n=n,
            difficulty=eff_difficulty,
            context=context[:3000],
            topic=topic,
        )
        try:
            resp = await llm.acomplete(prompt)
            parsed = _safe_parse_json(resp.text)
            if isinstance(parsed, list):
                all_raw.extend(parsed)
        except Exception as exc:
            log.warning("Question generation failed for type", type=qt, error=str(exc))

    return {**state, "raw_questions": all_raw}


async def _node_validate(state: QuizState) -> QuizState:
    valid: list[QuizQuestion] = []
    for raw in state["raw_questions"]:
        try:
            q = QuizQuestion(
                type=QuestionType(raw.get("type", "mcq")),
                question=str(raw["question"]),
                options=[MCQOption(**o) for o in raw["options"]] if raw.get("options") else None,
                answer=str(raw["answer"]),
                explanation=str(raw.get("explanation", "")),
                difficulty=DifficultyLevel(raw.get("difficulty", "medium")),
            )
            valid.append(q)
        except Exception as exc:
            log.debug("Skipping invalid question", error=str(exc))
    return {**state, "valid_questions": valid}


# ── Build graph ────────────────────────────────────────────────────────────────

def _build_quiz_graph():
    g = StateGraph(QuizState)
    g.add_node("retrieve_content", _node_retrieve_content)
    g.add_node("generate_questions", _node_generate_questions)
    g.add_node("validate", _node_validate)
    g.add_edge(START, "retrieve_content")
    g.add_edge("retrieve_content", "generate_questions")
    g.add_edge("generate_questions", "validate")
    g.add_edge("validate", END)
    return g.compile()


_quiz_graph = None


def get_quiz_graph():
    global _quiz_graph
    if _quiz_graph is None:
        _quiz_graph = _build_quiz_graph()
    return _quiz_graph


# ── Public API ─────────────────────────────────────────────────────────────────

@observe(name="quiz_agent")
async def run_quiz(request: QuizRequest) -> QuizResponse:
    graph = get_quiz_graph()
    state: QuizState = {
        "request": request.model_dump(),
        "context": "",
        "raw_questions": [],
        "valid_questions": [],
    }
    final = await graph.ainvoke(state)
    title_parts = [request.topic or request.chapter or "Study Material"]
    title = f"Quiz: {', '.join(title_parts)}"
    return QuizResponse(
        title=title,
        questions=final["valid_questions"],
        session_id=request.session_id,
    )


@observe(name="test_paper_agent")
async def run_test_paper(request: TestPaperRequest) -> TestPaperResponse:
    """
    Generate a structured test paper with multiple sections.
    Default sections if not specified:
      Section A — MCQ (40%)
      Section B — Short answer (30%)
      Section C — Long answer (30%)
    """
    default_sections = request.sections or [
        {"type": "mcq",   "count": 10, "marks_each": 2},
        {"type": "short", "count": 5,  "marks_each": 4},
        {"type": "long",  "count": 3,  "marks_each": 10},
    ]

    graph = get_quiz_graph()
    test_sections: list[TestSection] = []

    for sec_def in default_sections:
        q_type = sec_def["type"]
        count = sec_def["count"]
        marks_each = sec_def["marks_each"]

        quiz_req = QuizRequest(
            doc_ids=request.doc_ids,
            topic=None,
            chapter=",".join(request.chapters) if request.chapters else None,
            num_questions=count,
            question_types=[QuestionType(q_type)],
            difficulty=request.difficulty,
            session_id=request.session_id,
        )
        state: QuizState = {
            "request": quiz_req.model_dump(),
            "context": "",
            "raw_questions": [],
            "valid_questions": [],
        }
        final = await graph.ainvoke(state)
        qs = final["valid_questions"]

        section_titles = {
            "mcq": "Section A — Multiple Choice Questions",
            "short": "Section B — Short Answer Questions",
            "long": "Section C — Long Answer Questions",
            "true_false": "Section D — True / False",
            "fill_blank": "Section E — Fill in the Blanks",
        }
        instructions_map = {
            "mcq": "Choose the best answer. Each question carries equal marks.",
            "short": "Answer in 2–4 sentences.",
            "long": "Answer in detail. Write structured paragraphs.",
            "true_false": "Mark True (T) or False (F).",
            "fill_blank": "Fill in the blanks with appropriate words.",
        }

        test_sections.append(
            TestSection(
                title=section_titles.get(q_type, f"Section — {q_type.upper()}"),
                instructions=instructions_map.get(q_type, ""),
                questions=qs,
                marks_per_question=marks_each,
                total_marks=len(qs) * marks_each,
            )
        )

    actual_total = sum(s.total_marks for s in test_sections)
    return TestPaperResponse(
        title=request.title,
        duration_minutes=request.duration_minutes,
        total_marks=actual_total,
        sections=test_sections,
        session_id=request.session_id,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_parse_json(text: str) -> list | dict:
    """Strip markdown fences and parse JSON."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)
