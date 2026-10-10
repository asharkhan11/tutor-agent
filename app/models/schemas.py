"""
app/models/schemas.py — All request / response Pydantic models.
"""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid


# ── Shared ─────────────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    MCQ = "mcq"
    SHORT = "short"
    LONG = "long"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    MIXED = "mixed"


# ── Document Ingestion ─────────────────────────────────────────────────────────

class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    subject: str | None = None
    chapter: str | None = None
    tags: list[str] = []
    page_count: int = 0
    chunk_count: int = 0
    has_images: bool = False
    status: str = "pending"   # pending | processing | ready | failed
    error: str | None = None


class IngestResponse(BaseModel):
    doc_id: str
    message: str
    status: str


# ── Q&A ───────────────────────────────────────────────────────────────────────

class QARequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    doc_ids: list[str] | None = None       # None → search all docs
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False

class SourceNode(BaseModel):
    doc_id: str
    filename: str
    page: int | None = None
    score: float
    text: str

class QAResponse(BaseModel):
    answer: str
    sources: list[SourceNode] = []
    session_id: str
    tokens_used: int | None = None


# ── Quiz Generation ────────────────────────────────────────────────────────────

class QuizRequest(BaseModel):
    doc_ids: list[str]
    topic: str | None = None              # optional topic filter within doc
    chapter: str | None = None
    num_questions: int = Field(default=10, ge=1, le=50)
    question_types: list[QuestionType] = [QuestionType.MCQ, QuestionType.SHORT]
    difficulty: DifficultyLevel = DifficultyLevel.MIXED
    user_id: str | None = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class MCQOption(BaseModel):
    key: str      # A, B, C, D
    text: str


class QuizQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: QuestionType
    question: str
    options: list[MCQOption] | None = None   # Only for MCQ / True-False
    answer: str
    explanation: str
    difficulty: DifficultyLevel
    source_chunk: str | None = None


class QuizResponse(BaseModel):
    quiz_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    questions: list[QuizQuestion]
    session_id: str


# ── Test Paper ─────────────────────────────────────────────────────────────────

class TestPaperRequest(BaseModel):
    doc_ids: list[str]
    title: str = "Test Paper"
    chapters: list[str] | None = None
    total_marks: int = Field(default=100, ge=10, le=500)
    duration_minutes: int = Field(default=60, ge=10, le=180)
    sections: list[dict] | None = None    # [{type, count, marks_each}]
    difficulty: DifficultyLevel = DifficultyLevel.MIXED
    user_id: str | None = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class TestSection(BaseModel):
    title: str
    instructions: str
    questions: list[QuizQuestion]
    marks_per_question: int
    total_marks: int


class TestPaperResponse(BaseModel):
    paper_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    duration_minutes: int
    total_marks: int
    sections: list[TestSection]
    session_id: str


# ── Teaching / Explanation ─────────────────────────────────────────────────────

class TeachRequest(BaseModel):
    concept: str = Field(..., min_length=2, max_length=500)
    doc_ids: list[str] | None = None
    detail_level: str = "medium"    # brief | medium | detailed
    include_examples: bool = True
    include_analogies: bool = True
    user_id: str | None = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class TeachResponse(BaseModel):
    concept: str
    explanation: str
    key_points: list[str]
    examples: list[str]
    analogies: list[str]
    related_concepts: list[str]
    sources: list[SourceNode] = []
    session_id: str


# ── Document listing ───────────────────────────────────────────────────────────

class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]
    total: int


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    qdrant: str
    llm_provider: str
    embed_model: str
    langfuse: str
