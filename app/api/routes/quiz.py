"""
app/api/routes/quiz.py — Quiz generation endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import QuizRequest, QuizResponse, TestPaperRequest, TestPaperResponse
from app.agents.quiz_agent import run_quiz, run_test_paper
from app.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/quiz", tags=["Quiz & Test"])


@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    """
    Generate a quiz from one or more documents.

    - Supports MCQ, short-answer, long-answer, true/false, fill-in-blank
    - Set `topic` or `chapter` to narrow down the content
    - `difficulty`: easy | medium | hard | mixed
    """
    try:
        return await run_quiz(request)
    except Exception as exc:
        log.error("Quiz generation failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Quiz error: {str(exc)}")


@router.post("/test-paper", response_model=TestPaperResponse)
async def generate_test_paper(request: TestPaperRequest):
    """
    Generate a full structured test paper with multiple sections.

    Default structure (if `sections` not specified):
    - Section A: 10 MCQs × 2 marks
    - Section B: 5 Short answers × 4 marks  
    - Section C: 3 Long answers × 10 marks

    Custom sections example:
    ```json
    "sections": [
      {"type": "mcq",        "count": 20, "marks_each": 1},
      {"type": "short",      "count": 5,  "marks_each": 4},
      {"type": "fill_blank", "count": 5,  "marks_each": 2}
    ]
    ```
    """
    try:
        return await run_test_paper(request)
    except Exception as exc:
        log.error("Test paper generation failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Test paper error: {str(exc)}")
