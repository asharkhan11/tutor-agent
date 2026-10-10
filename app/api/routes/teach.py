"""
app/api/routes/teach.py — Concept teaching/explanation endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import TeachRequest, TeachResponse
from app.agents.teach_agent import run_teach
from app.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/teach", tags=["Teaching"])


@router.post("", response_model=TeachResponse)
async def teach_concept(request: TeachRequest):
    """
    Explain a concept pulled from your ingested documents.

    The agent:
    1. Retrieves relevant passages about the concept
    2. Generates a structured explanation at the requested detail level
    3. Enriches with key points, examples, analogies, and related concepts

    `detail_level`: `brief` | `medium` | `detailed`
    """
    try:
        return await run_teach(request)
    except Exception as exc:
        log.error("Teach agent failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Teach error: {str(exc)}")
