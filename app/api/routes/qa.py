"""
app/api/routes/qa.py — Question & Answer endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import json

from app.models.schemas import QARequest, QAResponse
from app.agents.qa_agent import run_qa
from app.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/qa", tags=["Q&A"])


@router.post("", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    Ask a question over your ingested documents.

    - `doc_ids`: list of document IDs to search (omit to search all)
    - `top_k`: number of source chunks to retrieve (1-20)
    - `session_id`: pass the same ID across turns for conversation tracking
    """
    try:
        response = await run_qa(
            question=request.question,
            doc_ids=request.doc_ids,
            top_k=request.top_k,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        return response
    except Exception as exc:
        log.error("QA failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"QA error: {str(exc)}")


@router.post("/stream")
async def ask_question_stream(request: QARequest):
    """
    Streaming Q&A — returns Server-Sent Events (SSE).
    Each event is a JSON chunk: {"delta": "...", "done": false}
    Final event: {"answer": "...", "sources": [...], "done": true}
    """
    async def event_generator():
        try:
            # Run the full agent then stream the answer token-by-token
            response = await run_qa(
                question=request.question,
                doc_ids=request.doc_ids,
                top_k=request.top_k,
                session_id=request.session_id,
                user_id=request.user_id,
            )
            # Simulate streaming by yielding answer in chunks
            words = response.answer.split()
            for i in range(0, len(words), 5):
                chunk = " ".join(words[i:i+5])
                yield f"data: {json.dumps({'delta': chunk + ' ', 'done': False})}\n\n"
                await asyncio.sleep(0.02)

            # Final event with full structured response
            final = {
                "delta": "",
                "done": True,
                "answer": response.answer,
                "sources": [s.model_dump() for s in response.sources],
                "session_id": response.session_id,
            }
            yield f"data: {json.dumps(final)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
