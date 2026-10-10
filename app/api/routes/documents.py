"""
app/api/routes/documents.py — PDF upload & document management endpoints.
"""
from __future__ import annotations
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.models.schemas import DocumentListResponse, DocumentMeta, IngestResponse
from app.services.document_service import (
    handle_upload,
    get_document,
    get_all_documents,
    remove_document,
)

router = APIRouter(prefix="/documents", tags=["Documents"])
_settings = get_settings()

MAX_BYTES = _settings.max_upload_size_mb * 1024 * 1024


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF file to ingest")],
    subject: str | None = Form(default=None, description="Subject / course name"),
    chapter: str | None = Form(default=None, description="Chapter name or number"),
    tags: str | None = Form(default=None, description="Comma-separated tags"),
):
    """
    Upload a PDF to be processed and indexed.

    - Accepts multipart/form-data
    - Returns 202 immediately with a `doc_id`; poll `/documents/{doc_id}` for status
    - Max file size: configurable via MAX_UPLOAD_SIZE_MB env var
    """
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {_settings.max_upload_size_mb} MB.",
        )

    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]

    # Generate doc_id here so we can return it immediately while ingestion runs in background
    doc_id = str(uuid.uuid4())
    background_tasks.add_task(
        handle_upload,
        file_bytes=raw,
        filename=file.filename or "upload.pdf",
        subject=subject,
        chapter=chapter,
        tags=tag_list,
        doc_id=doc_id,
    )

    return IngestResponse(
        doc_id=doc_id,
        message="Upload received. Ingestion running — poll /documents/{doc_id} for status.",
        status="processing",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents():
    """List all ingested documents with their metadata."""
    docs = get_all_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/{doc_id}", response_model=DocumentMeta)
def get_doc(doc_id: str):
    """Get metadata and ingestion status for a specific document."""
    meta = get_document(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found.")
    return meta


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_doc_route(doc_id: str):
    """Delete a document and all its vectors from Qdrant."""
    ok = remove_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found.")
