"""
app/services/document_service.py — Document upload / management service.
"""
from __future__ import annotations
import uuid
from pathlib import Path

import aiofiles

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import DocumentMeta, IngestResponse
from app.rag.ingestor import ingest_pdf, load_meta, list_all_meta, delete_doc, _save_meta

log = get_logger(__name__)
_settings = get_settings()


async def handle_upload(
    file_bytes: bytes,
    filename: str,
    subject: str | None = None,
    chapter: str | None = None,
    tags: list[str] | None = None,
    doc_id: str | None = None,
) -> IngestResponse:
    """Save uploaded file and run ingestion pipeline."""
    doc_id = doc_id or str(uuid.uuid4())
    dest = _settings.upload_dir / f"{doc_id}_{filename}"

    async with aiofiles.open(dest, "wb") as f:
        await f.write(file_bytes)
    log.info("File saved", doc_id=doc_id, filename=filename, size=len(file_bytes))

    # Persist a 'processing' stub so clients can poll immediately
    pending_meta = DocumentMeta(
        doc_id=doc_id,
        filename=filename,
        subject=subject,
        chapter=chapter,
        tags=tags or [],
        status="processing",
    )
    _save_meta(pending_meta)

    meta = await ingest_pdf(
        file_path=dest,
        doc_id=doc_id,
        filename=filename,
        subject=subject,
        chapter=chapter,
        tags=tags,
    )

    return IngestResponse(
        doc_id=doc_id,
        message=f"Ingestion {'complete' if meta.status == 'ready' else 'failed'}",
        status=meta.status,
    )


def get_document(doc_id: str) -> DocumentMeta | None:
    return load_meta(doc_id)


def get_all_documents() -> list[DocumentMeta]:
    return list_all_meta()


def remove_document(doc_id: str) -> bool:
    return delete_doc(doc_id)
