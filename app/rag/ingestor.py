"""
app/rag/ingestor.py — PDF ingestion pipeline.

Pipeline:
  1. Extract text per page (pdfplumber — handles tables/columns better)
  2. Extract images per page (pymupdf) → encode to base64
  3. For image-heavy pages → describe with Groq vision model
  4. Chunk text with LlamaIndex SentenceSplitter
  5. Embed with FastEmbed (local)
  6. Upsert into Qdrant (per-doc collection + global collection)
  7. Persist metadata to JSON sidecar
"""
from __future__ import annotations
import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import AsyncGenerator

import pdfplumber
import fitz  # pymupdf
from PIL import Image
import io

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, Settings as LISettings
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.core.config import get_settings
from app.core.embeddings import get_embed_model
from app.core.vector_store import (
    get_qdrant_client,
    ensure_collection,
    collection_for_doc,
    global_collection,
)
from app.core.logging import get_logger
from app.models.schemas import DocumentMeta

log = get_logger(__name__)
_settings = get_settings()

META_DIR = _settings.processed_dir / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)


# ── Image description via Groq vision ─────────────────────────────────────────

async def _describe_image(image_bytes: bytes, context: str = "") -> str:
    """Send an image to Groq vision and return a text description."""
    try:
        import httpx, base64
        b64 = base64.b64encode(image_bytes).decode()
        headers = {
            "Authorization": f"Bearer {_settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": _settings.groq_vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"This image is from a textbook/study note. "
                                f"Context: {context or 'educational content'}. "
                                "Describe every detail: diagrams, labels, data in charts, "
                                "formulas, arrows, annotations. Be thorough so a student "
                                "who cannot see the image can understand it fully."
                            ),
                        },
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        log.warning("Image description failed", error=str(exc))
        return "[Image: description unavailable]"


def _extract_images_from_page(page: fitz.Page, min_size: int = 100) -> list[bytes]:
    """Return list of JPEG bytes for images on a pymupdf page."""
    images = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base_image = page.parent.extract_image(xref)
            img_bytes = base_image["image"]
            # Convert to JPEG, resize if needed
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil.size
            if w < min_size or h < min_size:
                continue
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            images.append(buf.getvalue())
        except Exception:
            continue
    return images


# ── Main ingestion function ────────────────────────────────────────────────────

async def ingest_pdf(
    file_path: Path,
    doc_id: str,
    filename: str,
    subject: str | None = None,
    chapter: str | None = None,
    tags: list[str] | None = None,
) -> DocumentMeta:
    """
    Full async ingestion pipeline. Returns updated DocumentMeta.
    Safe to call multiple times — re-ingestion replaces old vectors.
    """
    meta = DocumentMeta(
        doc_id=doc_id,
        filename=filename,
        subject=subject,
        chapter=chapter,
        tags=tags or [],
        status="processing",
    )
    _save_meta(meta)

    try:
        documents = await _pdf_to_documents(file_path, meta)
        meta.page_count = _count_pages(file_path)
        meta.has_images = any(
            d.metadata.get("has_images") for d in documents
        )

        # Chunk
        splitter = SentenceSplitter(
            chunk_size=_settings.chunk_size,
            chunk_overlap=_settings.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents(documents)
        meta.chunk_count = len(nodes)
        log.info("Chunked document", doc_id=doc_id, chunks=len(nodes))

        # Embed + upsert into doc-specific AND global collection
        embed_model = get_embed_model()
        qdrant = get_qdrant_client()

        for coll_name in [collection_for_doc(doc_id), global_collection()]:
            ensure_collection(coll_name)
            vector_store = QdrantVectorStore(
                client=qdrant,
                collection_name=coll_name,
            )
            storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
            LISettings.embed_model = embed_model
            VectorStoreIndex(
                nodes,
                storage_context=storage_ctx,
                show_progress=False,
            )
            log.info("Upserted nodes into Qdrant", collection=coll_name, count=len(nodes))

        meta.status = "ready"
    except Exception as exc:
        log.error("Ingestion failed", doc_id=doc_id, error=str(exc))
        meta.status = "failed"
        meta.error = str(exc)

    _save_meta(meta)
    return meta


async def _pdf_to_documents(file_path: Path, meta: DocumentMeta) -> list[Document]:
    """Extract text (and image descriptions) from every page."""
    documents: list[Document] = []
    fitz_doc = fitz.open(str(file_path))

    with pdfplumber.open(str(file_path)) as pdf:
        for page_num, plumber_page in enumerate(pdf.pages):
            page_text = plumber_page.extract_text() or ""

            # Extract images from the same page via pymupdf
            fitz_page = fitz_doc[page_num]
            raw_images = _extract_images_from_page(fitz_page)
            has_images = len(raw_images) > 0

            img_descriptions: list[str] = []
            if raw_images:
                context = f"{meta.subject or ''} {meta.chapter or ''} page {page_num + 1}"
                tasks = [_describe_image(img, context) for img in raw_images]
                img_descriptions = await asyncio.gather(*tasks)

            combined_text = page_text
            if img_descriptions:
                combined_text += "\n\n[Image Descriptions]\n" + "\n".join(
                    f"- {d}" for d in img_descriptions
                )

            if combined_text.strip():
                documents.append(
                    Document(
                        text=combined_text,
                        metadata={
                            "doc_id": meta.doc_id,
                            "filename": meta.filename,
                            "subject": meta.subject or "",
                            "chapter": meta.chapter or "",
                            "tags": ",".join(meta.tags),
                            "page_number": page_num + 1,
                            "has_images": has_images,
                        },
                        doc_id=f"{meta.doc_id}_p{page_num + 1}",
                    )
                )

    fitz_doc.close()
    return documents


def _count_pages(file_path: Path) -> int:
    try:
        doc = fitz.open(str(file_path))
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return 0


# ── Metadata persistence ───────────────────────────────────────────────────────

def _save_meta(meta: DocumentMeta):
    path = META_DIR / f"{meta.doc_id}.json"
    path.write_text(meta.model_dump_json(indent=2))


def load_meta(doc_id: str) -> DocumentMeta | None:
    path = META_DIR / f"{doc_id}.json"
    if not path.exists():
        return None
    return DocumentMeta.model_validate_json(path.read_text())


def list_all_meta() -> list[DocumentMeta]:
    metas = []
    for p in META_DIR.glob("*.json"):
        try:
            metas.append(DocumentMeta.model_validate_json(p.read_text()))
        except Exception:
            continue
    return sorted(metas, key=lambda m: m.filename)


def delete_doc(doc_id: str) -> bool:
    """Delete metadata and Qdrant vectors for a document."""
    meta = load_meta(doc_id)
    if not meta:
        return False
    qdrant = get_qdrant_client()
    coll = collection_for_doc(doc_id)
    try:
        qdrant.delete_collection(coll)
    except Exception:
        pass
    path = META_DIR / f"{doc_id}.json"
    path.unlink(missing_ok=True)
    return True
