"""
app/core/vector_store.py — Qdrant cloud vector store (singleton).

Uses named collections per document_id so each PDF/book is isolated
and can be queried independently or jointly.
"""
from __future__ import annotations
from functools import lru_cache

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
_settings = get_settings()


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    log.info("Connecting to Qdrant", url=_settings.qdrant_url)
    return QdrantClient(
        url=_settings.qdrant_url,
        api_key=_settings.qdrant_api_key,
        timeout=30,
    )


@lru_cache(maxsize=1)
def get_async_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=_settings.qdrant_url,
        api_key=_settings.qdrant_api_key,
        timeout=30,
    )


def ensure_collection(collection_name: str, dim: int | None = None) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    dim = dim or _settings.embed_dimension
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        log.info("Created Qdrant collection", name=collection_name, dim=dim)
    else:
        log.debug("Collection already exists", name=collection_name)


def collection_for_doc(doc_id: str) -> str:
    """Derive a Qdrant collection name from a document id."""
    # Qdrant collection names: alphanumeric + underscores
    safe = doc_id.replace("-", "_").replace(" ", "_").lower()
    return f"doc_{safe}"


def global_collection() -> str:
    return _settings.qdrant_collection_name
