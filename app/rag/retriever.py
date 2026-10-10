"""
app/rag/retriever.py — Flexible RAG retriever.

Supports:
  - Single-doc retrieval (doc-specific Qdrant collection)
  - Multi-doc retrieval (global collection, filtered by doc_ids metadata)
  - Contextual compression (re-ranking via cross-encoder if available)
"""
from __future__ import annotations
from typing import Any

from llama_index.core import VectorStoreIndex, QueryBundle, Settings as LISettings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchAny

from app.core.config import get_settings
from app.core.embeddings import get_embed_model
from app.core.vector_store import (
    get_qdrant_client,
    collection_for_doc,
    global_collection,
)
from app.core.logging import get_logger
from app.models.schemas import SourceNode

log = get_logger(__name__)
_settings = get_settings()


def _build_retriever(collection_name: str, top_k: int, doc_id_filter: list[str] | None = None):
    """Build a LlamaIndex retriever over a Qdrant collection."""
    qdrant = get_qdrant_client()
    embed = get_embed_model()

    # Inject embed model into LlamaIndex global Settings so VectorStoreIndex
    # never silently falls back to OpenAI embeddings.
    LISettings.embed_model = embed

    qdrant_filter = None
    if doc_id_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=doc_id_filter),
                )
            ]
        )

    vector_store = QdrantVectorStore(
        client=qdrant,
        collection_name=collection_name,
        enable_hybrid=False,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        vector_store_kwargs={"qdrant_filters": qdrant_filter} if qdrant_filter else {},
    )
    return retriever


async def retrieve_nodes(
    query: str,
    doc_ids: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve relevant nodes for a query.

    - If doc_ids is a single item → query its dedicated collection.
    - If doc_ids has multiple items or is None → query global collection.
    Returns list of dicts with keys: text, score, metadata.
    """
    if doc_ids and len(doc_ids) == 1:
        coll = collection_for_doc(doc_ids[0])
        retriever = _build_retriever(coll, top_k)
    else:
        coll = global_collection()
        retriever = _build_retriever(coll, top_k, doc_id_filter=doc_ids)

    bundle = QueryBundle(query_str=query)
    nodes = await retriever.aretrieve(bundle)

    results = []
    for n in nodes:
        results.append({
            "text": n.node.text,
            "score": float(n.score or 0.0),
            "metadata": n.node.metadata or {},
        })
    return results


def nodes_to_source_refs(nodes: list[dict]) -> list[SourceNode]:
    out = []
    for n in nodes:
        meta = n["metadata"]
        out.append(
            SourceNode(
                doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", ""),
                page=meta.get("page_number"),
                score=round(n["score"], 4),
                text=n["text"][:300] + ("…" if len(n["text"]) > 300 else ""),
            )
        )
    return out


def build_context_string(nodes: list[dict]) -> str:
    """Concatenate retrieved chunks into a compact context block."""
    parts = []
    for i, n in enumerate(nodes):
        meta = n["metadata"]
        header = f"[Source {i+1}: {meta.get('filename', 'unknown')}, page {meta.get('page_number', '?')}]"
        parts.append(f"{header}\n{n['text']}")
    return "\n\n---\n\n".join(parts)
