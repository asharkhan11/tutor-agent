"""
app/core/embeddings.py — Google GenAI embeddings via the new google-genai SDK.

Uses gemini-embedding-001:
  - 3072 dims (default output_dimensionality)
  - Free tier: 1,500 requests/min
  - Stable v1 API — no v1beta issues
  - https://ai.google.dev/gemini-api/docs/models#text-embedding

MIGRATION NOTE:
  llama-index-embeddings-gemini 0.4.2 is the last release of that package.
  It used the old google-generativeai SDK which routes through v1beta — that
  endpoint no longer serves embedding models (HTTP 404). The replacement is
  llama-index-embeddings-google-genai + google-genai SDK (v1 API).
"""
from functools import lru_cache
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embed_model():
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    log.info("Initialising Google GenAI embedding model", model=settings.embed_model)
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    return GoogleGenAIEmbedding(
        model_name=settings.embed_model,   # "gemini-embedding-001"
        api_key=settings.gemini_api_key,
    )
