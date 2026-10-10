"""
app/api/routes/health.py — Liveness and readiness probes.
"""
from __future__ import annotations

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])
_settings = get_settings()


@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Returns the operational status of all subsystems.
    Safe to call from Docker/K8s health probes.
    """
    # Check Qdrant
    qdrant_status = "ok"
    try:
        from app.core.vector_store import get_qdrant_client
        client = get_qdrant_client()
        client.get_collections()
    except Exception as exc:
        qdrant_status = f"error: {str(exc)[:60]}"

    # Check LLM provider
    llm_status = _settings.primary_llm_provider
    if _settings.primary_llm_provider == "groq" and not _settings.groq_api_key:
        llm_status = "groq: missing API key"
    elif _settings.primary_llm_provider == "cerebras" and not _settings.cerebras_api_key:
        llm_status = "cerebras: missing API key"

    # Check Langfuse
    lf_status = "disabled"
    if _settings.langfuse_public_key and _settings.langfuse_secret_key:
        lf_status = "enabled"

    return HealthResponse(
        status="ok" if qdrant_status == "ok" else "degraded",
        qdrant=qdrant_status,
        llm_provider=llm_status,
        embed_model=_settings.embed_model,
        langfuse=lf_status,
    )
