"""
app/main.py — FastAPI application factory.
"""
from __future__ import annotations
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.api import api_router

setup_logging()
log = get_logger(__name__)
_settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown lifecycle hooks."""
    log.info(
        "Starting Tutor Agent API",
        env=_settings.app_env,
        llm=_settings.primary_llm_provider,
        embed=_settings.embed_model,
    )
    # Validate Gemini embedding model is reachable (no heavy download needed)
    try:
        from app.core.embeddings import get_embed_model
        get_embed_model()
        log.info("Gemini embedding model initialised", model=_settings.embed_model)
    except Exception as exc:
        log.warning(
            "Gemini embedding init failed — check GEMINI_API_KEY",
            error=str(exc),
        )

    # Ensure global Qdrant collection exists
    try:
        from app.core.vector_store import ensure_collection, global_collection
        ensure_collection(global_collection())
        log.info("Qdrant collection ready", name=global_collection())
    except Exception as exc:
        log.warning("Qdrant setup failed", error=str(exc))

    yield

    log.info("Shutting down Tutor Agent API")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tutor Agent API",
        description=(
            "RAG-based AI tutor for study notes — Q&A, quiz generation, "
            "test papers, and concept teaching. Powered by LlamaIndex + LangGraph."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS (wide-open for dev; tighten for prod) ─────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ──────────────────────────────────────────────
    @app.middleware("http")
    async def add_timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        response.headers["X-Response-Time-Ms"] = str(ms)
        log.debug("Request completed", path=request.url.path, ms=ms, status=response.status_code)
        return response

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error("Unhandled exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    # ── Routes ─────────────────────────────────────────────────────────────────
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "service": "Tutor Agent API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()
