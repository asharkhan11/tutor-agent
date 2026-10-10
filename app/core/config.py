"""
app/core/config.py — Centralised settings via pydantic-settings.
All values are read from environment variables / .env file.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    primary_llm_provider: str = "groq"
    groq_model: str = "llama-3.1-8b-instant"
    cerebras_model: str = "llama3.1-8b"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "tutor_notes"

    # ── Langfuse ───────────────────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Embedding — Google Gemini (free tier, online, no local model) ──────────
    gemini_api_key: str = ""                          # from https://aistudio.google.com
    embed_model: str = "gemini-embedding-001"   # text-embedding-004 deprecated Nov 2025; gemini-embedding-001 is the replacement
    embed_dimension: int = 3072                    # gemini-embedding-001 outputs 3072-dim (was 768 for text-embedding-004); must match model

    # ── Storage ────────────────────────────────────────────────────────────────
    upload_dir: Path = Path("./data/uploads")
    processed_dir: Path = Path("./data/processed")
    max_upload_size_mb: int = 50

    # ── RAG tuning ─────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── App ────────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    allowed_origins: list[str] = ["*"]

    def ensure_dirs(self):
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
