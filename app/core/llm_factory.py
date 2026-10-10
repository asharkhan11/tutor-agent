"""
app/core/llm_factory.py — LLM provider with automatic Groq → Cerebras fallback.

Strategy:
- Try Groq first (fastest free tier, ~500 RPM limit).
- On RateLimitError / 429 → transparently retry with Cerebras.
- Both use llama-3.1-8b — outputs are equivalent.
"""
from __future__ import annotations
import asyncio
from functools import lru_cache
from typing import Literal

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
_settings = get_settings()

Provider = Literal["groq", "cerebras"]


def _make_groq_llm(model: str | None = None):
    from llama_index.llms.groq import Groq
    return Groq(
        model=model or _settings.groq_model,
        api_key=_settings.groq_api_key,
        temperature=0.1,
        max_tokens=2048,
    )


def _make_cerebras_llm(model: str | None = None):
    from llama_index.llms.cerebras import Cerebras
    return Cerebras(
        model=model or _settings.cerebras_model,
        api_key=_settings.cerebras_api_key,
        temperature=0.1,
        max_tokens=2048,
    )


@lru_cache(maxsize=4)
def get_llm(provider: Provider | None = None, model: str | None = None):
    """Return a cached LlamaIndex LLM instance."""
    p = provider or _settings.primary_llm_provider
    if p == "groq":
        return _make_groq_llm(model)
    return _make_cerebras_llm(model)


def get_fallback_llm():
    """Return the secondary provider LLM."""
    if _settings.primary_llm_provider == "groq":
        return _make_cerebras_llm()
    return _make_groq_llm()


class ResilientLLM:
    """
    Thin wrapper that tries the primary provider then falls back gracefully.
    Exposes .acomplete() and .achat() matching LlamaIndex interface.
    """

    def __init__(self):
        self._primary = get_llm()
        self._fallback = get_fallback_llm()

    async def acomplete(self, prompt: str, **kwargs):
        try:
            return await self._primary.acomplete(prompt, **kwargs)
        except Exception as exc:
            if _is_rate_limit(exc):
                log.warning("Primary LLM rate-limited, switching to fallback", error=str(exc))
                return await self._fallback.acomplete(prompt, **kwargs)
            raise

    async def achat(self, messages, **kwargs):
        try:
            return await self._primary.achat(messages, **kwargs)
        except Exception as exc:
            if _is_rate_limit(exc):
                log.warning("Primary LLM rate-limited, switching to fallback", error=str(exc))
                return await self._fallback.achat(messages, **kwargs)
            raise

    def complete(self, prompt: str, **kwargs):
        try:
            return self._primary.complete(prompt, **kwargs)
        except Exception as exc:
            if _is_rate_limit(exc):
                log.warning("Primary LLM rate-limited, switching to fallback", error=str(exc))
                return self._fallback.complete(prompt, **kwargs)
            raise

    def chat(self, messages, **kwargs):
        try:
            return self._primary.chat(messages, **kwargs)
        except Exception as exc:
            if _is_rate_limit(exc):
                log.warning("Primary LLM rate-limited, switching to fallback", error=str(exc))
                return self._fallback.chat(messages, **kwargs)
            raise


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "quota" in msg


_resilient_llm: ResilientLLM | None = None


def get_resilient_llm() -> ResilientLLM:
    global _resilient_llm
    if _resilient_llm is None:
        _resilient_llm = ResilientLLM()
    return _resilient_llm
