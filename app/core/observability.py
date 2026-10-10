"""
app/core/observability.py — Langfuse tracing helpers.

All agent calls are wrapped with traces so you can monitor latency,
token usage, and quality in the Langfuse dashboard for free.
"""
from __future__ import annotations
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
_settings = get_settings()

# Lazy-initialise so the app starts even if Langfuse keys are absent
_langfuse = None


def _get_langfuse():
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    if not (_settings.langfuse_public_key and _settings.langfuse_secret_key):
        log.warning("Langfuse keys not configured — tracing disabled")
        return None
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=_settings.langfuse_public_key,
            secret_key=_settings.langfuse_secret_key,
            host=_settings.langfuse_host,
        )
        log.info("Langfuse tracing enabled", host=_settings.langfuse_host)
    except Exception as exc:
        log.warning("Langfuse init failed", error=str(exc))
        _langfuse = None
    return _langfuse


@contextmanager
def trace_span(
    name: str,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> Generator[Any, None, None]:
    """Context manager that wraps a block in a Langfuse trace span."""
    lf = _get_langfuse()
    if lf is None:
        yield None
        return

    trace = lf.trace(
        name=name,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata or {},
    )
    span = trace.span(name=name)
    try:
        yield span
    except Exception as exc:
        span.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        span.end()
        lf.flush()


def observe(name: str | None = None):
    """Decorator for async functions — auto-traces with Langfuse."""
    def decorator(fn: Callable):
        span_name = name or fn.__name__

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            lf = _get_langfuse()
            if lf is None:
                return await fn(*args, **kwargs)

            # Extract user_id / session_id from kwargs if present
            user_id = kwargs.get("user_id")
            session_id = kwargs.get("session_id")
            trace = lf.trace(name=span_name, user_id=user_id, session_id=session_id)
            span = trace.span(name=span_name)
            try:
                result = await fn(*args, **kwargs)
                span.update(level="DEFAULT")
                return result
            except Exception as exc:
                span.update(level="ERROR", status_message=str(exc))
                raise
            finally:
                span.end()
                lf.flush()

        return wrapper
    return decorator


def log_generation(
    trace_id: str | None,
    name: str,
    model: str,
    prompt: str,
    completion: str,
    usage: dict | None = None,
):
    """Log an individual LLM generation event to Langfuse."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        gen = lf.generation(
            trace_id=trace_id,
            name=name,
            model=model,
            prompt=prompt,
            completion=completion,
            usage=usage or {},
        )
        gen.end()
        lf.flush()
    except Exception as exc:
        log.warning("Langfuse log_generation failed", error=str(exc))
