"""
tests/test_api.py — Integration tests for all major API endpoints.
Run with: pytest tests/ -v --asyncio-mode=auto

Requires a .env file with real credentials (or set env vars directly).
Tests use a fixture PDF so you don't need to upload manually.
"""
from __future__ import annotations
import io
import os
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# ── Minimal PDF bytes (valid 1-page PDF) ──────────────────────────────────────
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
    b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td "
    b"(Photosynthesis converts light to energy.) Tj ET\nendstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000062 00000 n \n0000000119 00000 n \n0000000274 00000 n \n"
    b"0000000374 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\n"
    b"startxref\n455\n%%EOF\n"
)

DOC_ID: str = ""  # filled by upload fixture


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="module")
async def client():
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    r = await client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "qdrant" in data


@pytest.mark.asyncio
async def test_upload_pdf(client: AsyncClient):
    global DOC_ID
    r = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("test_notes.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")},
        data={"subject": "Biology", "chapter": "Chapter 1", "tags": "test,photosynthesis"},
    )
    assert r.status_code in (200, 202), r.text
    data = r.json()
    assert "doc_id" in data
    DOC_ID = data["doc_id"]


@pytest.mark.asyncio
async def test_list_documents(client: AsyncClient):
    r = await client.get("/api/v1/documents")
    assert r.status_code == 200
    data = r.json()
    assert "documents" in data
    assert isinstance(data["total"], int)


@pytest.mark.asyncio
async def test_get_document(client: AsyncClient):
    if not DOC_ID:
        pytest.skip("Upload test must run first")
    r = await client.get(f"/api/v1/documents/{DOC_ID}")
    assert r.status_code == 200
    data = r.json()
    assert data["doc_id"] == DOC_ID
    assert data["subject"] == "Biology"


@pytest.mark.asyncio
async def test_qa(client: AsyncClient):
    if not DOC_ID:
        pytest.skip("Upload test must run first")
    r = await client.post(
        "/api/v1/qa",
        json={
            "question": "What is photosynthesis?",
            "doc_ids": [DOC_ID],
            "top_k": 3,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


@pytest.mark.asyncio
async def test_quiz_generation(client: AsyncClient):
    if not DOC_ID:
        pytest.skip("Upload test must run first")
    r = await client.post(
        "/api/v1/quiz/generate",
        json={
            "doc_ids": [DOC_ID],
            "topic": "photosynthesis",
            "num_questions": 3,
            "question_types": ["mcq"],
            "difficulty": "easy",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "questions" in data
    assert isinstance(data["questions"], list)


@pytest.mark.asyncio
async def test_teach(client: AsyncClient):
    if not DOC_ID:
        pytest.skip("Upload test must run first")
    r = await client.post(
        "/api/v1/teach",
        json={
            "concept": "photosynthesis",
            "doc_ids": [DOC_ID],
            "detail_level": "brief",
            "include_examples": True,
            "include_analogies": True,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "explanation" in data
    assert "key_points" in data


@pytest.mark.asyncio
async def test_delete_document(client: AsyncClient):
    if not DOC_ID:
        pytest.skip("Upload test must run first")
    r = await client.delete(f"/api/v1/documents/{DOC_ID}")
    assert r.status_code == 204
