"""
app/api/__init__.py — Aggregate all route modules.
"""
from fastapi import APIRouter
from app.api.routes import documents, qa, quiz, teach, health

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(qa.router)
api_router.include_router(quiz.router)
api_router.include_router(teach.router)
