"""Integration tests cho parse-cv endpoint (SCRUM-300)."""
from __future__ import annotations

import io
import json

import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from api.app import app
from config.settings import Settings
from services.cv_parse_service import CvParseService
from services.document_parser import DocumentParser


def _docx_bytes(lines: list[str]) -> bytes:
    doc = DocxDocument()
    for line in lines:
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        DATABASE_URL="postgresql://postgres:postgres@localhost:5432/iqgs",
        BACKEND_INTERNAL_BASE_URL="http://localhost:5000",
        INTERNAL_API_KEY="test-secret",
        EMBEDDING_DIMENSION=768,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", settings.database_url)
    monkeypatch.setenv("BACKEND_INTERNAL_BASE_URL", settings.backend_internal_base_url)
    monkeypatch.setenv("INTERNAL_API_KEY", settings.internal_api_key)
    monkeypatch.setenv("EMBEDDING_DIMENSION", str(settings.embedding_dimension))

    from config.settings import get_settings

    get_settings.cache_clear()

    import api.deps as deps
    from api.deps import get_cv_parse_service

    monkeypatch.setattr(deps, "startup", lambda: None)
    monkeypatch.setattr(deps, "shutdown", lambda: None)

    # Mock LLM: trả JSON cố định khi parse DOCX
    class _FakeCompletions:
        def create(self, **kwargs):  # noqa: ANN003
            content = json.dumps(
                {"skills": ["React", "TypeScript"], "summary": "Frontend dev."},
                ensure_ascii=False,
            )
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    fake_client = type("Client", (), {"chat": type("Chat", (), {"completions": _FakeCompletions()})()})()
    cv_service = CvParseService(parser=DocumentParser(), client=fake_client, settings=settings)
    app.dependency_overrides[get_cv_parse_service] = lambda: cv_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.pop(get_cv_parse_service, None)


def test_parse_cv_docx_endpoint_success(client: TestClient) -> None:
    docx = _docx_bytes(["Skills: React, TypeScript"])
    files = {"file": ("cv.docx", docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    response = client.post(
        "/internal/rag/parse-cv",
        files=files,
        headers={"X-Internal-Api-Key": "test-secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["skills"] == ["React", "TypeScript"]
    assert body["summary"] == "Frontend dev."
    assert body["fileName"] == "cv.docx"


def test_parse_cv_invalid_extension(client: TestClient) -> None:
    files = {"file": ("bad.gif", b"data", "image/gif")}
    response = client.post(
        "/internal/rag/parse-cv",
        files=files,
        headers={"X-Internal-Api-Key": "test-secret"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["stage"] == "CV_PARSE"
    assert body["errors"]
