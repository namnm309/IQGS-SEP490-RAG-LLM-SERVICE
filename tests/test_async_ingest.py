"""Unit tests cho async ingest dispatch."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.app import app
from config.settings import Settings
from models.internal_schemas import IngestRequest


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

    monkeypatch.setattr(deps, "startup", lambda: None)
    monkeypatch.setattr(deps, "shutdown", lambda: None)

    mock_async_ingest = MagicMock()
    deps._async_ingest_service = mock_async_ingest
    deps._ingest_service = MagicMock()

    return TestClient(app)


def test_ingest_async_returns_202(client: TestClient, settings: Settings) -> None:
    document_id = str(uuid.uuid4())
    response = client.post(
        "/internal/rag/ingest/async",
        headers={"X-Internal-Api-Key": settings.internal_api_key},
        json={
            "documentId": document_id,
            "blobReadUrl": "https://example.com/file.pdf",
            "scope": "HR",
            "ownerId": str(uuid.uuid4()),
            "fileName": "test.pdf",
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] is True
    assert body["documentId"] == document_id
    assert body["phase"] == "INGEST"


def test_async_ingest_service_delegates_to_execute_ingest() -> None:
    from services.async_ingest_service import AsyncIngestService

    mock_ingest = MagicMock()
    service = AsyncIngestService(mock_ingest)
    request = IngestRequest(
        documentId="doc-1",
        blobReadUrl="https://example.com/a.pdf",
        scope="HR",
        ownerId=str(uuid.uuid4()),
        fileName="a.pdf",
    )

    service.run_ingest(request)

    mock_ingest.execute_ingest.assert_called_once_with(request)
