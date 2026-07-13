"""Smoke tests cho internal RAG API — không cần Ollama/PostgreSQL khi mock."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.app import app
from config.settings import Settings
from models.internal_schemas import IngestRequest, IngestResponse
from services.rag_ingest_service import RagIngestService
from vectorstores.base import ChunkRecord, RetrievedChunk


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

    mock_store = MagicMock()
    mock_store.ping.return_value = True
    mock_store.upsert_chunks.return_value = 2
    mock_store.delete_document_chunks.return_value = 3
    mock_store.similarity_search.side_effect = _mock_similarity_search

    mock_ingest = MagicMock()
    mock_ingest.ingest.return_value = IngestResponse(
        documentId="doc-1",
        status="COMPLETED",
        chunkCount=2,
    )
    mock_ingest.delete_document.return_value = 3

    mock_question = MagicMock()
    mock_question.generate.return_value = MagicMock(
        success=True,
        questions=[],
        raw_answer="{}",
        processing_time_ms=1.0,
        error=None,
    )
    mock_question.generate_from_plan.return_value = MagicMock(
        success=True,
        questions=[],
        processing_time_ms=1.0,
        error=None,
    )

    mock_plan = MagicMock()
    mock_plan.generate.return_value = MagicMock(
        success=True,
        plan=None,
        processing_time_ms=1.0,
        error=None,
    )

    import api.deps as deps

    monkeypatch.setattr(deps, "startup", lambda: None)
    monkeypatch.setattr(deps, "shutdown", lambda: None)

    deps._pool = MagicMock()
    deps._vector_store = mock_store
    deps._ingest_service = mock_ingest
    deps._question_service = mock_question
    deps._plan_service = mock_plan

    mock_async = MagicMock()
    deps._async_generation_service = mock_async

    return TestClient(app)


def _mock_similarity_search(query_embedding, scope, owner_id, top_k):
    return [
        RetrievedChunk(
            document_id=str(uuid.uuid4()),
            chunk_index=0,
            content="Sample chunk content for testing.",
            scope=scope,
            owner_id=owner_id,
            score=0.9,
            metadata={"fileName": "test.pdf"},
        )
    ]


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "database" in body
    assert "config" in body


def test_ingest_requires_api_key(client: TestClient) -> None:
    response = client.post(
        "/internal/rag/ingest",
        json={
            "documentId": str(uuid.uuid4()),
            "scope": "SYSTEM",
            "blobReadUrl": "http://example.com/file.pdf",
            "fileName": "file.pdf",
        },
    )
    assert response.status_code == 401


def test_ingest_with_api_key(client: TestClient, settings: Settings) -> None:
    response = client.post(
        "/internal/rag/ingest",
        headers={"X-Internal-Api-Key": settings.internal_api_key},
        json={
            "documentId": str(uuid.uuid4()),
            "scope": "SYSTEM",
            "blobReadUrl": "http://example.com/file.pdf",
            "fileName": "file.pdf",
            "sourceTitle": "Rubric",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"


def test_delete_chunks(client: TestClient, settings: Settings) -> None:
    doc_id = str(uuid.uuid4())
    response = client.delete(
        f"/internal/rag/documents/{doc_id}",
        headers={"X-Internal-Api-Key": settings.internal_api_key},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["deletedCount"] == 3
    assert "thành công" in body["message"]


def test_ingest_scope_validation() -> None:
    service = RagIngestService(
        vector_store=MagicMock(),
        downloader=MagicMock(),
        parser=MagicMock(),
        chunking=MagicMock(),
        embedding=MagicMock(),
        callback_client=MagicMock(),
    )
    request = IngestRequest(
        documentId=str(uuid.uuid4()),
        scope="HR",
        ownerId=None,
        blobReadUrl="http://x",
        fileName="a.pdf",
    )
    with pytest.raises(HTTPException) as exc:
        service.ingest(request)
    assert exc.value.status_code == 400


def test_ingest_error_response_has_stage(client: TestClient, settings: Settings, monkeypatch) -> None:
    import api.deps as deps

    failing = MagicMock()
    failing.ingest.side_effect = HTTPException(
        status_code=500,
        detail={
            "error": "[EMBEDDING] connection refused",
            "detail": "connection refused",
            "stage": "EMBEDDING",
            "exceptionType": "ConnectionError",
        },
    )
    deps._ingest_service = failing

    response = client.post(
        "/internal/rag/ingest",
        headers={"X-Internal-Api-Key": settings.internal_api_key},
        json={
            "documentId": str(uuid.uuid4()),
            "scope": "SYSTEM",
            "blobReadUrl": "http://example.com/file.pdf",
            "fileName": "file.pdf",
        },
    )
    assert response.status_code == 500
    body = response.json()
    assert body["stage"] == "EMBEDDING"
    assert "connection refused" in body["detail"]


def test_chunk_record_scope_rules(settings: Settings) -> None:
    from vectorstores.pgvector_store import PgVectorStore

    store = PgVectorStore(MagicMock(), settings)
    with pytest.raises(ValueError, match="owner_id"):
        store._validate_chunk_scope(
            ChunkRecord(
                chunk_index=0,
                content="x",
                embedding=[0.0] * settings.embedding_dimension,
                scope="SYSTEM",
                owner_id="should-be-null",
            )
        )


def test_generate_plan_requires_api_key(client: TestClient) -> None:
    response = client.post(
        "/internal/rag/generate-plan",
        json={
            "ownerId": "11111111-1111-1111-1111-111111111111",
            "jobDescription": "Backend role",
            "numberOfQuestions": 5,
        },
    )
    assert response.status_code == 401


def test_generate_plan_with_api_key(client: TestClient, settings: Settings) -> None:
    response = client.post(
        "/internal/rag/generate-plan",
        headers={"X-Internal-Api-Key": settings.internal_api_key},
        json={
            "ownerId": "11111111-1111-1111-1111-111111111111",
            "jobDescription": "Backend role",
            "numberOfQuestions": 5,
            "questionTypes": ["technical"],
        },
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_generate_questions_from_plan_requires_api_key(client: TestClient) -> None:
    response = client.post(
        "/internal/rag/generate-questions-from-plan",
        json={
            "ownerId": "11111111-1111-1111-1111-111111111111",
            "jobDescription": "Backend role",
            "approvedPlan": {
                "roleTitle": "Backend Developer",
                "experienceLevel": "senior",
                "totalQuestions": 1,
                "questionTypeDistribution": [
                    {"type": "technical", "count": 1, "reason": "JD"}
                ],
            },
        },
    )
    assert response.status_code == 401


    assert response.status_code == 200
    assert response.json()["success"] is True


def test_generate_plan_async_returns_202(client: TestClient, settings: Settings) -> None:
    response = client.post(
        "/internal/rag/generate-plan/async",
        headers={"X-Internal-Api-Key": settings.internal_api_key},
        json={
            "jobId": "d03fc5db-3eab-44dc-bba0-40b2192a7a10",
            "ownerId": "11111111-1111-1111-1111-111111111111",
            "jobDescription": "Backend role",
            "numberOfQuestions": 5,
            "questionTypes": ["technical"],
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] is True
    assert body["phase"] == "PLAN"
    assert body["jobId"] == "d03fc5db-3eab-44dc-bba0-40b2192a7a10"


def test_generate_questions_from_plan_async_returns_202(
    client: TestClient, settings: Settings
) -> None:
    response = client.post(
        "/internal/rag/generate-questions-from-plan/async",
        headers={"X-Internal-Api-Key": settings.internal_api_key},
        json={
            "jobId": "d03fc5db-3eab-44dc-bba0-40b2192a7a10",
            "ownerId": "11111111-1111-1111-1111-111111111111",
            "jobDescription": "Backend role",
            "approvedPlan": {
                "roleTitle": "Backend Developer",
                "experienceLevel": "senior",
                "totalQuestions": 1,
                "questionTypeDistribution": [
                    {"type": "technical", "count": 1, "reason": "JD"}
                ],
            },
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] is True
    assert body["phase"] == "QUESTIONS"
