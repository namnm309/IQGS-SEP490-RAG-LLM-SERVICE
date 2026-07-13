"""Tests cho parse-jd và validate-jd endpoints."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from api.app import app
from models.internal_schemas import ValidateJdResponse
from services.jd_parse_service import JdParseService
from services.document_parser import DocumentParser
from config.settings import Settings


VALID_JD_TEXT = """
Job Description - Senior .NET Developer

Vị trí: Nhân viên phát triển phần mềm backend.

Trách nhiệm:
- Thiết kế và phát triển Web API với ASP.NET Core.
- Tối ưu hiệu năng hệ thống và viết unit test.
- Phối hợp với team QA và DevOps trong quy trình CI/CD.

Yêu cầu:
- Tối thiểu 3 năm kinh nghiệm với C# và .NET.
- Kỹ năng Entity Framework Core, PostgreSQL, REST API.
- Kinh nghiệm Clean Architecture là lợi thế.

Level: Mid-Senior level, 3+ years experience required.
""" * 2


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
    from api.deps import get_jd_parse_service

    monkeypatch.setattr(deps, "startup", lambda: None)
    monkeypatch.setattr(deps, "shutdown", lambda: None)

    parser = DocumentParser()
    jd_service = JdParseService(parser=parser, settings=settings)
    app.dependency_overrides[get_jd_parse_service] = lambda: jd_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.pop(get_jd_parse_service, None)


def test_validate_jd_success(settings: Settings):
    from models.internal_schemas import ValidateJdRequest

    service = JdParseService(DocumentParser(), settings)
    result = service.validate_text(
        ValidateJdRequest(jobDescription=VALID_JD_TEXT, fileName="jd.txt")
    )
    assert result.success is True
    assert result.job_description


def test_validate_jd_too_short(settings: Settings):
    service = JdParseService(DocumentParser(), settings)
    from models.internal_schemas import ValidateJdRequest

    result = service.validate_text(
        ValidateJdRequest(jobDescription="too short", fileName="jd.txt")
    )
    assert result.success is False
    assert result.stage == "JD_VALIDATION"
    assert len(result.errors) >= 1


def test_parse_jd_txt_success(client: TestClient):
    files = {"file": ("jd.txt", VALID_JD_TEXT.encode("utf-8"), "text/plain")}
    response = client.post(
        "/internal/rag/parse-jd",
        files=files,
        headers={"X-Internal-Api-Key": "test-secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["jobDescription"]


def test_parse_jd_invalid_extension(client: TestClient):
    files = {"file": ("bad.exe", b"data", "application/octet-stream")}
    response = client.post(
        "/internal/rag/parse-jd",
        files=files,
        headers={"X-Internal-Api-Key": "test-secret"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["stage"] == "JD_PARSE"
    assert body["errors"]


def test_validate_jd_endpoint_fail(client: TestClient):
    response = client.post(
        "/internal/rag/validate-jd",
        json={"jobDescription": "short", "fileName": "jd.txt"},
        headers={"X-Internal-Api-Key": "test-secret"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["stage"] == "JD_VALIDATION"
    assert body["errors"]
