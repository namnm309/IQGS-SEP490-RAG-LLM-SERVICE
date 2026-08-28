"""Mocked tests for analyze-jd responsibilities + SCRUM-432 classify gate."""
from __future__ import annotations

from unittest.mock import MagicMock

from models.internal_schemas import AnalyzeJdRequest
from services.jd_analyze_service import JdAnalyzeService


def _service_with_llm(payload: str) -> JdAnalyzeService:
    client = MagicMock()
    settings = MagicMock()
    settings.chat_model = "test-model"
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=payload))]
    client.chat.completions.create.return_value = response
    return JdAnalyzeService(client=client, settings=settings)


def test_analyze_jd_maps_responsibilities_and_summary():
    llm_payload = """{
      "documentType": "job_description",
      "isItRole": true,
      "rejectReason": null,
      "jobTitle": "Senior .NET Backend Developer",
      "detectedRole": "backend engineer",
      "experienceLevel": "senior",
      "detectedLanguage": "english",
      "skills": ["ASP.NET Core", "PostgreSQL"],
      "responsibilities": ["Design REST APIs", "Optimize database queries"],
      "summary": "Hiring a senior backend engineer for API and database work."
    }"""
    service = _service_with_llm(llm_payload)
    result = service.analyze(AnalyzeJdRequest(job_description="Senior .NET role with PostgreSQL."))

    assert result.success is True
    assert result.document_type == "job_description"
    assert result.is_it_role is True
    assert result.job_title == "Senior .NET Backend Developer"
    assert result.experience_level == "senior"
    assert len(result.responsibilities) == 2
    assert "REST APIs" in result.responsibilities[0]
    assert result.summary is not None
    assert "backend" in result.summary.lower()


def test_analyze_jd_rejects_react_tutorial():
    llm_payload = """{
      "documentType": "article",
      "isItRole": true,
      "rejectReason": "Đây là bài tutorial React, không phải tin tuyển dụng.",
      "jobTitle": null,
      "skills": ["React"],
      "responsibilities": [],
      "summary": null
    }"""
    service = _service_with_llm(llm_payload)
    result = service.analyze(
        AnalyzeJdRequest(job_description="Learn React hooks: useState and useEffect explained...")
    )

    assert result.success is False
    assert result.stage == "JD_CLASSIFY"
    assert result.exception_type == "NotItJobPosting"
    assert result.document_type == "article"
    assert "tutorial" in (result.reject_reason or "").lower() or "tuyển" in (result.detail or "").lower()


def test_analyze_jd_rejects_csharp_resume():
    llm_payload = """{
      "documentType": "resume",
      "isItRole": true,
      "rejectReason": "Đây là CV ứng viên C#, không phải Job Description.",
      "jobTitle": null,
      "skills": ["C#"],
      "responsibilities": [],
      "summary": null
    }"""
    service = _service_with_llm(llm_payload)
    result = service.analyze(
        AnalyzeJdRequest(job_description="I am a Senior C# Developer with 5 years experience...")
    )

    assert result.success is False
    assert result.stage == "JD_CLASSIFY"
    assert result.document_type == "resume"


def test_analyze_jd_rejects_marketing_not_it_role():
    llm_payload = """{
      "documentType": "job_description",
      "isItRole": false,
      "rejectReason": "Vị trí Digital Marketing không thuộc IT/phần mềm.",
      "jobTitle": "Digital Marketing Executive",
      "skills": ["SEO"],
      "responsibilities": ["Chạy campaign"],
      "summary": null
    }"""
    service = _service_with_llm(llm_payload)
    result = service.analyze(
        AnalyzeJdRequest(job_description="Hiring Digital Marketing Executive for SEO campaigns...")
    )

    assert result.success is False
    assert result.stage == "JD_CLASSIFY"
    assert result.document_type == "job_description"
    assert result.is_it_role is False
    assert "IT" in (result.detail or "") or "phần mềm" in (result.detail or "")
