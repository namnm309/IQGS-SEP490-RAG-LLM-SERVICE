"""Unit tests cho CV parse service (SCRUM-300 / SCRUM-466) — nhánh text, nhánh ảnh, classify IT."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from docx import Document as DocxDocument

from services.cv_parse_service import CvParseService
from services.document_parser import DocumentParser


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        chat_model="test-model",
        ollama_base_url="http://localhost:11434/v1",
        request_timeout_seconds=30,
        temperature=0.1,
    )


def _cv_json(**overrides) -> str:
    payload = {
        "documentType": "resume",
        "isItRole": True,
        "rejectReason": None,
        "skills": ["C#", "ASP.NET Core", "PostgreSQL"],
        "summary": "Backend dev 3 năm.",
        "suggestedRole": ".NET Backend Developer",
        "yearsOfExperienceHint": 3,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class _FakeCompletions:
    def __init__(self, content: str, calls: list) -> None:
        self._content = content
        self._calls = calls

    def create(self, **kwargs):  # noqa: ANN003
        self._calls.append(kwargs)
        message = type("Message", (), {"content": self._content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.calls: list = []
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content, self.calls)})()


def _docx_bytes(lines: list[str]) -> bytes:
    doc = DocxDocument()
    for line in lines:
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _it_cv_lines(*extra: str) -> list[str]:
    base = [
        "Nguyen Van A - Software Engineer / Backend Developer",
        "Skills: C#, ASP.NET Core, PostgreSQL, Docker, React, TypeScript",
        "Experience: built REST API microservices with Kubernetes and CI/CD.",
    ]
    return [*base, *extra]


def _service(content: str) -> CvParseService:
    return CvParseService(
        parser=DocumentParser(),
        client=_FakeClient(content),
        settings=_settings(),
    )


def test_parse_cv_docx_text_success() -> None:
    service = _service(_cv_json())
    docx = _docx_bytes(_it_cv_lines())

    result, error, status = service.parse_upload(docx, "cv.docx")

    assert error is None
    assert status == 200
    assert result is not None
    assert result.success is True
    assert result.skills == ["C#", "ASP.NET Core", "PostgreSQL"]
    assert result.summary == "Backend dev 3 năm."
    assert result.file_name == "cv.docx"
    assert result.document_type == "resume"
    assert result.is_it_role is True


def test_parse_cv_dedup_and_strip_skills() -> None:
    service = _service(_cv_json(skills=["C#", " c# ", "Docker", ""], summary=""))
    docx = _docx_bytes(_it_cv_lines())

    result, _, _ = service.parse_upload(docx, "cv.docx")

    assert result is not None
    assert result.skills == ["C#", "Docker"]
    assert result.summary is None


def test_parse_cv_markdown_fenced_json() -> None:
    content = "```json\n" + _cv_json(skills=["Python"], summary="AI eng") + "\n```"
    service = _service(content)
    docx = _docx_bytes(_it_cv_lines("Skills: Python, FastAPI, Docker"))

    result, error, status = service.parse_upload(docx, "cv.docx")

    assert error is None and status == 200
    assert result is not None and result.skills == ["Python"]


def test_parse_cv_json_retry_then_success() -> None:
    bad = "không phải json"
    good = _cv_json(skills=["Go"], summary="SRE")

    calls: list = []

    class _RetryCompletions:
        def __init__(self) -> None:
            self._seq = [bad, good]

        def create(self, **kwargs):  # noqa: ANN003
            calls.append(kwargs)
            content = self._seq.pop(0)
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    client = SimpleNamespace(chat=SimpleNamespace(completions=_RetryCompletions()))
    service = CvParseService(parser=DocumentParser(), client=client, settings=_settings())

    result, error, status = service.parse_upload(
        _docx_bytes(_it_cv_lines("Skills: Golang, Docker, Kubernetes")), "cv.docx"
    )

    assert error is None and status == 200
    assert result is not None and result.skills == ["Go"]
    assert len(calls) == 2


def test_parse_cv_rejects_non_it_classify() -> None:
    service = _service(
        _cv_json(
            documentType="resume",
            isItRole=False,
            rejectReason="CV Marketing không thuộc IT.",
            skills=["SEO", "Content Marketing"],
        )
    )
    # L1 có thể pass nếu text đủ tech; L2 phải reject isItRole=false
    docx = _docx_bytes(_it_cv_lines())
    result, error, status = service.parse_upload(docx, "cv.docx")
    assert result is None
    assert status == 422
    assert error is not None
    assert error["stage"] == "CV_CLASSIFY"


def test_parse_cv_rejects_empty_skills() -> None:
    service = _service(_cv_json(skills=[]))
    result, error, status = service.parse_upload(_docx_bytes(_it_cv_lines()), "cv.docx")
    assert result is None
    assert status == 422
    assert error is not None
    assert error["exceptionType"] == "CvNoItSkills"


def test_parse_cv_image_branch_uses_native_api(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        captured["url"] = url
        captured["payload"] = json
        payload_content = _cv_json(skills=["Java", "Spring"], summary="Java dev")
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": payload_content}},
        )

    import services.cv_parse_service as mod

    monkeypatch.setattr(mod.httpx, "post", fake_post)

    service = CvParseService(
        parser=DocumentParser(),
        client=_FakeClient("unused"),
        settings=_settings(),
    )

    result, error, status = service.parse_upload(b"\x89PNG fake bytes", "cv.png")

    assert error is None and status == 200
    assert result is not None and result.skills == ["Java", "Spring"]
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "test-model"
    assert "images" in captured["payload"]["messages"][1]
    assert len(captured["payload"]["messages"][1]["images"]) == 1


def test_parse_cv_invalid_extension() -> None:
    service = _service("{}")
    result, error, status = service.parse_upload(b"data", "cv.exe")

    assert result is None
    assert status == 422
    assert error is not None
    assert error["stage"] == "CV_PARSE"
    assert error["errors"]


def test_parse_cv_empty_file() -> None:
    service = _service("{}")
    result, error, status = service.parse_upload(b"", "cv.pdf")

    assert result is None
    assert status == 422
    assert error is not None
    assert error["exceptionType"] == "EmptyDocument"


def test_normalize_skills_sanitizes_en_dash_and_parens() -> None:
    """SCRUM-493: en-dash → hyphen; giữ (SDUI); drop marketing."""
    raw = [
        "Server\u2013Driven UI (SDUI)",
        "React",
        "marketing",
        "🔥",
        "a",
    ]
    out = CvParseService._normalize_skills(raw)
    assert "Server-Driven UI (SDUI)" in out
    assert "React" in out
    assert "marketing" not in out
    assert all(len(s) >= 2 for s in out)


def test_parse_cv_ai_failure_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, json=None, timeout=None):  # noqa: A002
        raise RuntimeError("vision model down")

    import services.cv_parse_service as mod

    monkeypatch.setattr(mod.httpx, "post", boom)

    service = _service("unused")
    result, error, status = service.parse_upload(b"\x89PNG fake", "cv.jpg")

    assert result is None
    assert status == 502
    assert error is not None
    assert error["stage"] == "CV_AI_ANALYSIS"
