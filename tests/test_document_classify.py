"""SCRUM-466: unit tests cho document_classify (CV / KB pass rules)."""
from __future__ import annotations

from services.document_classify import check_pass_rules, normalize_document_type


def test_check_pass_cv_resume_it_ok() -> None:
    result = check_pass_rules("cv", "resume", True, None)
    assert result.success is True


def test_check_pass_cv_marketing_resume_rejects() -> None:
    result = check_pass_rules("cv", "resume", False, "CV Marketing không thuộc IT.")
    assert result.success is False
    assert result.stage == "DOC_CLASSIFY"
    assert "Marketing" in (result.detail or "")


def test_check_pass_cv_article_rejects() -> None:
    result = check_pass_rules("cv", "article", True, None)
    assert result.success is False


def test_check_pass_kb_documentation_ok() -> None:
    result = check_pass_rules("kb", "documentation", True, None)
    assert result.success is True


def test_check_pass_kb_resume_rejects() -> None:
    result = check_pass_rules("kb", "resume", True, None)
    assert result.success is False


def test_check_pass_jd_job_description_ok() -> None:
    result = check_pass_rules("jd", "job_description", True, None)
    assert result.success is True


def test_normalize_document_type_aliases() -> None:
    assert normalize_document_type("CV") == "resume"
    assert normalize_document_type("docs") == "documentation"
    assert normalize_document_type("blog") == "article"
