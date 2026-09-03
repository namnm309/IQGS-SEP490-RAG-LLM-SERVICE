"""SCRUM-421: Tests waterfall provenance trên câu hỏi."""
from __future__ import annotations

from models.internal_schemas import GeneratedQuestionItem, QuestionCitationItem
from services.question_provenance_validator import apply_provenance_to_questions
from services.rag_context_helpers import JD_SOURCE_FILE
from vectorstores.base import RetrievedChunk

JD = "Yêu cầu kinh nghiệm Git, branching và merge request."


def _jd_citation(excerpt: str = "kinh nghiệm Git") -> QuestionCitationItem:
    return QuestionCitationItem(
        knowledge_base="hr",
        source_file=JD_SOURCE_FILE,
        chunk_index=0,
        excerpt=excerpt,
    )


def _system_chunk(content: str, file_name: str = "git-guide.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        document_id="sys-1",
        chunk_index=0,
        content=content,
        scope="SYSTEM",
        owner_id=None,
        score=0.92,
        metadata={"fileName": file_name},
    )


def test_git_skill_gets_hr_and_system_citations() -> None:
    git_chunk = _system_chunk(
        "Git branching: feature branch, merge request, rebase workflow."
    )
    q = GeneratedQuestionItem(
        question="Giải thích workflow Git branching?",
        question_type="technical",
        difficulty="medium",
        skill="Git",
        focus_area="Branching",
        citations=[_jd_citation()],
    )
    result = apply_provenance_to_questions(
        [q], chunks=[git_chunk], job_description=JD
    )[0]

    origins = {c.origin for c in result.citations}
    assert "HR" in origins
    assert "SYSTEM" in origins
    system_cits = [c for c in result.citations if c.origin == "SYSTEM"]
    assert system_cits[0].used_for == ["technical-body"]
    assert result.missing_admin_warning is False
    assert result.source_provenance is not None
    assert result.source_provenance.primary_origin in ("HR", "SYSTEM")


def test_no_admin_chunk_sets_warning_and_llm_provenance() -> None:
    q = GeneratedQuestionItem(
        question="So sánh merge và rebase trong Git?",
        question_type="technical",
        difficulty="medium",
        skill="Git",
        sample_answer="Merge tạo merge commit; rebase viết lại history.",
        citations=[_jd_citation()],
    )
    result = apply_provenance_to_questions([q], chunks=[], job_description=JD)[0]

    assert result.missing_admin_warning is True
    assert not any(c.origin == "SYSTEM" for c in result.citations)
    assert result.source_provenance is not None
    llm_items = [
        i for i in result.source_provenance.items if i.origin == "LLM"
    ]
    assert len(llm_items) >= 1


def test_auto_attach_system_when_llm_forgot_citation() -> None:
    git_chunk = _system_chunk("Git stash lưu thay đổi tạm thời trước khi switch branch.")
    q = GeneratedQuestionItem(
        question="Khi nào dùng git stash?",
        question_type="technical",
        difficulty="medium",
        skill="Git",
        citations=[_jd_citation("Git")],
    )
    result = apply_provenance_to_questions(
        [q], chunks=[git_chunk], job_description=JD
    )[0]

    assert any(c.origin == "SYSTEM" for c in result.citations)
    assert result.missing_admin_warning is False


def test_two_questions_prefer_different_system_chunks() -> None:
    chunk_a = RetrievedChunk(
        document_id="sys-a",
        chunk_index=0,
        content="Git pull fetches and merges remote branch changes.",
        scope="SYSTEM",
        owner_id=None,
        score=0.9,
        metadata={"fileName": "git-pull.pdf"},
    )
    chunk_b = RetrievedChunk(
        document_id="sys-b",
        chunk_index=1,
        content="Git rebase rewrites commit history onto another base.",
        scope="SYSTEM",
        owner_id=None,
        score=0.9,
        metadata={"fileName": "git-rebase.pdf"},
    )
    q1 = GeneratedQuestionItem(
        question="Giải thích git pull?",
        question_type="technical",
        difficulty="medium",
        skill="Git",
        focus_area="pull",
        citations=[_jd_citation()],
    )
    q2 = GeneratedQuestionItem(
        question="Giải thích git rebase?",
        question_type="technical",
        difficulty="medium",
        skill="Git",
        focus_area="rebase",
        citations=[_jd_citation()],
    )
    results = apply_provenance_to_questions(
        [q1, q2], chunks=[chunk_a, chunk_b], job_description=JD
    )
    keys = set()
    for r in results:
        for c in r.citations:
            if c.origin == "SYSTEM":
                keys.add((c.source_file, c.chunk_index))
    assert len(keys) >= 2
