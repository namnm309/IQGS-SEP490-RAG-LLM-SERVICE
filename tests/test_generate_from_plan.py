"""Tests cho generate_from_plan."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from models.internal_schemas import (
    GenerateQuestionsFromPlanRequest,
    QuestionGenerationPlan,
    QuestionTypeDistributionItem,
    RecommendedQuestionOutlineItem,
)
from services.question_generation_service import QuestionGenerationService
from vectorstores.base import RetrievedChunk


def _approved_plan(total: int = 1) -> QuestionGenerationPlan:
    return QuestionGenerationPlan(
        roleTitle="Backend Developer",
        summary="Approved plan",
        difficulty="medium",
        experienceLevel="senior",
        totalQuestions=total,
        skills=["C#"],
        questionTypeDistribution=[
            QuestionTypeDistributionItem(type="technical", count=total, reason="JD")
        ],
        recommendedQuestionOutline=[
            RecommendedQuestionOutlineItem(
                order=1,
                type="technical",
                difficulty="medium",
                skill="C#",
                focusArea="DI",
                goal="Test DI",
            )
        ],
        notes="HR approved",
    )


def _valid_questions_json(total: int = 1) -> str:
    questions = []
    for i in range(1, total + 1):
        questions.append(
            {
                "order": i,
                "question": f"Câu hỏi {i} về DI?",
                "question_type": "technical",
                "difficulty": "medium",
                "skill": "C#",
                "focus_area": "DI",
                "rationale": "Kiểm tra DI",
                "sample_answer": "DI giúp tách coupling.",
                "evaluation_criteria": ["Giải thích được DI"],
                "citations": [],
            }
        )
    return json.dumps({"questions": questions})


@pytest.fixture
def settings() -> Settings:
    return Settings(
        DATABASE_URL="postgresql://postgres:postgres@localhost:5432/iqgs",
        BACKEND_INTERNAL_BASE_URL="http://localhost:5000",
        INTERNAL_API_KEY="test-secret",
        EMBEDDING_DIMENSION=768,
        DEBUG=False,
    )


@pytest.fixture
def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc-1",
        chunk_index=0,
        content="ASP.NET Core hỗ trợ dependency injection built-in.",
        scope="SYSTEM",
        owner_id=None,
        score=0.9,
        metadata={"fileName": "rubric.pdf"},
    )


def _make_service(
    settings: Settings,
    llm_responses: list[str],
    sample_chunk: RetrievedChunk | None,
) -> QuestionGenerationService:
    client = MagicMock()
    choices = []
    for content in llm_responses:
        choice = MagicMock()
        choice.message.content = content
        choices.append(choice)
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[choice]) for choice in choices
    ]

    retrieval = MagicMock()
    retrieval.retrieve_for_job.return_value = ([sample_chunk], []) if sample_chunk else ([], [])

    return QuestionGenerationService(retrieval=retrieval, client=client, settings=settings)


def test_generate_from_plan_success(
    settings: Settings, sample_chunk: RetrievedChunk
) -> None:
    service = _make_service(settings, [_valid_questions_json(1)], sample_chunk)
    request = GenerateQuestionsFromPlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        approvedPlan=_approved_plan(1),
    )

    result = service.generate_from_plan(request)

    assert result.success is True
    assert len(result.questions) == 1
    assert result.questions[0].order == 1
    assert result.questions[0].skill == "C#"


def test_generate_from_plan_falls_back_to_plan_when_no_retrieved_chunks(
    settings: Settings,
) -> None:
    service = _make_service(settings, [_valid_questions_json(1)], None)
    request = GenerateQuestionsFromPlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        approvedPlan=_approved_plan(1),
    )

    result = service.generate_from_plan(request)

    assert result.success is True
    assert len(result.questions) == 1
    # SCRUM-392/394: không có KB chunk vẫn phải có citation JD + excerpt
    assert len(result.questions[0].citations) == 1
    jd_cit = result.questions[0].citations[0]
    assert jd_cit.source_file == "job-description"
    assert jd_cit.excerpt
    assert "Backend" in jd_cit.excerpt or ".NET" in jd_cit.excerpt
    assert service._client.chat.completions.create.call_count == 1


def test_generate_from_plan_retry_on_invalid_json(
    settings: Settings, sample_chunk: RetrievedChunk
) -> None:
    service = _make_service(
        settings,
        ["not json", _valid_questions_json(1)],
        sample_chunk,
    )
    request = GenerateQuestionsFromPlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        approvedPlan=_approved_plan(1),
    )

    result = service.generate_from_plan(request)

    assert result.success is True
    assert service._client.chat.completions.create.call_count == 2


def test_generate_from_plan_count_mismatch(
    settings: Settings, sample_chunk: RetrievedChunk
) -> None:
    service = _make_service(settings, [_valid_questions_json(2)], sample_chunk)
    request = GenerateQuestionsFromPlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        approvedPlan=_approved_plan(1),
    )

    result = service.generate_from_plan(request)

    assert result.success is False
    assert result.error is not None
    assert "không khớp" in result.error.lower() or "totalQuestions" in result.error
