"""Tests cho PlanGenerationService."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from models.internal_schemas import GeneratePlanRequest
from services.plan_generation_service import PlanGenerationService
from vectorstores.base import RetrievedChunk


def _valid_plan_json(
    total: int = 2,
    include_level: bool = True,
    experience_level: str = "senior",
) -> str:
    plan_body: dict = {
        "role_title": "Backend Developer",
        "summary": "Plan summary",
        "difficulty": "medium",
        "experience_level": experience_level,
        "total_questions": total,
        "skills": ["C#"],
        "question_type_distribution": [
            {"type": "technical", "count": total, "reason": "JD focus"}
        ],
        "difficulty_distribution": [
            {"difficulty": "medium", "count": total}
        ],
        "coverage": [
            {
                "skill": "C#",
                "question_count": total,
                "focus_areas": ["ASP.NET Core"],
            }
        ],
        "recommended_question_outline": [
            {
                "order": 1,
                "type": "technical",
                "difficulty": "medium",
                "skill": "C#",
                "focus_area": "DI",
                "goal": "Test DI knowledge",
            }
        ],
        "notes": "Test plan",
        "citations": [],
    }
    if include_level:
        plan_body["level"] = "medium"
    return json.dumps({"plan": plan_body})


@pytest.fixture
def settings() -> Settings:
    return Settings(
        DATABASE_URL="postgresql://postgres:postgres@localhost:5432/iqgs",
        BACKEND_INTERNAL_BASE_URL="http://localhost:5000",
        INTERNAL_API_KEY="test-secret",
        EMBEDDING_DIMENSION=768,
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
) -> PlanGenerationService:
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

    return PlanGenerationService(retrieval=retrieval, client=client, settings=settings)


def test_generate_plan_success(settings: Settings, sample_chunk: RetrievedChunk) -> None:
    service = _make_service(settings, [_valid_plan_json(2)], sample_chunk)
    request = GeneratePlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        numberOfQuestions=2,
        questionTypes=["technical"],
        skills=["C#"],
    )

    result = service.generate(request)

    assert result.success is True
    assert result.plan is not None
    assert result.plan.total_questions == 2
    assert result.plan.role_title == "Backend Developer"
    assert result.plan.level == "medium"
    assert result.plan.difficulty == "medium"
    assert result.plan.experience_level == "senior"


def test_generate_plan_fails_when_missing_difficulty(
    settings: Settings, sample_chunk: RetrievedChunk
) -> None:
    # JSON chỉ có level, không có difficulty — strict mode phải fail
    plan_only_level = json.dumps(
        {
            "plan": {
                "role_title": "Backend Developer",
                "summary": "Plan summary",
                "level": "hard",
                "experience_level": "junior",
                "total_questions": 2,
                "skills": ["C#"],
                "question_type_distribution": [
                    {"type": "technical", "count": 2, "reason": "JD focus"}
                ],
                "difficulty_distribution": [
                    {"difficulty": "hard", "count": 2}
                ],
                "coverage": [
                    {
                        "skill": "C#",
                        "question_count": 2,
                        "focus_areas": ["ASP.NET Core"],
                    }
                ],
                "recommended_question_outline": [
                    {
                        "order": 1,
                        "type": "technical",
                        "difficulty": "hard",
                        "skill": "C#",
                        "focus_area": "DI",
                        "goal": "Test DI knowledge",
                    }
                ],
                "notes": "Test plan",
                "citations": [],
            }
        }
    )
    service = _make_service(settings, [plan_only_level, plan_only_level], sample_chunk)
    request = GeneratePlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        numberOfQuestions=2,
        difficulty="hard",
        questionTypes=["technical"],
        skills=["C#"],
    )

    result = service.generate(request)

    assert result.success is False
    assert result.plan is None
    assert "difficulty" in (result.detail or "").lower()


def test_generate_plan_retries_when_llm_omits_experience_level(
    settings: Settings, sample_chunk: RetrievedChunk
) -> None:
    missing_exp_json = json.dumps(
        {
            "plan": {
                "role_title": "Backend Developer",
                "summary": "Plan summary",
                "difficulty": "medium",
                "level": "medium",
                "total_questions": 2,
                "skills": ["C#"],
                "question_type_distribution": [
                    {"type": "technical", "count": 2, "reason": "JD focus"}
                ],
                "difficulty_distribution": [
                    {"difficulty": "medium", "count": 2}
                ],
                "coverage": [
                    {
                        "skill": "C#",
                        "question_count": 2,
                        "focus_areas": ["ASP.NET Core"],
                    }
                ],
                "recommended_question_outline": [
                    {
                        "order": 1,
                        "type": "technical",
                        "difficulty": "medium",
                        "skill": "C#",
                        "focus_area": "DI",
                        "goal": "Test DI knowledge",
                    }
                ],
                "notes": "Test plan",
                "citations": [],
            }
        }
    )
    service = _make_service(
        settings,
        [missing_exp_json, _valid_plan_json(2, experience_level="mid")],
        sample_chunk,
    )
    request = GeneratePlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend Developer có 2–4 năm kinh nghiệm",
        numberOfQuestions=2,
        questionTypes=["technical"],
    )

    result = service.generate(request)

    assert result.success is True
    assert result.plan is not None
    assert result.plan.experience_level == "mid"
    assert service._client.chat.completions.create.call_count == 2


def test_generate_plan_validation_fail_missing_experience_level(
    settings: Settings, sample_chunk: RetrievedChunk
) -> None:
    missing_exp_json = json.dumps(
        {
            "plan": {
                "role_title": "Backend Developer",
                "summary": "Plan summary",
                "difficulty": "medium",
                "level": "medium",
                "total_questions": 2,
                "skills": ["C#"],
                "question_type_distribution": [
                    {"type": "technical", "count": 2, "reason": "JD focus"}
                ],
                "difficulty_distribution": [
                    {"difficulty": "medium", "count": 2}
                ],
                "coverage": [
                    {
                        "skill": "C#",
                        "question_count": 2,
                        "focus_areas": ["ASP.NET Core"],
                    }
                ],
                "recommended_question_outline": [
                    {
                        "order": 1,
                        "type": "technical",
                        "difficulty": "medium",
                        "skill": "C#",
                        "focus_area": "DI",
                        "goal": "Test DI knowledge",
                    }
                ],
                "notes": "Test plan",
                "citations": [],
            }
        }
    )
    service = _make_service(settings, [missing_exp_json, missing_exp_json], sample_chunk)
    request = GeneratePlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        numberOfQuestions=2,
        questionTypes=["technical"],
    )

    result = service.generate(request)

    assert result.success is False
    assert result.plan is None
    assert "experience_level" in (result.detail or "")


def test_generate_plan_validation_fail_wrong_difficulty(
    settings: Settings, sample_chunk: RetrievedChunk
) -> None:
    wrong_level_json = json.dumps(
        {
            "plan": {
                "role_title": "Backend Developer",
                "summary": "Plan summary",
                "difficulty": "easy",
                "level": "easy",
                "experience_level": "intern",
                "total_questions": 2,
                "skills": ["C#"],
                "question_type_distribution": [
                    {"type": "technical", "count": 2, "reason": "JD focus"}
                ],
                "difficulty_distribution": [
                    {"difficulty": "easy", "count": 2}
                ],
                "coverage": [
                    {
                        "skill": "C#",
                        "question_count": 2,
                        "focus_areas": ["ASP.NET Core"],
                    }
                ],
                "recommended_question_outline": [
                    {
                        "order": 1,
                        "type": "technical",
                        "difficulty": "easy",
                        "skill": "C#",
                        "focus_area": "DI",
                        "goal": "Test DI knowledge",
                    }
                ],
                "notes": "Test plan",
                "citations": [],
            }
        }
    )
    service = _make_service(settings, [wrong_level_json, wrong_level_json], sample_chunk)
    request = GeneratePlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        numberOfQuestions=2,
        difficulty="medium",
        questionTypes=["technical"],
    )

    result = service.generate(request)

    assert result.success is False
    assert result.plan is None
    assert "không khớp difficulty" in (result.detail or "")


def test_generate_plan_falls_back_to_jd_when_no_retrieved_chunks(
    settings: Settings,
) -> None:
    service = _make_service(settings, [_valid_plan_json(2)], None)
    request = GeneratePlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        numberOfQuestions=2,
        questionTypes=["technical"],
        skills=["C#"],
    )

    result = service.generate(request)

    assert result.success is True
    assert result.plan is not None
    assert result.plan.total_questions == 2
    assert service._client.chat.completions.create.call_count == 1


def test_generate_plan_retry_on_invalid_json(
    settings: Settings, sample_chunk: RetrievedChunk
) -> None:
    service = _make_service(
        settings,
        ["```not json```", _valid_plan_json(1)],
        sample_chunk,
    )
    request = GeneratePlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        numberOfQuestions=1,
        questionTypes=["technical"],
    )

    result = service.generate(request)

    assert result.success is True
    assert service._client.chat.completions.create.call_count == 2


def test_generate_plan_validation_fail_wrong_total(
    settings: Settings, sample_chunk: RetrievedChunk
) -> None:
    service = _make_service(settings, [_valid_plan_json(5), _valid_plan_json(5)], sample_chunk)
    request = GeneratePlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        numberOfQuestions=2,
        questionTypes=["technical"],
    )

    result = service.generate(request)

    assert result.success is False
    assert result.plan is None
    assert "totalQuestions" in (result.detail or "")
