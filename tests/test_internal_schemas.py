"""Tests cho questionTypes normalization và plan schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.internal_schemas import (
    GeneratePlanRequest,
    GenerateQuestionsRequest,
    QuestionGenerationPlan,
    QuestionTypeDistributionItem,
)


def _base_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "ownerId": "11111111-1111-1111-1111-111111111111",
        "jobDescription": "Senior backend developer",
        "numberOfQuestions": 5,
        "questionTypes": ["Technical", "Behavioral"],
    }
    payload.update(overrides)
    return payload


def test_normalize_technical_and_system_design() -> None:
    req = GenerateQuestionsRequest.model_validate(
        _base_payload(questionTypes=["Technical", "System Design"])
    )
    assert req.question_types == ["technical", "system-design"]


def test_normalize_problem_solving_variants() -> None:
    req = GenerateQuestionsRequest.model_validate(
        _base_payload(questionTypes=["Problem-Solving", "problem solving"])
    )
    assert req.question_types == ["problem-solving", "problem-solving"]


def test_invalid_question_type_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        GenerateQuestionsRequest.model_validate(
            _base_payload(questionTypes=["Technical", "Culture Fit"])
        )
    assert "questionTypes không hợp lệ" in str(exc.value)


def test_normalize_underscore_variants() -> None:
    req = GeneratePlanRequest.model_validate(
        _base_payload(questionTypes=["system_design", "problem_solving"])
    )
    assert req.question_types == ["system-design", "problem-solving"]


def test_normalize_experience_level_aliases() -> None:
    from models.internal_schemas import normalize_experience_level

    assert normalize_experience_level("Senior") == "senior"
    assert normalize_experience_level("thuc-tap") == "intern"
    assert normalize_experience_level("entry-level") == "junior"


def test_question_generation_plan_parse() -> None:
    plan = QuestionGenerationPlan.model_validate(
        {
            "roleTitle": "Backend Developer",
            "summary": "Summary",
            "difficulty": "medium",
            "experienceLevel": "junior",
            "totalQuestions": 5,
            "skills": ["C#"],
            "questionTypeDistribution": [
                {"type": "technical", "count": 5, "reason": "JD"}
            ],
        }
    )
    assert plan.total_questions == 5
    assert plan.question_type_distribution[0].type == "technical"
