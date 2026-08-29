"""Mocked tests for recommend interview configuration validator + service."""
from __future__ import annotations

from unittest.mock import MagicMock

from models.internal_schemas import (
    JobProfileInput,
    RecommendInterviewConfigurationRequest,
)
from services.recommend_configuration_validator import validate_and_normalize_recommendation
from services.recommend_interview_configuration_service import (
    RecommendInterviewConfigurationService,
)


def test_validator_normalizes_legacy_system_design_category_to_technical():
    raw = {
        "recommendedConfiguration": {
            "numberOfQuestions": 10,
            "difficulty": "Hard",
            "questionDistribution": [
                {"category": "system_design", "percentage": 50, "questionCount": 5},
                {"category": "behavioral", "percentage": 50, "questionCount": 5},
            ],
            "focusAreas": [
                {
                    "name": "ASP.NET Core",
                    "weight": 100,
                    "description": "Core backend",
                    "sourceReason": "JD lists ASP.NET Core.",
                    "orderIndex": 0,
                }
            ],
            "questionStyles": ["System Design", "debugging"],
            "codingTaskTypes": ["SYSTEM_DESIGN", "BUG_DETECTION"],
        }
    }

    normalized, errors = validate_and_normalize_recommendation(raw, default_total=10)

    assert errors == []
    assert normalized["difficulty"] == "hard"
    cats = {d["category"] for d in normalized["questionDistribution"]}
    assert cats <= {"technical", "behavioral", "situational"}
    assert "system_design" not in cats
    assert "system_design" in normalized["questionStyles"]
    assert "SYSTEM_DESIGN" not in normalized["codingTaskTypes"]
    assert normalized["codingTaskTypes"] == ["BUG_DETECTION"]
    assert sum(d["questionCount"] for d in normalized["questionDistribution"]) == 10


def test_validator_rejects_empty_focus_areas():
    raw = {
        "recommendedConfiguration": {
            "numberOfQuestions": 5,
            "difficulty": "medium",
            "questionDistribution": [
                {"category": "technical", "percentage": 100, "questionCount": 5},
            ],
            "focusAreas": [],
            "questionStyles": [],
            "codingTaskTypes": [],
            "codingTasksRecommended": False,
        }
    }

    _, errors = validate_and_normalize_recommendation(raw, default_total=5)
    assert any("focusAreas" in e for e in errors)


def test_recommend_service_empty_jd_returns_error():
    service = RecommendInterviewConfigurationService(
        retrieval=MagicMock(),
        client=MagicMock(),
        settings=MagicMock(),
    )
    result = service.recommend(
        RecommendInterviewConfigurationRequest(
            owner_id="00000000-0000-0000-0000-000000000001",
            job_description="   ",
            job_profile=JobProfileInput(),
        )
    )
    assert result.success is False
    assert result.stage == "RECOMMEND_CONFIG"
