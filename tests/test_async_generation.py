"""Unit tests cho async generation + callback payload."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from models.internal_schemas import (
    GeneratePlanRequest,
    GenerateQuestionsFromPlanRequest,
    QuestionGenerationPlan,
    QuestionTypeDistributionItem,
)
from services.async_generation_service import AsyncGenerationService


@pytest.fixture
def callback_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def plan_service() -> MagicMock:
    return MagicMock()


@pytest.fixture
def question_service() -> MagicMock:
    return MagicMock()


@pytest.fixture
def async_service(
    plan_service: MagicMock,
    question_service: MagicMock,
    callback_client: MagicMock,
) -> AsyncGenerationService:
    return AsyncGenerationService(plan_service, question_service, callback_client)


def test_run_plan_success_callbacks_payload(
    async_service: AsyncGenerationService,
    plan_service: MagicMock,
    callback_client: MagicMock,
) -> None:
    mock_plan = QuestionGenerationPlan(
        roleTitle="Backend Developer",
        summary="Plan summary",
        experienceLevel="mid",
        totalQuestions=2,
        questionTypeDistribution=[
            QuestionTypeDistributionItem(type="technical", count=2, reason="JD")
        ],
    )
    plan_service.generate.return_value = MagicMock(
        success=True,
        plan=mock_plan,
        processing_time_ms=1500.0,
        error=None,
        detail=None,
        stage=None,
        errors=[],
    )
    request = GeneratePlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend role",
        numberOfQuestions=2,
    )

    async_service.run_plan("job-1", request)

    callback_client.notify_generation_result.assert_called_once()
    job_id, payload = callback_client.notify_generation_result.call_args[0]
    assert job_id == "job-1"
    assert payload["phase"] == "PLAN"
    assert payload["success"] is True
    assert payload["plan"]["roleTitle"] == "Backend Developer"


def test_run_plan_failure_callbacks_error(
    async_service: AsyncGenerationService,
    plan_service: MagicMock,
    callback_client: MagicMock,
) -> None:
    plan_service.generate.return_value = MagicMock(
        success=False,
        plan=None,
        processing_time_ms=100.0,
        error="Lỗi sinh plan",
        detail="LLM timeout",
        stage="PLAN_GENERATION",
        errors=["LLM timeout"],
    )
    request = GeneratePlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend role",
        numberOfQuestions=2,
    )

    async_service.run_plan("job-2", request)

    payload = callback_client.notify_generation_result.call_args[0][1]
    assert payload["success"] is False
    assert payload["stage"] == "PLAN_GENERATION"
    assert payload["source"] == "RAG"


def test_run_questions_success_callbacks_payload(
    async_service: AsyncGenerationService,
    question_service: MagicMock,
    callback_client: MagicMock,
) -> None:
    mock_question = MagicMock()
    mock_question.model_dump.return_value = {
        "question": "Explain DI?",
        "questionType": "technical",
        "difficulty": "medium",
    }
    question_service.generate_from_plan.return_value = MagicMock(
        success=True,
        questions=[mock_question],
        processing_time_ms=2000.0,
        error=None,
    )
    approved_plan = QuestionGenerationPlan(
        roleTitle="Backend Developer",
        summary="Plan",
        experienceLevel="senior",
        totalQuestions=1,
        questionTypeDistribution=[
            QuestionTypeDistributionItem(type="technical", count=1, reason="JD")
        ],
    )
    request = GenerateQuestionsFromPlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend role",
        approvedPlan=approved_plan,
    )

    async_service.run_questions_from_plan("job-3", request)

    payload = callback_client.notify_generation_result.call_args[0][1]
    assert payload["phase"] == "QUESTIONS"
    assert payload["success"] is True
    assert len(payload["questions"]) == 1
