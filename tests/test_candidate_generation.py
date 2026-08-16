"""Tests luồng Candidate generate — prompt riêng, SYSTEM-only, 0 chunk vẫn sinh."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from models.internal_schemas import (
    CandidateGenerateQuestionsFromPlanRequest,
    QuestionGenerationPlan,
    QuestionTypeDistributionItem,
    RecommendedQuestionOutlineItem,
)
from services.candidate_generation_prompts import (
    candidate_questions_system_prompt,
)
from services.candidate_question_generation_service import CandidateQuestionGenerationService
from services.question_generation_service import QUESTIONS_FROM_PLAN_SYSTEM_PROMPT


def _approved_plan(total: int = 1) -> QuestionGenerationPlan:
    return QuestionGenerationPlan(
        roleTitle="CV check",
        summary="Coach",
        difficulty="medium",
        experienceLevel="junior",
        totalQuestions=total,
        skills=["C#"],
        questionTypeDistribution=[
            QuestionTypeDistributionItem(type="technical", count=total, reason="CV")
        ],
        recommendedQuestionOutline=[
            RecommendedQuestionOutlineItem(
                order=1,
                type="technical",
                difficulty="medium",
                skill="C#",
                focusArea="DI",
                goal="Test knowledge",
            )
        ],
        notes="coach",
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


def test_candidate_prompt_is_not_hr_prompt() -> None:
    text = candidate_questions_system_prompt("Vietnamese")
    assert "JD là nguồn CHÍNH" not in text
    assert "JD là nguồn CHÍNH" in QUESTIONS_FROM_PLAN_SYSTEM_PROMPT
    assert "ỨNG VIÊN" in text
    assert "Tối thiểu 10" in text or "tối thiểu 10" in text
    assert "TĂNG DẦN" in text or "tăng dần" in text


def test_candidate_generate_from_plan_zero_chunks(settings: Settings) -> None:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = _valid_questions_json(1)
    client.chat.completions.create.return_value = MagicMock(choices=[choice])

    retrieval = MagicMock()
    retrieval.retrieve_system_only.return_value = []
    retrieval.retrieve_for_job.side_effect = AssertionError("Candidate không được gọi retrieve_for_job HR")

    service = CandidateQuestionGenerationService(
        retrieval=retrieval, client=client, settings=settings
    )
    request = CandidateGenerateQuestionsFromPlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="unused",
        cvContext="Skills: C#",
        audience="coach",
        approvedPlan=_approved_plan(1),
    )
    result = service.generate_from_plan(request)
    assert result.success is True
    assert len(result.questions) == 1
    retrieval.retrieve_system_only.assert_called_once()
    retrieval.retrieve_for_job.assert_not_called()
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert "JD là nguồn CHÍNH" not in messages[0]["content"]
    assert "ỨNG VIÊN" in messages[0]["content"] or "ứng viên" in messages[0]["content"].lower()
