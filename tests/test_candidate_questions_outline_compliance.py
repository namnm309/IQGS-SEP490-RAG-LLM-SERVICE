"""SCRUM-452: audience=coach sinh đúng outline, không ép min 10 câu."""
from __future__ import annotations

from unittest.mock import MagicMock

from models.internal_schemas import (
    CandidateGenerateQuestionsFromPlanRequest,
    QuestionGenerationPlan,
    QuestionTypeDistributionItem,
    RecommendedQuestionOutlineItem,
)
from services.candidate_question_generation_service import CandidateQuestionGenerationService


def _plan(total: int = 9) -> QuestionGenerationPlan:
    outline = [
        RecommendedQuestionOutlineItem(
            order=i + 1,
            type="technical",
            difficulty=["easy", "medium", "hard"][i % 3],
            skill="C#",
            focusArea=f"Topic {i + 1}",
            goal="Assess",
        )
        for i in range(total)
    ]
    return QuestionGenerationPlan(
        roleTitle="Diagnostic",
        summary="Coach",
        difficulty="medium",
        experienceLevel="junior",
        totalQuestions=total,
        skills=["C#"],
        questionTypeDistribution=[
            QuestionTypeDistributionItem(type="technical", count=total, reason="blueprint")
        ],
        recommendedQuestionOutline=outline,
        notes="blueprint",
    )


def test_coach_user_message_uses_blueprint_count_not_min_ten() -> None:
    retrieval = MagicMock()
    retrieval.retrieve_system_only.return_value = []
    svc = CandidateQuestionGenerationService(
        retrieval=retrieval,
        client=MagicMock(),
        settings=type("S", (), {"request_timeout_seconds": 30, "chat_model": "t", "debug": False})(),
    )
    request = CandidateGenerateQuestionsFromPlanRequest(
        ownerId="00000000-0000-0000-0000-000000000001",
        jobDescription="Coach diagnostic",
        approvedPlan=_plan(9),
        audience="coach",
    )
    msg = svc._build_candidate_from_plan_user_message(request, [])
    assert "ĐÚNG 9 câu" in msg
    assert "tối thiểu 10" not in msg.lower()
    assert "recommendedQuestionOutline" in msg
    assert "Không gộp, không thêm, không bớt" in msg
    assert "Thiếu chunk SYSTEM vẫn sinh đủ câu" not in msg
    assert "bám chunk" in msg.lower() or "chunk SYSTEM" in msg.lower()
