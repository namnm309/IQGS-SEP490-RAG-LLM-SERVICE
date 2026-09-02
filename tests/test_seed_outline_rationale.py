"""SCRUM-427: seed rationale từ outline.goal."""
from __future__ import annotations

from models.internal_schemas import (
    GeneratedQuestionItem,
    QuestionGenerationPlan,
    RecommendedQuestionOutlineItem,
)
from services.question_generation_service import QuestionGenerationService


def test_seed_rationale_from_outline_overwrites_llm() -> None:
    plan = QuestionGenerationPlan(
        roleTitle="Dev",
        summary="s",
        difficulty="medium",
        experienceLevel="junior",
        totalQuestions=1,
        recommendedQuestionOutline=[
            RecommendedQuestionOutlineItem(
                order=1,
                type="technical",
                difficulty="medium",
                skill="SOLID",
                focusArea="SRP",
                goal="Đánh giá SOLID",
            )
        ],
    )
    q = GeneratedQuestionItem(
        question="Giải thích SRP?",
        question_type="technical",
        difficulty="medium",
        order=1,
        skill="SOLID",
        focus_area="SRP",
        rationale="LLM viết lý do khác hoàn toàn",
    )
    QuestionGenerationService._seed_rationale_from_outline([q], plan)
    assert q.rationale == "Đánh giá SOLID"


def test_seed_rationale_skips_empty_goal() -> None:
    plan = QuestionGenerationPlan(
        roleTitle="Dev",
        summary="s",
        difficulty="medium",
        experienceLevel="junior",
        totalQuestions=1,
        recommendedQuestionOutline=[
            RecommendedQuestionOutlineItem(
                order=1,
                type="technical",
                difficulty="medium",
                skill="SOLID",
                focusArea="SRP",
                goal="   ",
            )
        ],
    )
    q = GeneratedQuestionItem(
        question="Q?",
        question_type="technical",
        difficulty="easy",
        order=1,
        rationale="Giữ rationale LLM",
    )
    QuestionGenerationService._seed_rationale_from_outline([q], plan)
    assert q.rationale == "Giữ rationale LLM"
