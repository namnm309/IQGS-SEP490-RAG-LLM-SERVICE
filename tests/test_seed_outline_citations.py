"""SCRUM-426: seed + skip_locked JD citations từ outline."""
from __future__ import annotations

from models.internal_schemas import (
    GeneratedQuestionItem,
    PlanCitationItem,
    QuestionGenerationPlan,
    RecommendedQuestionOutlineItem,
)
from services.question_generation_service import QuestionGenerationService
from services.rag_context_helpers import JD_SOURCE_FILE


def test_seed_citations_from_outline_copies_chunk_index() -> None:
    locked = PlanCitationItem(
        knowledgeBase="hr",
        sourceFile=JD_SOURCE_FILE,
        chunkIndex=3,
        excerpt="Nắm vững OOP, design patterns và SOLID.",
        origin="HR",
        usedFor=["why-asked"],
    )
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
                goal="x",
                citations=[locked],
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
        citations=[],
    )
    QuestionGenerationService._seed_citations_from_outline([q], plan)
    assert q.citations
    assert q.citations[0].chunk_index == 3
    assert "OOP" in (q.citations[0].excerpt or "")


def test_assign_jd_skips_locked() -> None:
    jd = """Yêu cầu:
- Nắm vững OOP và SOLID.
- Tiếng Anh tốt.
"""
    locked = PlanCitationItem(
        knowledgeBase="hr",
        sourceFile=JD_SOURCE_FILE,
        chunkIndex=0,
        excerpt="Nắm vững OOP và SOLID.",
        origin="HR",
        usedFor=["why-asked"],
    )
    q = GeneratedQuestionItem(
        question="SOLID?",
        question_type="technical",
        difficulty="easy",
        skill="SOLID",
        focus_area="SRP",
        citations=[locked],
    )
    QuestionGenerationService._assign_jd_primary_citations(
        [q], jd, skip_locked=True
    )
    assert q.citations[0].chunk_index == 0
    assert q.citations[0].excerpt == "Nắm vững OOP và SOLID."
