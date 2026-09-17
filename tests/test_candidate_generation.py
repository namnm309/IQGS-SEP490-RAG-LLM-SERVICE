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
    """jd_practice vẫn được sinh khi thiếu chunk; coach thì không (xem test SCRUM-458)."""
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
        audience="jd_practice",
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


def test_coach_generate_from_plan_empty_retrieval_inferred(settings: Settings) -> None:
    """Coach thiếu KB vẫn sinh LLM và gắn kbSource=inferred."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = _valid_questions_json(1)
    client.chat.completions.create.return_value = MagicMock(choices=[choice])

    retrieval = MagicMock()
    retrieval.retrieve_system_only.return_value = []

    service = CandidateQuestionGenerationService(
        retrieval=retrieval, client=client, settings=settings
    )
    request = CandidateGenerateQuestionsFromPlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Coach diagnostic",
        audience="coach",
        approvedPlan=_approved_plan(1),
    )
    result = service.generate_from_plan(request)
    assert result.success is True
    assert len(result.questions) == 1
    assert result.kb_source == "inferred"
    client.chat.completions.create.assert_called_once()
    user_msg = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "Thiếu chunk SYSTEM" in user_msg
    assert "suy luận" in user_msg.lower() or "inferred" in user_msg.lower()
    # Query extra phải mang skill/topic từ outline
    call_kwargs = retrieval.retrieve_system_only.call_args.kwargs
    extra = call_kwargs.get("query_extra") or ""
    assert "C#" in extra or "DI" in extra


def test_coach_generate_from_plan_with_chunks_grounds_prompt(settings: Settings) -> None:
    """Có chunk thì LLM được gọi, prompt bám SYSTEM, kbSource=system."""
    from vectorstores.base import RetrievedChunk

    client = MagicMock()
    choice = MagicMock()
    choice.message.content = _valid_questions_json(1)
    client.chat.completions.create.return_value = MagicMock(choices=[choice])

    chunk = RetrievedChunk(
        document_id="doc-tech",
        chunk_index=0,
        content="Dependency Injection trong ASP.NET Core.",
        scope="SYSTEM",
        owner_id=None,
        score=0.91,
        metadata={"fileName": "dotnet.md", "documentType": "InternalStack"},
    )
    retrieval = MagicMock()
    retrieval.retrieve_system_only.return_value = [chunk]

    service = CandidateQuestionGenerationService(
        retrieval=retrieval, client=client, settings=settings
    )
    request = CandidateGenerateQuestionsFromPlanRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Coach diagnostic",
        audience="coach",
        approvedPlan=_approved_plan(1),
    )
    result = service.generate_from_plan(request)
    assert result.success is True
    assert len(result.questions) == 1
    assert result.kb_source == "system"
    user_msg = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "ĐÚNG 1 câu" in user_msg
    assert "chunk SYSTEM" in user_msg.lower() or "bám chunk" in user_msg.lower()
    assert "Thiếu chunk SYSTEM — sinh từ blueprint/CV" not in user_msg
    assert "Dependency Injection" in user_msg or "HỆ THỐNG" in user_msg
