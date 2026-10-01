"""Studio regen 1 câu có HR_REGEN_NOTE: được đổi chủ đề theo lưu ý HR (vd. OOP)."""
from __future__ import annotations

import json

from models.internal_schemas import (
    GenerateQuestionsFromPlanRequest,
    QuestionTypeDistributionItem,
)
from services.question_generation_service import (
    _build_plan_retrieve_query,
    _extract_hr_regen_note,
)
from tests.test_hg01_slot_coherence import OWNER_ID, _plan, _service, _slot

_LONG_PREFIX = (
    "STUDIO_UI=1; STUDIO_REGEN=1; HR_REGEN_NOTE=Hỏi về OOP, tính đa hình; "
    + "Studio project x; " * 60
)


def _oop_reply() -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "order": 1,
                    "question": "Phân biệt overloading và overriding trong OOP?",
                    "skill": "OOP",
                    "focus_area": "Polymorphism",
                    "rationale": "Kiểm tra hiểu đa hình",
                    "question_type": "technical",
                }
            ]
        }
    )


def test_extract_hr_regen_note() -> None:
    assert _extract_hr_regen_note(_LONG_PREFIX) == "Hỏi về OOP, tính đa hình"
    assert _extract_hr_regen_note("STUDIO_REGEN=1; CONTENT_MODE=Mixed") is None


def test_retrieve_query_starts_with_regen_note_even_when_hr_note_is_long() -> None:
    plan = _plan([_slot(1, "technical", "ASP.NET Core", "Đánh giá middleware", "Middleware")])
    query = _build_plan_retrieve_query(plan, _LONG_PREFIX)
    assert query.startswith("Hỏi về OOP, tính đa hình")


def test_regen_with_hr_note_keeps_new_topic_and_does_not_regen_back() -> None:
    plan = _plan(
        [_slot(1, "technical", "ASP.NET Core", "Đánh giá middleware", "Middleware")],
        [QuestionTypeDistributionItem(type="technical", count=1)],
    )
    service = _service(_oop_reply())

    result = service.generate_from_plan(
        GenerateQuestionsFromPlanRequest(
            ownerId=OWNER_ID,
            jobDescription="Backend role",
            approvedPlan=plan,
            hrNote="STUDIO_UI=1; STUDIO_REGEN=1; HR_REGEN_NOTE=Hỏi về OOP; CONTENT_MODE=Mixed",
        )
    )

    q = result.questions[0]
    assert "OOP" in q.question
    assert q.skill == "OOP"
    assert q.focus_area == "Polymorphism"
    assert q.topic_overridden is True
    assert q.model_dump(by_alias=True)["topicOverridden"] is True
    # Chỉ gọi LLM 1 lần — không regen kéo về ASP.NET Core.
    assert service._client.chat.completions.create.call_count == 1


def test_regen_without_hr_note_still_locks_to_slot() -> None:
    plan = _plan(
        [_slot(1, "technical", "ASP.NET Core", "Đánh giá middleware", "Middleware")],
        [QuestionTypeDistributionItem(type="technical", count=1)],
    )
    # LLM lệch sang OOP 2 lần liền (3 lượt retry) → vẫn bị khóa nhãn về slot.
    service = _service(*([_oop_reply()] * 4))

    result = service.generate_from_plan(
        GenerateQuestionsFromPlanRequest(
            ownerId=OWNER_ID,
            jobDescription="Backend role",
            approvedPlan=plan,
            hrNote="STUDIO_UI=1; STUDIO_REGEN=1; CONTENT_MODE=Mixed",
        )
    )

    q = result.questions[0]
    assert q.skill == "ASP.NET Core"
    assert q.topic_overridden is False
