"""Unit tests for practice-session-insight service (SCRUM-305)."""
from __future__ import annotations

import json

from models.internal_schemas import PracticeSessionInsightRequest, QuestionInsightSummary
from services.practice_session_insight_service import PracticeSessionInsightService


class _FakeCompletions:
    def create(self, **kwargs):  # noqa: ANN003
        payload = json.loads(kwargs["messages"][1]["content"])
        answered = int(payload.get("answeredCount") or 0)
        total = int(payload.get("totalQuestions") or 1)
        summaries = payload.get("questionSummaries") or []

        # Case ít câu trả lời → nhắc hoàn thành bài
        if answered < max(1, total // 3):
            content = json.dumps(
                {
                    "insightVi": (
                        f"Bạn mới trả lời {answered}/{total} câu. "
                        "Hãy luyện tập trả lời đầy đủ và bổ sung ví dụ cụ thể."
                    ),
                    "insightEn": (
                        f"You answered {answered}/{total} questions. "
                        "Focus on completing more questions with specific examples."
                    ),
                    "skillsToImproveVi": ["C# / .NET", "Giao tiếp kỹ thuật", "Kiến trúc hệ thống"],
                    "skillsToImproveEn": ["C# / .NET", "Technical communication", "System architecture"],
                },
                ensure_ascii=False,
            )
        else:
            # Case dimensionScores depth thấp → gợi ý depth-related skill
            has_low_depth = any(
                (s.get("dimensionScores") or {}).get("depth", 100) < 50 for s in summaries
            )
            skills_vi = ["Độ sâu kỹ thuật", "C# / .NET"] if has_low_depth else ["C# / .NET", "Refactoring"]
            skills_en = ["Technical depth", "C# / .NET"] if has_low_depth else ["C# / .NET", "Refactoring"]
            content = json.dumps(
                {
                    "insightVi": "Hiệu suất ổn định. Hãy bổ sung ví dụ định lượng và cấu trúc câu trả lời rõ hơn.",
                    "insightEn": "Solid performance. Add quantifiable examples and clearer answer structure.",
                    "skillsToImproveVi": skills_vi,
                    "skillsToImproveEn": skills_en,
                },
                ensure_ascii=False,
            )

        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def _service() -> PracticeSessionInsightService:
    return PracticeSessionInsightService(
        client=_FakeClient(),
        settings=type("S", (), {"chat_model": "test"})(),
    )


def test_insight_partial_session_mentions_completion_and_skills() -> None:
    result = _service().generate(
        PracticeSessionInsightRequest(
            overall_score=3.17,
            total_questions=30,
            answered_count=1,
            set_title="C#/.NET Developer",
            set_skills=["C#", "ASP.NET", "SQL"],
            question_summaries=[
                QuestionInsightSummary(
                    question_type="technical",
                    skill="C#",
                    score=95,
                    strengths=["Rõ ràng"],
                    improvements=["Thêm ví dụ"],
                )
            ],
        )
    )

    assert result.success is True
    assert result.insight_vi and "1/30" in result.insight_vi
    assert result.insight_en and "1/30" in result.insight_en
    assert 2 <= len(result.skills_to_improve_vi) <= 5
    assert 2 <= len(result.skills_to_improve_en) <= 5
    dumped = result.model_dump(by_alias=True)
    assert "insightVi" in dumped
    assert "skillsToImproveVi" in dumped


def test_insight_low_depth_dimension_suggests_depth_skill() -> None:
    result = _service().generate(
        PracticeSessionInsightRequest(
            overall_score=72,
            total_questions=5,
            answered_count=5,
            set_skills=["C#", "SQL"],
            question_summaries=[
                QuestionInsightSummary(
                    question_type="technical",
                    skill="C#",
                    score=60,
                    improvements=["Thiếu chi tiết"],
                    dimension_scores={"clarity": 70, "depth": 30},
                ),
                QuestionInsightSummary(
                    question_type="behavioral",
                    skill="Communication",
                    score=80,
                    dimension_scores={"clarity": 85, "depth": 75},
                ),
            ],
        )
    )

    assert result.success is True
    assert result.insight_vi and result.insight_en
    assert any("độ sâu" in s.lower() or "depth" in s.lower() for s in result.skills_to_improve_vi + result.skills_to_improve_en)


def test_insight_invalid_total_questions_fails() -> None:
    result = _service().generate(
        PracticeSessionInsightRequest(
            overall_score=0,
            total_questions=0,
            answered_count=0,
        )
    )
    assert result.success is False
    assert result.error
