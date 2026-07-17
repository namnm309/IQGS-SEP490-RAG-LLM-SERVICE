"""Unit tests for evaluate-answer service (SCRUM-281) — câu trả lời tốt vs yếu."""
from __future__ import annotations

import json

from models.internal_schemas import EvaluateAnswerRequest
from services.evaluate_answer_service import EvaluateAnswerService


class _FakeCompletions:
    def create(self, **kwargs):  # noqa: ANN003
        payload = json.loads(kwargs["messages"][1]["content"])
        answer = (payload.get("candidateAnswer") or "").lower()

        # Câu trả lời yếu / ngắn → điểm thấp
        if len(answer) < 40 or "không biết" in answer or "i don't know" in answer:
            content = json.dumps(
                {
                    "score": 35,
                    "strengths": ["Có cố gắng trả lời"],
                    "improvements": ["Thiếu chi tiết kỹ thuật", "Không có ví dụ cụ thể"],
                    "suggestion": "Bổ sung giải thích rõ ràng hơn kèm ví dụ thực tế.",
                    "dimensionScores": {"clarity": 40, "depth": 25},
                },
                ensure_ascii=False,
            )
        else:
            content = json.dumps(
                {
                    "score": 88,
                    "strengths": ["Giải thích rõ ràng", "Có ví dụ thực tế", "Cấu trúc tốt"],
                    "improvements": ["Có thể nêu thêm trade-off"],
                    "suggestion": "Thêm so sánh ngắn với phương án thay thế.",
                    "dimensionScores": {"clarity": 90, "depth": 85},
                },
                ensure_ascii=False,
            )

        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def _service() -> EvaluateAnswerService:
    return EvaluateAnswerService(
        client=_FakeClient(),
        settings=type("S", (), {"chat_model": "test"})(),
    )


def test_evaluate_strong_answer_returns_high_score() -> None:
    result = _service().evaluate(
        EvaluateAnswerRequest(
            question="Explain React virtual DOM.",
            evaluation_criteria=["Clarity", "Technical accuracy", "Examples"],
            candidate_answer=(
                "React uses a virtual DOM tree and diffs it against the previous tree "
                "so only changed nodes are updated in the real DOM for better performance."
            ),
            sample_answer="Internal sample — must not appear in response fields.",
            skill="React",
            question_type="technical",
        )
    )

    assert result.success is True
    assert result.score is not None and result.score >= 70
    assert len(result.strengths) >= 1
    assert len(result.improvements) >= 1
    assert result.suggestion
    assert result.dimension_scores is not None
    # AC-04: không trả sampleAnswer trong response
    dumped = result.model_dump(by_alias=True)
    assert "sampleAnswer" not in dumped


def test_evaluate_weak_answer_returns_low_score() -> None:
    result = _service().evaluate(
        EvaluateAnswerRequest(
            question="Explain React virtual DOM.",
            evaluation_criteria=["Clarity", "Technical accuracy"],
            candidate_answer="Không biết.",
            question_type="technical",
        )
    )

    assert result.success is True
    assert result.score is not None and result.score < 50
    assert len(result.improvements) >= 1
    assert result.suggestion
