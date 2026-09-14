"""SCRUM-447/452: evaluate coach bắt buộc 3 dimension, không kết luận level."""
from __future__ import annotations

import json

from models.internal_schemas import EvaluateAnswerRequest
from services.evaluate_answer_service import EvaluateAnswerService


class _FakeCompletions:
    def create(self, **kwargs):  # noqa: ANN003
        content = json.dumps(
            {
                "score": None,
                "strengths": ["Nêu đúng DI"],
                "improvements": ["Thiếu ví dụ lifetime"],
                "suggestion": "Thêm so sánh Singleton/Scoped.",
                "dimensionScores": {
                    "correctness": 80,
                    "relevance": 70,
                    "clarity": 60,
                },
            },
            ensure_ascii=False,
        )
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def test_coach_mode_returns_three_dimensions() -> None:
    service = EvaluateAnswerService(
        client=_FakeClient(),
        settings=type("S", (), {"chat_model": "test", "temperature": 0.2})(),
    )
    result = service.evaluate(
        EvaluateAnswerRequest(
            question="Explain DI in ASP.NET Core.",
            evaluation_criteria=["Correctness", "Relevance", "Clarity"],
            candidate_answer=(
                "DI registers services in the container so controllers receive "
                "abstractions instead of constructing dependencies themselves."
            ),
            scoring_mode="coach",
            skill="ASP.NET Core",
            question_type="technical",
        )
    )
    assert result.success is True
    assert result.dimension_scores is not None
    assert set(result.dimension_scores) >= {"correctness", "relevance", "clarity"}
    dumped = json.dumps(result.model_dump(by_alias=True)).lower()
    assert "junior" not in dumped
    assert "senior" not in dumped
