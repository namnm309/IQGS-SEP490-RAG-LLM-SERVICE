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
        settings=type("S", (), {"chat_model": "test", "temperature": 0.2})(),
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


# ── Chấm theo rubric (từng tiêu chí) ─────────────────────────────────────────


class _RubricCompletions:
    """Fake LLM: trả criterionScores theo nội dung do test cài sẵn."""

    def __init__(self, contents: list[dict]) -> None:
        self._contents = contents
        self.calls: list[dict] = []

    def create(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._contents) - 1)
        message = type("Message", (), {"content": json.dumps(self._contents[index])})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


def _rubric_service(contents: list[dict], temperature: float = 0.3):
    completions = _RubricCompletions(contents)
    client = type("C", (), {"chat": type("Chat", (), {"completions": completions})()})()
    service = EvaluateAnswerService(
        client=client,
        settings=type("S", (), {"chat_model": "test", "temperature": temperature})(),
    )
    return service, completions


def _rubric_request():
    from models.internal_schemas import RubricCriterionInput

    return EvaluateAnswerRequest(
        question="Explain React virtual DOM.",
        candidate_answer="React keeps a virtual tree and diffs it to update only changed nodes.",
        rubricCriteria=[
            RubricCriterionInput(code="C1", label="Accuracy", weight=60, anchors={"50": "ok"}),
            RubricCriterionInput(code="C2", label="Examples", weight=40, anchors={"50": "ok"}),
        ],
    )


def test_rubric_mode_returns_criterion_scores_and_uses_rubric_prompt() -> None:
    service, completions = _rubric_service(
        [
            {
                "criterionScores": [{"code": "C1", "score": 80}, {"code": "C2", "score": 50}],
                "strengths": ["Đúng ý chính"],
                "improvements": ["Thiếu ví dụ"],
                "suggestion": "Thêm ví dụ.",
            }
        ]
    )
    result = service.evaluate(_rubric_request())

    assert result.success is True
    assert result.criterion_scores == {"C1": 80.0, "C2": 50.0}
    # 80*0.6 + 50*0.4 = 68
    assert result.score == 68.0
    assert "rubricCriteria" in completions.calls[0]["messages"][1]["content"]
    assert "THEO TỪNG TIÊU CHÍ" in completions.calls[0]["messages"][0]["content"]


def test_rubric_mode_fails_when_a_code_is_missing() -> None:
    service, _ = _rubric_service(
        [{"criterionScores": [{"code": "C1", "score": 80}], "strengths": [], "improvements": []}]
    )
    result = service.evaluate(_rubric_request())

    assert result.success is False
    assert result.criterion_scores is None


def test_rubric_mode_fails_when_score_out_of_range() -> None:
    service, _ = _rubric_service(
        [{"criterionScores": [{"code": "C1", "score": 180}, {"code": "C2", "score": 50}]}]
    )
    assert service.evaluate(_rubric_request()).success is False


def test_evaluate_caps_temperature_for_stable_scoring() -> None:
    service, completions = _rubric_service(
        [{"criterionScores": [{"code": "C1", "score": 70}, {"code": "C2", "score": 70}]}],
        temperature=0.9,
    )
    service.evaluate(_rubric_request())
    assert completions.calls[0]["temperature"] <= 0.1


def test_no_rubric_keeps_holistic_prompt() -> None:
    result = _service().evaluate(
        EvaluateAnswerRequest(
            question="Explain React virtual DOM.",
            candidate_answer="Không biết.",
        )
    )
    assert result.success is True
    assert result.criterion_scores is None
