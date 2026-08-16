"""Unit tests for evaluate-question-set (qualitative JD fit, no numeric scores)."""
from __future__ import annotations

import json

from models.internal_schemas import EvaluateQuestionSetItem, EvaluateQuestionSetRequest
from services.evaluate_question_set_service import EvaluateQuestionSetService


class _FakeCompletions:
    def create(self, **kwargs):  # noqa: ANN003
        payload = json.loads(kwargs["messages"][1]["content"])
        jd = (payload.get("jobDescription") or "").lower()
        if "kubernetes" in jd:
            content = json.dumps(
                {
                    "verdict": "fair",
                    "summaryVi": "Bộ tương đối bám JD backend nhưng còn thiếu Kubernetes.",
                    "summaryEn": "The set fairly matches the backend JD but misses Kubernetes.",
                    "jdSources": [{"chunkIndex": 0}],
                    "questionFlags": [
                        {
                            "questionId": "q1",
                            "order": 1,
                            "flag": "onJd",
                            "noteVi": "Đúng ASP.NET trong JD.",
                            "noteEn": "Matches ASP.NET in the JD.",
                            "sources": [{"chunkIndex": 0}],
                        },
                        {
                            "questionId": "q2",
                            "order": 2,
                            "flag": "offJd",
                            "noteVi": "React không có trong JD backend.",
                            "noteEn": "React is not in this backend JD.",
                        },
                    ],
                    "missingTopics": ["Kubernetes"],
                    "suggestedActions": [
                        {
                            "type": "add",
                            "questionId": None,
                            "reasonVi": "Thêm câu về Kubernetes.",
                            "reasonEn": "Add a Kubernetes question.",
                        },
                        {
                            "type": "remove",
                            "questionId": "q2",
                            "reasonVi": "Bỏ câu React vì lạc đề.",
                            "reasonEn": "Remove the React question as off-JD.",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        else:
            content = json.dumps(
                {
                    "verdict": "excellent",
                    "summaryVi": "Bộ tuyệt vời so với JD, đúng role và kỹ năng.",
                    "summaryEn": "Excellent fit with the JD, covering the role and skills.",
                    "questionFlags": [
                        {
                            "questionId": "q1",
                            "order": 1,
                            "flag": "onJd",
                            "noteVi": "Phù hợp.",
                            "noteEn": "On JD.",
                        }
                    ],
                    "missingTopics": [],
                    "suggestedActions": [],
                },
                ensure_ascii=False,
            )

        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def _service() -> EvaluateQuestionSetService:
    return EvaluateQuestionSetService(
        client=_FakeClient(),
        settings=type("S", (), {"chat_model": "test", "temperature": 0.2})(),
    )


def _item(qid: str, order: int, question: str, skill: str) -> EvaluateQuestionSetItem:
    return EvaluateQuestionSetItem(
        question_id=qid,
        order=order,
        question=question,
        question_type="technical",
        difficulty="medium",
        skill=skill,
    )


def test_evaluate_fair_when_jd_has_uncovered_skill() -> None:
    result = _service().evaluate(
        EvaluateQuestionSetRequest(
            job_description="Senior Backend with ASP.NET Core and Kubernetes.",
            set_title="Backend set",
            questions=[
                _item("q1", 1, "Explain ASP.NET middleware.", "ASP.NET Core"),
                _item("q2", 2, "Explain React hooks.", "React"),
            ],
        )
    )
    assert result.success is True
    assert result.verdict == "fair"
    assert result.summary_vi and "Kubernetes" in result.summary_vi
    dumped = result.model_dump(by_alias=True)
    assert "score" not in dumped
    assert dumped["summaryVi"]
    flags = {f.question_id: f.flag for f in result.question_flags}
    assert flags["q1"] == "onJd"
    assert flags["q2"] == "offJd"
    assert "Kubernetes" in result.missing_topics
    types = {a.type for a in result.suggested_actions}
    assert "add" in types
    assert "remove" in types
    assert result.jd_sources
    assert all(f.sources for f in result.question_flags)


def test_evaluate_excellent_when_aligned() -> None:
    result = _service().evaluate(
        EvaluateQuestionSetRequest(
            job_description="We need a C# backend engineer.",
            questions=[_item("q1", 1, "Explain DI in ASP.NET Core.", "C#")],
        )
    )
    assert result.success is True
    assert result.verdict == "excellent"
    dumped = result.model_dump()
    assert dumped.get("score") is None
    assert result.jd_sources
    assert result.question_flags[0].sources
    assert "C#" in result.question_flags[0].sources[0].excerpt or "backend" in result.question_flags[0].sources[0].excerpt.lower() or result.question_flags[0].sources[0].excerpt


def test_resolve_sources_keeps_valid_index_and_drops_invalid() -> None:
    units = ["Need ASP.NET Core.", "Must know Kubernetes."]
    kept = EvaluateQuestionSetService._resolve_sources(
        [{"chunkIndex": 1}, {"chunkIndex": 99}, {"chunkIndex": 1}, {"chunkIndex": 0}],
        units,
        max_n=8,
    )
    assert [s.chunk_index for s in kept] == [1, 0]
    assert kept[0].excerpt == "Must know Kubernetes."


def test_resolve_sources_empty_when_no_valid_index() -> None:
    units = ["Only one unit."]
    kept = EvaluateQuestionSetService._resolve_sources([{"chunkIndex": 7}], units, max_n=3)
    assert kept == []


def test_evaluate_rejects_empty_jd() -> None:
    result = _service().evaluate(
        EvaluateQuestionSetRequest(
            job_description="   ",
            questions=[_item("q1", 1, "Q?", "C#")],
        )
    )
    assert result.success is False
    assert "trống" in (result.error or "").lower() or "empty" in (result.error or "").lower() or "không" in (result.error or "")


def test_evaluate_rejects_empty_questions() -> None:
    result = _service().evaluate(
        EvaluateQuestionSetRequest(job_description="Senior Backend JD with enough text.", questions=[])
    )
    assert result.success is False


def test_strip_numeric_from_summary() -> None:
    svc = _service()
    cleaned = svc._strip_numeric("Bộ tốt 85/100 và cover 72%.")
    assert cleaned is not None
    assert "85" not in cleaned
    assert "72" not in cleaned
    assert "%" not in cleaned
