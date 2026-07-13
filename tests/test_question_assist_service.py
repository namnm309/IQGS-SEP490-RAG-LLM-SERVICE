"""Unit tests for question assist service (SCRUM-251)."""
from __future__ import annotations

import json

from models.internal_schemas import QuestionAssistContext, QuestionAssistRequest
from services.question_assist_service import QuestionAssistService


class _FakeCompletions:
    def create(self, **kwargs):  # noqa: ANN003
        payload = json.loads(kwargs["messages"][1]["content"])
        hr_message = payload.get("hrMessage", "").lower()

        if "weather" in hr_message:
            content = json.dumps(
                {
                    "reply": "Câu hỏi này nằm ngoài phạm vi JD và câu hỏi hiện tại.",
                    "suggestion": None,
                },
                ensure_ascii=False,
            )
        else:
            content = json.dumps(
                {
                    "reply": "Đã làm khó hơn câu hỏi.",
                    "suggestion": {
                        "question": "Harder version of question",
                        "difficulty": "hard",
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


def test_assist_returns_suggestion_for_edit_prompt() -> None:
    service = QuestionAssistService(
        client=_FakeClient(),
        settings=type("S", (), {"chat_model": "test"})(),
    )
    request = QuestionAssistRequest(
        owner_id="owner-1",
        hr_message="Làm khó hơn",
        context=QuestionAssistContext(
            job_description="JD backend developer",
            current_question={"question": "Explain REST"},
        ),
    )

    result = service.assist(request)

    assert result.success is True
    assert "khó hơn" in result.reply.lower()
    assert result.suggestion is not None
    assert result.suggestion.question == "Harder version of question"


def test_assist_handles_out_of_scope_prompt() -> None:
    service = QuestionAssistService(
        client=_FakeClient(),
        settings=type("S", (), {"chat_model": "test"})(),
    )
    request = QuestionAssistRequest(
        owner_id="owner-1",
        hr_message="What's the weather today?",
        context=QuestionAssistContext(
            job_description="JD backend developer",
            current_question={"question": "Explain REST"},
        ),
    )

    result = service.assist(request)

    assert result.success is True
    assert result.suggestion is None
