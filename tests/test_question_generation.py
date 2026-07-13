"""Tests cho LLM JSON retry và rawAnswer gating."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from models.internal_schemas import GenerateQuestionsRequest
from services.question_generation_service import QuestionGenerationService
from vectorstores.base import RetrievedChunk


VALID_JSON = json.dumps(
    {
        "questions": [
            {
                "question": "Giải thích dependency injection?",
                "question_type": "technical",
                "difficulty": "medium",
                "rationale": "Kiểm tra ASP.NET Core",
                "sample_answer": "DI giúp tách coupling.",
                "citations": [],
            }
        ]
    }
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        DATABASE_URL="postgresql://postgres:postgres@localhost:5432/iqgs",
        BACKEND_INTERNAL_BASE_URL="http://localhost:5000",
        INTERNAL_API_KEY="test-secret",
        EMBEDDING_DIMENSION=768,
        DEBUG=False,
    )


@pytest.fixture
def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc-1",
        chunk_index=0,
        content="ASP.NET Core hỗ trợ dependency injection built-in.",
        scope="SYSTEM",
        owner_id=None,
        score=0.9,
        metadata={"fileName": "rubric.pdf"},
    )


def _make_service(
    settings: Settings,
    llm_responses: list[str],
    sample_chunk: RetrievedChunk,
) -> QuestionGenerationService:
    client = MagicMock()
    choices = []
    for content in llm_responses:
        choice = MagicMock()
        choice.message.content = content
        choices.append(choice)
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[choice]) for choice in choices
    ]

    retrieval = MagicMock()
    retrieval.retrieve_for_job.return_value = ([sample_chunk], [])

    return QuestionGenerationService(retrieval=retrieval, client=client, settings=settings)


def test_llm_retry_on_invalid_json(settings: Settings, sample_chunk: RetrievedChunk) -> None:
    service = _make_service(
        settings,
        llm_responses=["```not json```", VALID_JSON],
        sample_chunk=sample_chunk,
    )
    request = GenerateQuestionsRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        numberOfQuestions=1,
        questionTypes=["technical"],
    )

    result = service.generate(request)

    assert result.success is True
    assert len(result.questions) == 1
    assert service._client.chat.completions.create.call_count == 2


def test_raw_answer_hidden_when_debug_false(
    settings: Settings, sample_chunk: RetrievedChunk
) -> None:
    service = _make_service(settings, [VALID_JSON], sample_chunk)
    request = GenerateQuestionsRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        numberOfQuestions=1,
    )

    result = service.generate(request)

    assert result.success is True
    assert result.raw_answer is None
    dumped = result.model_dump(by_alias=True, exclude_none=True)
    assert "rawAnswer" not in dumped


def test_raw_answer_included_when_debug_true(sample_chunk: RetrievedChunk) -> None:
    settings = Settings(
        DATABASE_URL="postgresql://postgres:postgres@localhost:5432/iqgs",
        BACKEND_INTERNAL_BASE_URL="http://localhost:5000",
        INTERNAL_API_KEY="test-secret",
        EMBEDDING_DIMENSION=768,
        DEBUG=True,
    )
    service = _make_service(settings, [VALID_JSON], sample_chunk)
    request = GenerateQuestionsRequest(
        ownerId="11111111-1111-1111-1111-111111111111",
        jobDescription="Backend .NET role",
        numberOfQuestions=1,
    )

    result = service.generate(request)

    assert result.raw_answer == VALID_JSON
