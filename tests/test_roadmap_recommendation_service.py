"""SCRUM-455: LLM chỉ được chọn topic trong candidateNodes, không đổi score."""
from __future__ import annotations

import json

from models.internal_schemas import (
    RoadmapCandidateNodeDto,
    RoadmapRecommendRequest,
    RoadmapWeakSkillDto,
)
from services.roadmap_recommendation_service import RoadmapRecommendationService


class _FakeCompletions:
    def create(self, **kwargs):  # noqa: ANN003
        content = json.dumps(
            {
                "topics": [
                    {"topic": "DI registration patterns", "reason": "Nền tảng"},
                    {"topic": "Invented topic", "reason": "không được giữ"},
                    {"topic": "Auth JWT endpoints", "reason": "Tiếp theo"},
                ],
                "explanation": "Luyện DI rồi JWT. Overall phải giữ nguyên.",
            },
            ensure_ascii=False,
        )
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


class _FakeRetrieval:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve_system_only(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls.append({"args": args, "kwargs": kwargs})
        return []


def _service(retrieval: _FakeRetrieval | None = None) -> RoadmapRecommendationService:
    return RoadmapRecommendationService(
        retrieval=retrieval or _FakeRetrieval(),  # type: ignore[arg-type]
        client=_FakeClient(),  # type: ignore[arg-type]
        settings=type("S", (), {"chat_model": "test"})(),
    )


def test_drops_topics_not_in_candidate_nodes() -> None:
    result = _service().recommend(
        RoadmapRecommendRequest(
            target_role="Backend",
            target_level="Junior",
            weak_skills=[
                RoadmapWeakSkillDto(skill="ASP.NET Core", current_score=50, target_score=70, gap=20)
            ],
            candidate_nodes=[
                RoadmapCandidateNodeDto(topic="DI registration patterns", skill="ASP.NET Core", importance=0.9),
                RoadmapCandidateNodeDto(topic="Auth JWT endpoints", skill="ASP.NET Core", importance=0.7),
            ],
        )
    )
    assert result.success is True
    topics = [t.topic for t in result.topics]
    assert topics == ["DI registration patterns", "Auth JWT endpoints"]
    assert "Invented topic" not in topics


def test_empty_nodes_fails() -> None:
    result = _service().recommend(RoadmapRecommendRequest(target_role="Backend"))
    assert result.success is False


def test_retrieve_skipped_without_document_ids() -> None:
    retrieval = _FakeRetrieval()
    _service(retrieval).recommend(
        RoadmapRecommendRequest(
            target_role="Backend",
            target_level="Junior",
            weak_skills=[RoadmapWeakSkillDto(skill="ASP.NET Core", current_score=50, target_score=70, gap=20)],
            candidate_nodes=[
                RoadmapCandidateNodeDto(topic="DI registration patterns", skill="ASP.NET Core", importance=0.9),
            ],
            document_ids=[],
        )
    )
    assert retrieval.calls == []


def test_retrieve_uses_document_ids_when_present() -> None:
    retrieval = _FakeRetrieval()
    _service(retrieval).recommend(
        RoadmapRecommendRequest(
            target_role="Backend",
            target_level="Junior",
            weak_skills=[RoadmapWeakSkillDto(skill="ASP.NET Core", current_score=50, target_score=70, gap=20)],
            candidate_nodes=[
                RoadmapCandidateNodeDto(topic="DI registration patterns", skill="ASP.NET Core", importance=0.9),
            ],
            document_ids=["doc-1", "doc-2"],
        )
    )
    assert len(retrieval.calls) == 1
    assert retrieval.calls[0]["kwargs"].get("document_ids") == ["doc-1", "doc-2"]
    assert "documentType" not in (retrieval.calls[0]["kwargs"].get("metadata_filters") or {})
