"""SCRUM-457 / SCRUM-461: Adaptive competency retrieve/blueprint/roadmap validation."""
from __future__ import annotations

from types import SimpleNamespace

from models.internal_schemas import (
    AdaptiveBlueprintRequest,
    AdaptiveRoadmapRequest,
    CompetencyContextChunkDto,
    CompetencyContextRequest,
)
from services.adaptive_competency_service import AdaptiveCompetencyService


class _FakeRetrieval:
    def __init__(self, chunks=None) -> None:
        self.chunks = chunks or []
        self.calls: list[dict] = []

    def retrieve_system_only(self, query, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append({"query": query, **kwargs})
        return self.chunks


class _FakeSettings:
    chat_model = "test"
    temperature = 0.1


def _chunk(**kwargs):
    meta = kwargs.pop("metadata", {})
    return SimpleNamespace(
        document_id=kwargs.get("document_id", "11111111-1111-1111-1111-111111111111"),
        chunk_index=0,
        content=kwargs.get("content", "Go concurrency and goroutines."),
        scope="SYSTEM",
        owner_id=None,
        score=0.9,
        metadata=meta,
    )


def test_retrieve_exposes_source_metadata():
    retrieval = _FakeRetrieval(
        [
            _chunk(
                metadata={
                    "sourceTitle": "Go handbook",
                    "sourceUrl": "https://example.test/go",
                    "section": "Concurrency",
                    "documentType": "InternalStack",
                },
                content="Go concurrency and goroutines.",
            )
        ]
    )
    svc = AdaptiveCompetencyService(retrieval, client=SimpleNamespace(), settings=_FakeSettings())
    result = svc.retrieve_context(
        CompetencyContextRequest(targetRole="Backend", targetLevel="Junior", skills=["Go"])
    )
    assert result.success
    assert result.chunks[0].source_title == "Go handbook"
    assert result.chunks[0].source_url == "https://example.test/go"
    assert result.chunks[0].document_type == "InternalStack"
    assert retrieval.calls[0]["metadata_filters"]["documentType"] == "InternalStack"


def test_retrieve_empty_with_cv_skills_is_inferred_success():
    """SCRUM-461: EMPTY index nhưng còn CV → success chunks=[] để BE inferred."""
    svc = AdaptiveCompetencyService(_FakeRetrieval([]), client=SimpleNamespace(), settings=_FakeSettings())
    result = svc.retrieve_context(
        CompetencyContextRequest(targetRole="Frontend", targetLevel="Junior", skills=["React"])
    )
    assert result.success is True
    assert result.chunks == []


def test_retrieve_empty_without_skills_is_controlled_failure():
    svc = AdaptiveCompetencyService(_FakeRetrieval([]), client=SimpleNamespace(), settings=_FakeSettings())
    result = svc.retrieve_context(CompetencyContextRequest(targetRole="Backend", targetLevel="Junior"))
    assert result.success is False
    assert "EMPTY_RETRIEVAL" in (result.error or "")


def test_retrieve_drops_dotnet_chunks_for_frontend_cv():
    retrieval = _FakeRetrieval(
        [
            _chunk(
                content="ASP.NET Core middleware pipeline and Identity.",
                metadata={"sourceTitle": "aspnet", "section": "DI", "documentType": "InternalStack"},
            ),
            _chunk(
                content="LINQ and JWT bearer authentication.",
                metadata={"sourceTitle": "dotnet", "section": "jwt", "documentType": "InternalStack"},
            ),
        ]
    )
    svc = AdaptiveCompetencyService(retrieval, client=SimpleNamespace(), settings=_FakeSettings())
    result = svc.retrieve_context(
        CompetencyContextRequest(
            targetRole="Frontend Developer",
            targetLevel="Junior",
            skills=["React", "TypeScript"],
        )
    )
    assert result.success is True
    assert result.chunks == []


def test_generate_blueprint_empty_chunks_with_cv_calls_llm(monkeypatch):
    called = {"n": 0}

    class _Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):  # noqa: ANN003
                    called["n"] += 1
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content=(
                                        '{"competencies":['
                                        '{"skillName":"React","category":"ROLE_CORE","weight":0.5,'
                                        '"topics":["hooks","components"]},'
                                        '{"skillName":"TypeScript","category":"FUNDAMENTAL","weight":0.5,'
                                        '"topics":["types","generics"]}'
                                        "]}"
                                    )
                                )
                            )
                        ]
                    )

    svc = AdaptiveCompetencyService(_FakeRetrieval([]), client=_Client(), settings=_FakeSettings())
    result = svc.generate_blueprint(
        AdaptiveBlueprintRequest(
            targetRole="Frontend",
            cvSkills=["React", "TypeScript"],
            chunks=[],
        )
    )
    assert result.success
    assert called["n"] >= 1
    assert {c.skill_name for c in result.competencies} == {"React", "TypeScript"}


def test_generate_blueprint_rejects_skill_outside_cv():
    class _Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):  # noqa: ANN003
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content=(
                                        '{"competencies":['
                                        '{"skillName":"ASP.NET Core","category":"ROLE_CORE","weight":1.0,'
                                        '"topics":["DI"],"chunkIndexes":[0]}'
                                        "]}"
                                    )
                                )
                            )
                        ]
                    )

    chunks = [CompetencyContextChunkDto(content="DI in ASP.NET", sourceTitle="net")]
    svc = AdaptiveCompetencyService(_FakeRetrieval(), client=_Client(), settings=_FakeSettings())
    result = svc.generate_blueprint(
        AdaptiveBlueprintRequest(
            targetRole="Frontend",
            cvSkills=["React", "TypeScript"],
            chunks=chunks,
        )
    )
    assert result.success is False
    assert "cvSkills" in (result.error or "") or "ASP.NET" in (result.error or "")


def test_generate_blueprint_empty_chunks_without_cv_does_not_call_llm():
    called = {"n": 0}

    class _Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):  # noqa: ANN003
                    called["n"] += 1
                    raise AssertionError("LLM must not run without chunks and cvSkills")

    svc = AdaptiveCompetencyService(_FakeRetrieval([]), client=_Client(), settings=_FakeSettings())
    result = svc.generate_blueprint(AdaptiveBlueprintRequest(targetRole="Backend", chunks=[]))
    assert result.success is False
    assert "EMPTY_RETRIEVAL" in (result.error or "")
    assert called["n"] == 0


def test_validate_blueprint_rejects_too_many_for_cv(monkeypatch):
    chunks = [
        CompetencyContextChunkDto(content="React hooks", sourceTitle="a"),
    ]
    svc = AdaptiveCompetencyService(_FakeRetrieval(), client=SimpleNamespace(), settings=_FakeSettings())
    comps, err = svc._validate_blueprint(  # noqa: SLF001
        {
            "competencies": [
                {"skillName": "React", "category": "ROLE_CORE", "weight": 0.5, "topics": ["hooks"]},
                {"skillName": "TypeScript", "category": "FUNDAMENTAL", "weight": 0.5, "topics": ["types"]},
            ]
        },
        chunks,
        cv_skills=["React"],
    )
    assert comps == []
    assert err is not None
    assert "1–1" in err or "cvSkills" in err or "TypeScript" in err


def test_validate_blueprint_accepts_valid_payload():
    chunks = [
        CompetencyContextChunkDto(content="Go goroutines", sourceTitle="a"),
        CompetencyContextChunkDto(content="SQL indexes", sourceTitle="b"),
        CompetencyContextChunkDto(content="HTTP REST", sourceTitle="c"),
    ]
    svc = AdaptiveCompetencyService(_FakeRetrieval(), client=SimpleNamespace(), settings=_FakeSettings())
    comps, err = svc._validate_blueprint(  # noqa: SLF001
        {
            "competencies": [
                {"skillName": "Go", "category": "ROLE_CORE", "weight": 0.4, "topics": ["goroutine"], "chunkIndexes": [0]},
                {"skillName": "SQL", "category": "FUNDAMENTAL", "weight": 0.3, "topics": ["index"], "chunkIndexes": [1]},
                {"skillName": "HTTP", "category": "ADVANCED", "weight": 0.3, "topics": ["rest"], "chunkIndexes": [2]},
            ]
        },
        chunks,
        cv_skills=["Go", "SQL", "HTTP"],
    )
    assert err is None
    assert len(comps) == 3
    assert comps[0].citations[0].source_title == "a"


def test_adaptive_roadmap_topics_must_be_in_allowed_set_when_chunks_present():
    class _Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):  # noqa: ANN003
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content='{"topics":[{"topic":"Invented"}],"explanation":"x"}'
                                )
                            )
                        ]
                    )

    svc = AdaptiveCompetencyService(_FakeRetrieval(), client=_Client(), settings=_FakeSettings())
    result = svc.generate_roadmap(
        AdaptiveRoadmapRequest(
            skill="Go",
            blueprintTopics=["goroutines", "channels"],
            chunks=[CompetencyContextChunkDto(content="Go channels", sourceTitle="go", section="channels")],
        )
    )
    assert result.success
    assert all(t.topic in {"goroutines", "channels"} for t in result.topics)
    assert "Invented" not in [t.topic for t in result.topics]
