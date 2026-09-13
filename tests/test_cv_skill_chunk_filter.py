"""SCRUM-461: lọc chunk lệch stack theo skill CV."""
from __future__ import annotations

from types import SimpleNamespace

from services.cv_skill_chunk_filter import (
    chunk_overlaps_skills,
    filter_competency_chunks,
    filter_retrieved_chunks,
    map_skill_to_allowed,
)


def test_filter_drops_dotnet_when_cv_is_frontend():
    skills = ["React", "TypeScript"]
    chunks = [
        SimpleNamespace(
            content="ASP.NET Core middleware and DI registration.",
            metadata={"sourceTitle": "aspnet-roadmap", "section": "DI"},
        ),
        SimpleNamespace(
            content="LINQ projections and Identity JWT auth.",
            metadata={"sourceTitle": "dotnet", "section": "jwt"},
        ),
        SimpleNamespace(
            content="React hooks useState useEffect patterns.",
            metadata={"sourceTitle": "react-guide", "section": "Hooks"},
        ),
    ]
    kept = filter_retrieved_chunks(chunks, skills)
    assert len(kept) == 1
    assert "React" in kept[0].content


def test_filter_competency_dto_style():
    skills = ["React"]
    chunks = [
        SimpleNamespace(content="JWT bearer tokens in ASP.NET", source_title="net", section="jwt"),
        SimpleNamespace(content="React Router nested routes", source_title="fe", section="routing"),
    ]
    kept = filter_competency_chunks(chunks, skills)
    assert len(kept) == 1
    assert "React Router" in kept[0].content


def test_map_skill_rejects_aspnet_for_react_cv():
    assert map_skill_to_allowed("ASP.NET Core", ["React", "TypeScript"]) is None
    assert map_skill_to_allowed("React hooks", ["React", "TypeScript"]) == "React"


def test_short_skill_go_does_not_match_going():
    assert not chunk_overlaps_skills(
        content="We are going to the market tomorrow.",
        skills=["Go"],
    )
    assert chunk_overlaps_skills(
        content="Go concurrency with goroutines.",
        skills=["Go"],
    )
