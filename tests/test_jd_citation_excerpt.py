"""SCRUM-394: JD primary citation excerpt phải trích từ JD (giống KB phụ)."""
from __future__ import annotations

from models.internal_schemas import QuestionCitationItem
from services.rag_context_helpers import (
    JD_SOURCE_FILE,
    ensure_jd_primary_citations,
    find_verbatim_jd_excerpt,
    make_jd_citation,
    resolve_jd_excerpt,
    select_relevant_jd_excerpt,
)


JD = """\
Senior Backend Developer

Requirements:
- 3+ years experience with ASP.NET Core and Web API
- Strong knowledge of PostgreSQL and EF Core
- Experience with Redis caching and message queues

Responsibilities:
- Design and implement microservices
- Mentor junior developers on clean architecture
"""


def test_find_verbatim_exact_substring() -> None:
    excerpt = "Strong knowledge of PostgreSQL and EF Core"
    found = find_verbatim_jd_excerpt(JD, excerpt)
    assert found is not None
    assert "PostgreSQL" in found
    assert found in JD.replace("\n", " ") or found in JD


def test_find_verbatim_rejects_paraphrase() -> None:
    assert find_verbatim_jd_excerpt(JD, "Must know SQL databases well") is None


def test_select_relevant_prefers_skill_section() -> None:
    excerpt = select_relevant_jd_excerpt(JD, hint="PostgreSQL Redis caching")
    assert "PostgreSQL" in excerpt or "Redis" in excerpt
    # Không phải chỉ lấy tiêu đề đầu file
    assert not excerpt.startswith("Senior Backend Developer")


def test_resolve_uses_llm_when_verbatim() -> None:
    llm = "Experience with Redis caching and message queues"
    resolved = resolve_jd_excerpt(JD, llm_excerpt=llm, hint="something else")
    assert "Redis" in resolved


def test_resolve_falls_back_to_hint_when_llm_paraphrase() -> None:
    resolved = resolve_jd_excerpt(
        JD,
        llm_excerpt="Candidate should know databases",
        hint="Ask about PostgreSQL and EF Core",
    )
    assert "PostgreSQL" in resolved or "EF Core" in resolved
    assert resolved  # non-empty


def test_ensure_jd_primary_always_has_excerpt() -> None:
    citations = ensure_jd_primary_citations(
        [
            QuestionCitationItem(
                knowledge_base="system",
                source_file="rubric.md",
                chunk_index=1,
                excerpt="coding criteria",
            )
        ],
        JD,
        hint="microservices clean architecture",
    )
    assert citations[0].source_file == JD_SOURCE_FILE
    assert citations[0].excerpt
    assert "microservices" in citations[0].excerpt.lower() or "architecture" in citations[
        0
    ].excerpt.lower()
    assert citations[1].source_file == "rubric.md"


def test_make_jd_citation_empty_jd() -> None:
    cit = make_jd_citation("", excerpt="x", hint="y")
    assert cit.source_file == JD_SOURCE_FILE
    assert cit.excerpt == ""
