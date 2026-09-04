"""SCRUM-394 / SCRUM-425: JD primary citation excerpt + per-question chunk_index."""
from __future__ import annotations

from models.internal_schemas import QuestionCitationItem
from services.rag_context_helpers import (
    JD_SOURCE_FILE,
    ensure_jd_primary_citations,
    find_verbatim_jd_excerpt,
    make_jd_citation,
    resolve_jd_excerpt,
    resolve_jd_excerpt_with_index,
    select_relevant_jd_excerpt,
    select_relevant_jd_unit,
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


def test_postgres_vs_redis_different_chunk_index() -> None:
    """SCRUM-425: skill khác → chunk_index JD khác nhau."""
    pg_ex, pg_idx = select_relevant_jd_unit(JD, "PostgreSQL EF Core database")
    redis_ex, redis_idx = select_relevant_jd_unit(JD, "Redis caching message queues")
    assert "PostgreSQL" in pg_ex or "EF Core" in pg_ex
    assert "Redis" in redis_ex
    assert pg_idx != redis_idx


def test_llm_title_fallback_to_hint_redis() -> None:
    """LLM copy đoạn đầu JD nhưng hint Redis → chọn unit Redis."""
    excerpt, idx = resolve_jd_excerpt_with_index(
        JD,
        llm_excerpt="Senior Backend Developer",
        hint="Redis caching message queues",
    )
    assert "Redis" in excerpt
    assert idx != 0 or "Redis" in excerpt


def test_used_indexes_diversifies_units() -> None:
    """used_indexes tránh 2 câu lấy cùng unit khi còn unit khớp khác."""
    used: set[int] = set()
    c1 = ensure_jd_primary_citations(
        [],
        JD,
        hint="PostgreSQL EF Core",
        used_indexes=used,
    )
    c2 = ensure_jd_primary_citations(
        [],
        JD,
        hint="Redis caching",
        used_indexes=used,
    )
    assert c1[0].chunk_index != c2[0].chunk_index
    assert c1[0].chunk_index in used and c2[0].chunk_index in used
    assert "PostgreSQL" in (c1[0].excerpt or "") or "EF" in (c1[0].excerpt or "")
    assert "Redis" in (c2[0].excerpt or "")
