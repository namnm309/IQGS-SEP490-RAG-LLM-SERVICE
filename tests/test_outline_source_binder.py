"""SCRUM-426: Plan-locked sources trên outline slot."""
from __future__ import annotations

from models.internal_schemas import RecommendedQuestionOutlineItem
from services.outline_source_binder import bind_sources_to_outline, outline_needs_rebind
from services.rag_context_helpers import JD_SOURCE_FILE, is_jd_source_file
from vectorstores.base import RetrievedChunk

JD = """\
Software Developer (Fresher/Junior)

Requirements:
- Solid grasp of OOP, design patterns and SOLID principles.
- Strong English communication skills (written and spoken).
- Primary tech stack .NET and Java.

Responsibilities:
- Work in Agile teams developing .NET services.
"""


def test_bind_oop_vs_english_different_jd_chunks() -> None:
    outline = [
        RecommendedQuestionOutlineItem(
            order=1,
            type="technical",
            difficulty="easy",
            skill="SOLID",
            focusArea="SRP",
            goal="Assess OOP and SOLID principles",
            answerMethod="Code",
        ),
        RecommendedQuestionOutlineItem(
            order=2,
            type="behavioral",
            difficulty="easy",
            skill="English",
            focusArea="Communication",
            goal="Assess English communication",
            answerMethod="Text",
        ),
    ]
    locked = bind_sources_to_outline(
        outline, job_description=JD, chunks=[], force_rebind=True
    )
    assert not outline_needs_rebind(locked)
    jd0 = next(c for c in locked[0].citations if is_jd_source_file(c.source_file))
    jd1 = next(c for c in locked[1].citations if is_jd_source_file(c.source_file))
    assert jd0.chunk_index != jd1.chunk_index
    assert "OOP" in (jd0.excerpt or "") or "SOLID" in (jd0.excerpt or "") or "design" in (
        jd0.excerpt or ""
    ).lower()
    assert "English" in (jd1.excerpt or "") or "communication" in (jd1.excerpt or "").lower()
    assert jd0.origin == "HR"
    assert jd0.used_for == ["why-asked"]


def test_bind_keeps_existing_when_not_force() -> None:
    from models.internal_schemas import PlanCitationItem

    existing = PlanCitationItem(
        knowledgeBase="hr",
        sourceFile=JD_SOURCE_FILE,
        chunkIndex=0,
        excerpt="Nắm vững OOP, design patterns và SOLID.",
        origin="HR",
        usedFor=["why-asked"],
    )
    outline = [
        RecommendedQuestionOutlineItem(
            order=1,
            type="technical",
            difficulty="medium",
            skill="SOLID",
            focusArea="SRP",
            goal="x",
            citations=[existing],
        )
    ]
    locked = bind_sources_to_outline(
        outline, job_description=JD, chunks=[], force_rebind=False
    )
    assert locked[0].citations[0].excerpt == existing.excerpt
    assert locked[0].citations[0].chunk_index == 0


def test_bind_attaches_system_for_technical() -> None:
    chunk = RetrievedChunk(
        document_id="sys-1",
        chunk_index=2,
        content="SOLID principles: Single Responsibility, Open-Closed, Liskov, Interface Segregation, Dependency Inversion.",
        scope="SYSTEM",
        owner_id=None,
        score=0.95,
        metadata={"fileName": "solid-guide.pdf"},
    )
    outline = [
        RecommendedQuestionOutlineItem(
            order=1,
            type="technical",
            difficulty="medium",
            skill="SOLID",
            focusArea="SRP",
            goal="Refactor OrderService",
            answerMethod="Code",
        )
    ]
    locked = bind_sources_to_outline(
        outline, job_description=JD, chunks=[chunk], force_rebind=True
    )
    origins = {c.origin for c in locked[0].citations}
    assert "HR" in origins
    assert "SYSTEM" in origins
    sys_cit = next(c for c in locked[0].citations if c.origin == "SYSTEM")
    assert sys_cit.source_file == "solid-guide.pdf"
    assert "SOLID" in (sys_cit.excerpt or "")
