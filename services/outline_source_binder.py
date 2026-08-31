"""SCRUM-426: Khóa nguồn JD + Admin trên từng outline slot."""
from __future__ import annotations

import re
from typing import Any

from models.internal_schemas import (
    PlanCitationItem,
    RecommendedQuestionOutlineItem,
)
from services.question_provenance_validator import (
    _chunk_key,
    _chunk_source_file,
    _pick_best_system_chunk,
    _short_excerpt_from_chunk,
)
from services.rag_context_helpers import (
    JD_KNOWLEDGE_BASE,
    JD_SOURCE_FILE,
    is_jd_source_file,
    make_jd_citation,
    select_relevant_jd_unit,
)
from vectorstores.base import RetrievedChunk

_TECHNICAL_TYPES = frozenset(
    {
        "technical",
        "problem-solving",
        "problemsolving",
        "system-design",
        "systemdesign",
        "coding",
        "algorithm",
    }
)


def _normalize_type(raw: str | None) -> str:
    return re.sub(r"[\s_-]+", "", (raw or "").strip().lower())


def slot_needs_system(item: RecommendedQuestionOutlineItem) -> bool:
    t = _normalize_type(item.type)
    if t in _TECHNICAL_TYPES:
        return True
    am = (item.answer_method or "").strip().lower()
    return am in ("code", "coding", "programming")


def _slot_hint(item: RecommendedQuestionOutlineItem) -> str:
    return " ".join(
        part
        for part in [
            item.skill or "",
            item.focus_area or "",
            item.goal or "",
            item.type or "",
        ]
        if part
    )


def _has_locked_jd(citations: list[PlanCitationItem] | None) -> bool:
    for c in citations or []:
        if is_jd_source_file(c.source_file) and (c.excerpt or "").strip():
            return True
    return False


def _has_system(citations: list[PlanCitationItem] | None) -> bool:
    for c in citations or []:
        if (c.origin or "").upper() == "SYSTEM":
            return True
        if (c.knowledge_base or "").lower() == "system" and not is_jd_source_file(
            c.source_file
        ):
            return True
    return False


def bind_sources_to_outline(
    outline: list[RecommendedQuestionOutlineItem],
    *,
    job_description: str,
    chunks: list[RetrievedChunk] | None = None,
    force_rebind: bool = False,
) -> list[RecommendedQuestionOutlineItem]:
    """
    Gắn citations JD (+ SYSTEM nếu technical) lên từng slot.
    force_rebind=True: ghi đè citations cũ (khi skill/focus đổi).
    """
    if not outline:
        return outline

    used_jd: set[int] = set()
    used_system: set[tuple[str, int]] = set()
    chunks = chunks or []
    result: list[RecommendedQuestionOutlineItem] = []

    for item in outline:
        existing = list(item.citations or [])
        keep = existing and not force_rebind and _has_locked_jd(existing)

        if keep:
            # Vẫn thu thập used keys để slot sau đa dạng
            for c in existing:
                if is_jd_source_file(c.source_file) and c.chunk_index is not None:
                    used_jd.add(int(c.chunk_index))
                if (c.origin or "").upper() == "SYSTEM" or (
                    (c.knowledge_base or "").lower() == "system"
                    and not is_jd_source_file(c.source_file)
                ):
                    used_system.add((c.source_file, int(c.chunk_index or 0)))
            result.append(item)
            continue

        hint = _slot_hint(item)
        jd_cit = make_jd_citation(
            job_description,
            hint=hint,
            used_indexes=used_jd,
        )
        jd_cit = jd_cit.model_copy(
            update={
                "origin": "HR",
                "used_for": ["why-asked"],
            }
        )
        if jd_cit.chunk_index is not None:
            used_jd.add(int(jd_cit.chunk_index))

        new_cits: list[PlanCitationItem] = [jd_cit]

        if slot_needs_system(item) and chunks:
            best = _pick_best_system_chunk(
                chunks,
                skill=item.skill or "",
                focus=item.focus_area or "",
                question_text=hint,
                used_keys=used_system,
                min_score=1.5,
            )
            if best is not None:
                sf = _chunk_source_file(best)
                idx = int(best.chunk_index or 0)
                used_system.add((sf, idx))
                new_cits.append(
                    PlanCitationItem(
                        knowledge_base="system",
                        source_file=sf,
                        chunk_index=idx,
                        excerpt=_short_excerpt_from_chunk(best),
                        origin="SYSTEM",
                        used_for=["technical-body"],
                    )
                )

        result.append(item.model_copy(update={"citations": new_cits}))

    return result


def outline_needs_rebind(outline: list[RecommendedQuestionOutlineItem]) -> bool:
    """True nếu thiếu JD lock trên bất kỳ slot nào."""
    if not outline:
        return False
    return any(not _has_locked_jd(it.citations) for it in outline)


def citations_to_dicts(citations: list[PlanCitationItem]) -> list[dict[str, Any]]:
    return [c.model_dump(by_alias=True, exclude_none=True) for c in citations]
