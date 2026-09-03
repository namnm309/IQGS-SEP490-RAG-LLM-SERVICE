"""SCRUM-421: Waterfall provenance HR → SYSTEM → LLM trên question JSON."""
from __future__ import annotations

import re
from typing import Any

from models.internal_schemas import (
    GeneratedQuestionItem,
    ProvenanceBlock,
    QuestionCitationItem,
)
from services.plan_provenance_validator import (
    _build_chunk_indexes,
    _chunk_scope_origin,
    _chunk_source_file,
    _chunk_text,
    _collapse_ws,
    _excerpt_in_text,
    _make_block,
    _make_provenance_item,
    _resolve_origin_for_excerpt,
)
from services.rag_context_helpers import JD_SOURCE_FILE, is_jd_source_file
from vectorstores.base import RetrievedChunk

_TECHNICAL_QUESTION_TYPES = frozenset(
    {"technical", "problem-solving", "system-design"}
)
_TECHNICAL_KEYWORDS = frozenset(
    {
        "git",
        "docker",
        "kubernetes",
        "sql",
        "api",
        "rest",
        "csharp",
        "python",
        "javascript",
        "react",
        "angular",
        "database",
        "algorithm",
        "microservice",
        "redis",
        "kafka",
        "azure",
        "aws",
    }
)


def _is_technical_question(question: GeneratedQuestionItem) -> bool:
    qtype = (question.question_type or "").strip().lower()
    if qtype in _TECHNICAL_QUESTION_TYPES:
        return True
    if (question.code_template_type or "").strip():
        return True
    if (question.code_snippet or "").strip():
        return True
    return False


def _system_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [c for c in chunks if (c.scope or "").upper() == "SYSTEM"]


def _has_system_citation(citations: list[QuestionCitationItem]) -> bool:
    for cit in citations:
        kb = (cit.knowledge_base or "").strip().lower()
        if kb == "system":
            return True
        if cit.origin == "SYSTEM":
            return True
        if not is_jd_source_file(cit.source_file) and kb != "hr":
            return True
    return False


def _score_chunk(
    chunk: RetrievedChunk,
    *,
    skill: str,
    focus: str,
    question_text: str,
) -> float:
    text = _chunk_text(chunk).lower()
    score = float(chunk.score or 0.0)
    for token in re.split(r"[\s,|/]+", f"{skill} {focus}".lower()):
        t = token.strip()
        if len(t) >= 3 and t in text:
            score += 2.0
    for kw in _TECHNICAL_KEYWORDS:
        if kw in text and kw in question_text.lower():
            score += 0.5
    return score


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, int]:
    return (_chunk_source_file(chunk), int(chunk.chunk_index or 0))


def _pick_best_system_chunk(
    chunks: list[RetrievedChunk],
    *,
    skill: str,
    focus: str,
    question_text: str,
    used_keys: set[tuple[str, int]] | None = None,
    min_score: float = 2.0,
) -> RetrievedChunk | None:
    """Chọn chunk System khớp skill/focus; ưu tiên chưa dùng; dưới ngưỡng → None (LLM)."""
    system = _system_chunks(chunks)
    if not system:
        return None
    used = used_keys or set()
    scored: list[tuple[float, RetrievedChunk]] = []
    for ch in system:
        score = _score_chunk(
            ch, skill=skill, focus=focus, question_text=question_text
        )
        key = _chunk_key(ch)
        # Phạt nhẹ chunk đã dùng để đa dạng hóa
        if key in used:
            score -= 3.0
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return None
    best_score, best = scored[0]
    if best_score < min_score:
        return None
    return best


def _short_excerpt_from_chunk(chunk: RetrievedChunk, *, max_len: int = 220) -> str:
    text = _collapse_ws(_chunk_text(chunk))
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _infer_used_for(
    origin: str,
    *,
    is_jd: bool,
    is_first_jd: bool,
) -> list[str]:
    if origin == "LLM":
        return ["sample-answer"]
    if is_jd or origin == "HR":
        return ["why-asked"] if is_first_jd else ["why-asked"]
    if origin == "SYSTEM":
        return ["technical-body"]
    return ["technical-body"]


def _citation_to_provenance_item(
    cit: QuestionCitationItem,
    *,
    job_description: str,
    chunk_by_key: dict[tuple[str, int], RetrievedChunk],
    chunk_by_file: dict[str, list[RetrievedChunk]],
    is_first_jd: bool,
) -> tuple[QuestionCitationItem, dict[str, Any]]:
    """Chuẩn hóa citation + provenance item tương ứng."""
    sf = cit.source_file
    idx = cit.chunk_index
    excerpt = cit.excerpt or ""
    is_jd = is_jd_source_file(sf)

    if cit.origin in ("HR", "SYSTEM", "LLM"):
        origin = cit.origin
        reason = cit.reason
    else:
        origin, reason = _resolve_origin_for_excerpt(
            excerpt,
            sf,
            idx,
            chunk_by_key=chunk_by_key,
            chunk_by_file=chunk_by_file,
            job_description=job_description,
        )

    used_for = list(cit.used_for) if cit.used_for else []
    if not used_for:
        used_for = _infer_used_for(origin, is_jd=is_jd, is_first_jd=is_first_jd)

    if origin in ("HR", "SYSTEM") and not excerpt.strip():
        origin = "LLM"
        reason = "Citation thiếu excerpt — không chứng minh grounding chunk"
        used_for = ["sample-answer"]

    kb = cit.knowledge_base
    if origin == "HR" or is_jd:
        kb = "hr"
    elif origin == "SYSTEM":
        kb = "system"

    updated = cit.model_copy(
        update={
            "knowledge_base": kb,
            "origin": origin,
            "used_for": used_for,
            "reason": reason if origin == "LLM" else cit.reason,
        }
    )

    prov_item = _make_provenance_item(
        origin,
        source_file=sf,
        chunk_index=idx,
        excerpt=excerpt or None,
        used_for=used_for,
        reason=reason if origin == "LLM" else None,
    )
    return updated, prov_item


def _attach_system_citation_if_needed(
    question: GeneratedQuestionItem,
    citations: list[QuestionCitationItem],
    *,
    chunks: list[RetrievedChunk],
    used_keys: set[tuple[str, int]] | None = None,
) -> tuple[list[QuestionCitationItem], bool]:
    """
    Returns (citations, attached).
    attached=False và thiếu System → caller có thể gắn LLM technical-body.
    """
    if not _is_technical_question(question):
        return citations, False
    if _has_system_citation(citations):
        # Đánh dấu chunk đã dùng nếu có SYSTEM citation
        if used_keys is not None:
            for cit in citations:
                if cit.origin == "SYSTEM" or (cit.knowledge_base or "").lower() == "system":
                    used_keys.add((cit.source_file, int(cit.chunk_index or 0)))
        return citations, True

    skill = (question.skill or "").strip()
    focus = (question.focus_area or "").strip()
    best = _pick_best_system_chunk(
        chunks,
        skill=skill,
        focus=focus,
        question_text=question.question,
        used_keys=used_keys,
    )
    if best is None:
        return citations, False

    excerpt = _short_excerpt_from_chunk(best)
    sf = _chunk_source_file(best)
    idx = int(best.chunk_index or 0)
    if used_keys is not None:
        used_keys.add((sf, idx))
    new_cit = QuestionCitationItem(
        knowledge_base="system",
        source_file=sf,
        chunk_index=idx,
        excerpt=excerpt,
        origin="SYSTEM",
        used_for=["technical-body"],
    )
    # JD đầu, System sau
    jd = [c for c in citations if is_jd_source_file(c.source_file)]
    rest = [c for c in citations if not is_jd_source_file(c.source_file)]
    return jd + [new_cit] + rest, True


def _llm_items_for_ungrounded_parts(
    question: GeneratedQuestionItem,
    *,
    prov_items: list[dict[str, Any]],
    need_technical_llm: bool = False,
) -> list[dict[str, Any]]:
    """Gắn LLM provenance cho sample_answer / rubric / technical thiếu chunk."""
    extra: list[dict[str, Any]] = []
    if need_technical_llm:
        extra.append(
            _make_provenance_item(
                "LLM",
                used_for=["technical-body"],
                reason="Phần kỹ thuật — suy luận LLM khi không có chunk System khớp skill/focus",
            )
        )
    if (question.sample_answer or "").strip():
        extra.append(
            _make_provenance_item(
                "LLM",
                used_for=["sample-answer"],
                reason="Đáp án mẫu — bổ sung từ LLM khi không map chunk Admin/JD",
            )
        )
    if question.evaluation_criteria:
        extra.append(
            _make_provenance_item(
                "LLM",
                used_for=["rubric"],
                reason="Tiêu chí chấm — suy luận LLM từ skill/difficulty/sample",
            )
        )
    # Tránh trùng usedFor nếu đã có LLM item cùng loại
    existing_used = {
        u
        for item in prov_items
        for u in (item.get("usedFor") or [])
    }
    return [e for e in extra if not set(e.get("usedFor") or []).issubset(existing_used)]


def apply_provenance_to_questions(
    questions: list[GeneratedQuestionItem],
    *,
    chunks: list[RetrievedChunk],
    job_description: str,
) -> list[GeneratedQuestionItem]:
    """Gắn origin/usedFor/reason trên citations + sourceProvenance + missingAdminWarning."""
    if not questions:
        return questions

    chunk_by_key, chunk_by_file = _build_chunk_indexes(chunks, job_description)
    system_in_retrieve = _system_chunks(chunks)
    used_system_keys: set[tuple[str, int]] = set()

    for question in questions:
        raw_citations = list(question.citations or [])
        raw_citations, attached_system = _attach_system_citation_if_needed(
            question, raw_citations, chunks=chunks, used_keys=used_system_keys
        )

        new_citations: list[QuestionCitationItem] = []
        prov_items: list[dict[str, Any]] = []
        saw_jd = False

        for cit in raw_citations:
            is_first_jd = not saw_jd and is_jd_source_file(cit.source_file)
            updated, prov = _citation_to_provenance_item(
                cit,
                job_description=job_description,
                chunk_by_key=chunk_by_key,
                chunk_by_file=chunk_by_file,
                is_first_jd=is_first_jd,
            )
            if is_jd_source_file(updated.source_file):
                saw_jd = True
            new_citations.append(updated)
            prov_items.append(prov)

        need_technical_llm = (
            _is_technical_question(question)
            and not attached_system
            and not _has_system_citation(new_citations)
        )
        prov_items.extend(
            _llm_items_for_ungrounded_parts(
                question,
                prov_items=prov_items,
                need_technical_llm=need_technical_llm,
            )
        )

        block_dict = _make_block(prov_items)
        question.citations = new_citations
        question.source_provenance = ProvenanceBlock.model_validate(block_dict)
        question.missing_admin_warning = (
            _is_technical_question(question)
            and not _has_system_citation(new_citations)
            and len(system_in_retrieve) == 0
        )

    return questions
