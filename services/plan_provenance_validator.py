"""SCRUM-420: Waterfall provenance HR → SYSTEM → LLM trên plan JSON."""
from __future__ import annotations

import re
from typing import Any

from services.rag_context_helpers import (
    JD_SOURCE_FILE,
    _split_jd_units,
    is_jd_source_file,
)
from vectorstores.base import RetrievedChunk

PlanOrigin = str  # "HR" | "SYSTEM" | "LLM"


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _excerpt_in_text(excerpt: str, haystack: str, *, min_len: int = 12) -> bool:
    ex = _collapse_ws(excerpt)
    if len(ex) < min_len:
        return False
    hay = _collapse_ws(haystack).lower()
    return ex.lower() in hay


def _chunk_text(chunk: RetrievedChunk) -> str:
    return chunk.content or ""


def _chunk_source_file(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("fileName", chunk.document_id)).strip()


def _chunk_scope_origin(chunk: RetrievedChunk) -> PlanOrigin:
    scope = (chunk.scope or "").upper()
    if scope == "SYSTEM":
        return "SYSTEM"
    return "HR"


def _build_chunk_indexes(
    chunks: list[RetrievedChunk],
    job_description: str,
) -> tuple[dict[tuple[str, int], RetrievedChunk], dict[str, list[RetrievedChunk]]]:
    by_key: dict[tuple[str, int], RetrievedChunk] = {}
    by_file: dict[str, list[RetrievedChunk]] = {}
    for ch in chunks:
        sf = _chunk_source_file(ch)
        idx = int(ch.chunk_index or 0)
        if sf:
            by_key[(sf.lower(), idx)] = ch
            by_file.setdefault(sf.lower(), []).append(ch)
    if job_description.strip():
        # SCRUM-425: virtual JD chunks theo unit — citation chunk_index khớp đoạn thật
        units = _split_jd_units(job_description)
        if not units:
            units = [_collapse_ws(job_description)]
        for i, unit in enumerate(units):
            jd = RetrievedChunk(
                document_id="jd",
                chunk_index=i,
                content=unit,
                scope="HR",
                owner_id=None,
                score=1.0,
                metadata={"fileName": JD_SOURCE_FILE},
            )
            by_key[(JD_SOURCE_FILE, i)] = jd
            by_file.setdefault(JD_SOURCE_FILE, []).append(jd)
            by_key[(JD_SOURCE_FILE.lower(), i)] = jd
            by_file.setdefault(JD_SOURCE_FILE.lower(), []).append(jd)
    return by_key, by_file


def _resolve_origin_for_excerpt(
    excerpt: str,
    source_file: str | None,
    chunk_index: int | None,
    *,
    chunk_by_key: dict[tuple[str, int], RetrievedChunk],
    chunk_by_file: dict[str, list[RetrievedChunk]],
    job_description: str,
) -> tuple[PlanOrigin, str | None]:
    """Trả (origin, reason) — reason chỉ dùng khi LLM."""
    ex = _collapse_ws(excerpt)
    sf = (source_file or "").strip()
    sf_l = sf.lower()

    if is_jd_source_file(sf) or sf_l == JD_SOURCE_FILE:
        if _excerpt_in_text(ex, job_description):
            return "HR", None

    if sf and chunk_index is not None:
        ch = chunk_by_key.get((sf_l, int(chunk_index)))
        if ch and _excerpt_in_text(ex, _chunk_text(ch)):
            return _chunk_scope_origin(ch), None

    if sf:
        for ch in chunk_by_file.get(sf_l, []):
            if _excerpt_in_text(ex, _chunk_text(ch)):
                return _chunk_scope_origin(ch), None

    for ch in chunk_by_key.values():
        if not _excerpt_in_text(ex, _chunk_text(ch)):
            continue
        if _chunk_scope_origin(ch) == "SYSTEM":
            return "SYSTEM", None
        if is_jd_source_file(_chunk_source_file(ch)):
            return "HR", None
        return "HR", None

    if _excerpt_in_text(ex, job_description):
        return "HR", None

    return "LLM", "Suy luận từ instruction/JD"


def _make_provenance_item(
    origin: PlanOrigin,
    *,
    source_file: str | None = None,
    chunk_index: int | None = None,
    excerpt: str | None = None,
    used_for: list[str] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "origin": origin,
        "usedFor": used_for or [],
    }
    if source_file:
        item["sourceFile"] = source_file
    if chunk_index is not None:
        item["chunkIndex"] = chunk_index
    if excerpt:
        item["excerpt"] = excerpt[:300]
    if origin == "LLM":
        item["reason"] = reason or "Suy luận LLM — không có excerpt khớp corpus"
    elif reason:
        item["reason"] = reason
    return item


def _make_block(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"primaryOrigin": "LLM", "items": []}
    priority = {"HR": 0, "SYSTEM": 1, "LLM": 2}
    primary = min(items, key=lambda x: priority.get(x.get("origin", "LLM"), 9))
    return {"primaryOrigin": primary.get("origin", "LLM"), "items": items}


def _validate_citation_item(
    cit: dict[str, Any],
    *,
    chunk_by_key: dict[tuple[str, int], RetrievedChunk],
    chunk_by_file: dict[str, list[RetrievedChunk]],
    job_description: str,
    used_for: list[str],
) -> dict[str, Any]:
    sf = cit.get("sourceFile") or cit.get("source_file")
    idx_raw = cit.get("chunkIndex", cit.get("chunk_index"))
    idx = int(idx_raw) if idx_raw is not None else None
    excerpt = str(cit.get("excerpt") or "")
    origin, reason = _resolve_origin_for_excerpt(
        excerpt,
        str(sf) if sf else None,
        idx,
        chunk_by_key=chunk_by_key,
        chunk_by_file=chunk_by_file,
        job_description=job_description,
    )
    if origin in ("HR", "SYSTEM") and not excerpt.strip():
        origin = "LLM"
        reason = "Citation thiếu excerpt — không chứng minh grounding chunk"
    prov = _make_provenance_item(
        origin,
        source_file=str(sf) if sf else None,
        chunk_index=idx,
        excerpt=excerpt or None,
        used_for=used_for,
        reason=reason,
    )
    cit = dict(cit)
    cit["provenance"] = _make_block([prov])
    return cit


def apply_provenance_to_plan_dict(
    plan_data: dict[str, Any],
    *,
    chunks: list[RetrievedChunk],
    job_description: str,
) -> dict[str, Any]:
    """Gắn/sửa provenance trên plan dict (camelCase output)."""
    chunk_by_key, chunk_by_file = _build_chunk_indexes(chunks, job_description)
    result = dict(plan_data)

    citations_raw = result.get("citations") or []
    new_citations = []
    for cit in citations_raw:
        if not isinstance(cit, dict):
            continue
        new_citations.append(
            _validate_citation_item(
                cit,
                chunk_by_key=chunk_by_key,
                chunk_by_file=chunk_by_file,
                job_description=job_description,
                used_for=["citation"],
            )
        )
    result["citations"] = new_citations

    coverage_raw = result.get("coverage") or []
    new_coverage = []
    for cov in coverage_raw:
        if not isinstance(cov, dict):
            continue
        skill = str(cov.get("skill") or "")
        items: list[dict[str, Any]] = []
        sf_list = cov.get("sourceFiles") or cov.get("source_files") or []
        if isinstance(sf_list, list):
            for sf in sf_list[:4]:
                sf_s = str(sf).strip()
                if not sf_s:
                    continue
                if is_jd_source_file(sf_s):
                    excerpt = _collapse_ws(job_description)[:180]
                    origin = "HR" if _excerpt_in_text(excerpt, job_description, min_len=8) else "LLM"
                    reason = None if origin == "HR" else "Coverage gắn JD nhưng không trích được excerpt"
                else:
                    matched = chunk_by_file.get(sf_s.lower(), [])
                    excerpt = _collapse_ws(_chunk_text(matched[0]))[:180] if matched else ""
                    if excerpt:
                        origin, reason = _resolve_origin_for_excerpt(
                            excerpt,
                            sf_s,
                            int(matched[0].chunk_index or 0),
                            chunk_by_key=chunk_by_key,
                            chunk_by_file=chunk_by_file,
                            job_description=job_description,
                        )
                    else:
                        origin, reason = "LLM", f"Không tìm chunk cho file {sf_s}"
                items.append(
                    _make_provenance_item(
                        origin,
                        source_file=sf_s,
                        excerpt=excerpt or None,
                        used_for=["coverage", skill],
                        reason=reason,
                    )
                )
        if not items:
            items.append(
                _make_provenance_item(
                    "LLM",
                    used_for=["coverage", skill],
                    reason="Coverage không có source_files — phân bổ suy luận",
                )
            )
        cov = dict(cov)
        cov["provenance"] = _make_block(items)
        new_coverage.append(cov)
    result["coverage"] = new_coverage

    skills_raw = result.get("skills") or []
    new_skills = []
    for sk in skills_raw:
        if isinstance(sk, str):
            new_skills.append(
                {
                    "name": sk,
                    "provenance": _make_block(
                        [
                            _make_provenance_item(
                                "LLM",
                                used_for=["skill", sk],
                                reason="Skill list — suy luận từ JD/instruction nếu không có excerpt riêng",
                            )
                        ]
                    ),
                }
            )
        elif isinstance(sk, dict):
            new_skills.append(sk)
        else:
            new_skills.append(sk)
    if new_skills:
        result["skills"] = new_skills

    outline_raw = result.get("recommendedQuestionOutline") or result.get(
        "recommended_question_outline", []
    )
    new_outline = []
    for item in outline_raw:
        if not isinstance(item, dict):
            continue
        goal = str(item.get("goal") or "")
        skill = str(item.get("skill") or "")
        origin, reason = _resolve_origin_for_excerpt(
            goal,
            None,
            None,
            chunk_by_key=chunk_by_key,
            chunk_by_file=chunk_by_file,
            job_description=job_description,
        )
        if origin == "LLM" and not goal.strip():
            reason = "Outline goal trống — mục tiêu câu hỏi suy luận"
        item = dict(item)
        item["provenance"] = _make_block(
            [
                _make_provenance_item(
                    origin,
                    excerpt=goal[:200] or None,
                    used_for=["outline", skill],
                    reason=reason,
                )
            ]
        )
        new_outline.append(item)
    if new_outline:
        result["recommendedQuestionOutline"] = new_outline

    summary = str(result.get("summary") or "")
    if summary:
        origin, reason = _resolve_origin_for_excerpt(
            summary[:200],
            None,
            None,
            chunk_by_key=chunk_by_key,
            chunk_by_file=chunk_by_file,
            job_description=job_description,
        )
        result["summaryProvenance"] = _make_block(
            [
                _make_provenance_item(
                    origin,
                    excerpt=summary[:200],
                    used_for=["summary"],
                    reason=reason,
                )
            ]
        )

    qtd = result.get("questionTypeDistribution") or result.get("question_type_distribution") or []
    new_qtd = []
    for item in qtd:
        if not isinstance(item, dict):
            continue
        reason_text = str(item.get("reason") or "")
        origin, reason = _resolve_origin_for_excerpt(
            reason_text,
            None,
            None,
            chunk_by_key=chunk_by_key,
            chunk_by_file=chunk_by_file,
            job_description=job_description,
        )
        item = dict(item)
        item["provenance"] = _make_block(
            [
                _make_provenance_item(
                    origin,
                    excerpt=reason_text[:200] or None,
                    used_for=["questionTypeDistribution", str(item.get("type") or "")],
                    reason=reason if origin == "LLM" else None,
                )
            ]
        )
        new_qtd.append(item)
    if new_qtd:
        result["questionTypeDistribution"] = new_qtd

    return result
