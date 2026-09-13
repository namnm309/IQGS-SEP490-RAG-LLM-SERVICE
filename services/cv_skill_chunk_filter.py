"""SCRUM-461: lọc chunk lệch stack — chỉ giữ chunk overlap skill CV/plan.

Tại sao: Adaptive retrieve toàn SYSTEM InternalStack; kho Tech đang đầy .NET
khiến CV frontend bị kéo sang ASP.NET. Chunk không khớp skill CV phải drop,
không đưa vào prompt LLM.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence


def normalize_skill(value: str | None) -> str:
    """Khớp BlueprintComplianceValidator.Normalize: chữ/số/#/+/. , lowercase."""
    if not value or not str(value).strip():
        return ""
    chars = [
        c
        for c in str(value).strip().lower()
        if c.isalnum() or c in {"#", "+", "."}
    ]
    return "".join(chars)


def _haystack_from_parts(*parts: str | None) -> str:
    return " ".join(str(p or "") for p in parts).lower()


def _skill_in_haystack(skill_norm: str, haystack: str) -> bool:
    if not skill_norm or not haystack:
        return False
    hay_norm = normalize_skill(haystack)
    if not hay_norm:
        return False
    # Skill ngắn (Go, C#): khớp token trong haystack gốc để tránh "going".
    if len(skill_norm) <= 3:
        tokens = re.findall(r"[a-z0-9#+.]+", haystack.lower())
        return any(normalize_skill(t) == skill_norm for t in tokens)
    return skill_norm in hay_norm or hay_norm in skill_norm


def chunk_overlaps_skills(
    *,
    content: str | None,
    source_title: str | None = None,
    section: str | None = None,
    skills: Sequence[str],
) -> bool:
    """True nếu ít nhất 1 skill CV/plan khớp haystack chunk."""
    norms = [normalize_skill(s) for s in skills if s and str(s).strip()]
    norms = [n for n in norms if n]
    if not norms:
        return False
    haystack = _haystack_from_parts(content, source_title, section)
    return any(_skill_in_haystack(n, haystack) for n in norms)


def filter_retrieved_chunks(
    chunks: Iterable[Any], skills: Sequence[str]
) -> list[Any]:
    """Lọc RetrievedChunk (metadata + content) theo skill CV/plan."""
    skill_list = [s for s in skills if s and str(s).strip()]
    if not skill_list:
        return list(chunks)

    kept: list[Any] = []
    for c in chunks:
        meta = getattr(c, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        content = getattr(c, "content", None) or ""
        title = (
            meta.get("sourceTitle")
            or meta.get("title")
            or meta.get("fileName")
            or ""
        )
        section = meta.get("section") or meta.get("topic") or ""
        if chunk_overlaps_skills(
            content=content, source_title=title, section=section, skills=skill_list
        ):
            kept.append(c)
    return kept


def filter_competency_chunks(
    chunks: Iterable[Any], skills: Sequence[str]
) -> list[Any]:
    """Lọc CompetencyContextChunkDto / object có content/source_title/section."""
    skill_list = [s for s in skills if s and str(s).strip()]
    if not skill_list:
        return list(chunks)

    kept: list[Any] = []
    for c in chunks:
        content = getattr(c, "content", None)
        if content is None and isinstance(c, dict):
            content = c.get("content")
        title = getattr(c, "source_title", None)
        if title is None and isinstance(c, dict):
            title = c.get("source_title") or c.get("sourceTitle")
        section = getattr(c, "section", None)
        if section is None and isinstance(c, dict):
            section = c.get("section")
        if chunk_overlaps_skills(
            content=content, source_title=title, section=section, skills=skill_list
        ):
            kept.append(c)
    return kept


def map_skill_to_allowed(raw: str | None, allowed: Sequence[str]) -> str | None:
    """Map tên LLM về skill CV (exact rồi contains, chọn dài nhất)."""
    value = normalize_skill(raw)
    if not value:
        return None
    for a in allowed:
        if normalize_skill(a) == value:
            return a
    best: str | None = None
    best_len = 0
    for a in allowed:
        an = normalize_skill(a)
        if not an:
            continue
        if (value in an or an in value) and len(an) > best_len:
            best = a
            best_len = len(an)
    return best
