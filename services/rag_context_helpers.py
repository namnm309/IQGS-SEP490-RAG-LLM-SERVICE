"""Helpers dùng chung cho retrieval context và citations."""
from __future__ import annotations

import re
from typing import Any

from models.internal_schemas import QuestionCitationItem
from vectorstores.base import RetrievedChunk

EXCERPT_FALLBACK_LEN = 300
# SCRUM-394: excerpt JD nên đủ ngắn để UI quote, đủ dài để nhận diện đoạn
JD_EXCERPT_TARGET_LEN = 180
_MIN_HINT_TOKEN_LEN = 3
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "have",
        "will",
        "your",
        "you",
        "are",
        "was",
        "were",
        "been",
        "being",
        "các",
        "của",
        "và",
        "cho",
        "với",
        "là",
        "được",
        "trong",
        "một",
        "những",
        "này",
        "khi",
        "có",
        "không",
        "về",
        "để",
        "như",
        "hay",
        "hoặc",
        "từ",
    }
)

# SCRUM-392: canonical JD citation — luôn đứng đầu, KB chỉ phụ
JD_SOURCE_FILE = "job-description"
JD_KNOWLEDGE_BASE = "hr"
JD_CHUNK_INDEX = 0


def is_jd_source_file(source_file: str | None) -> bool:
    if not source_file:
        return False
    name = str(source_file).strip().lower()
    return name in (JD_SOURCE_FILE, "jd", "job description", "job_description")


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _clip_excerpt(text: str, max_len: int = EXCERPT_FALLBACK_LEN) -> str:
    text = _collapse_ws(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-zÀ-ỹ0-9+#./-]{2,}", (text or "").lower())
    return [t for t in tokens if len(t) >= _MIN_HINT_TOKEN_LEN and t not in _STOPWORDS]


def find_verbatim_jd_excerpt(jd_text: str, candidate: str) -> str | None:
    """
    SCRUM-394: tìm đoạn nguyên văn trong JD khớp candidate (exact / casefold / collapse ws).
    Trả về substring lấy từ JD (không phải bản paraphrase của LLM).
    """
    jd = (jd_text or "").strip()
    cand = (candidate or "").strip()
    if not jd or not cand:
        return None

    if cand in jd:
        return _clip_excerpt(cand)

    lower_jd = jd.lower()
    lower_cand = cand.lower()
    idx = lower_jd.find(lower_cand)
    if idx >= 0:
        return _clip_excerpt(jd[idx : idx + len(cand)])

    # Khớp sau khi gom whitespace (LLM hay đổi xuống dòng / khoảng trắng)
    jd_flat = _collapse_ws(jd)
    cand_flat = _collapse_ws(cand)
    if not cand_flat:
        return None
    flat_idx = jd_flat.lower().find(cand_flat.lower())
    if flat_idx >= 0:
        return _clip_excerpt(jd_flat[flat_idx : flat_idx + len(cand_flat)])

    # Candidate quá dài → thử prefix ngắn hơn vẫn nằm trong JD
    if len(cand_flat) > 40:
        prefix = cand_flat[: min(120, len(cand_flat))]
        p_idx = jd_flat.lower().find(prefix.lower())
        if p_idx >= 0:
            end = min(len(jd_flat), p_idx + JD_EXCERPT_TARGET_LEN)
            return _clip_excerpt(jd_flat[p_idx:end])

    return None


def _split_jd_units(jd_text: str) -> list[str]:
    """Tách JD thành đoạn/câu để chọn excerpt liên quan câu hỏi."""
    text = (jd_text or "").strip()
    if not text:
        return []

    parts = re.split(r"\n\s*\n+", text)
    units: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= EXCERPT_FALLBACK_LEN:
            units.append(part)
            continue
        # Đoạn dài → tách câu
        sentences = re.split(r"(?<=[.!?。；;])\s+|\n+", part)
        buf = ""
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= JD_EXCERPT_TARGET_LEN:
                buf = f"{buf} {s}"
            else:
                units.append(buf)
                buf = s
        if buf:
            units.append(buf)

    # Sliding window nếu không tách được nhiều unit
    if len(units) <= 1 and len(text) > JD_EXCERPT_TARGET_LEN:
        flat = _collapse_ws(text)
        step = max(60, JD_EXCERPT_TARGET_LEN // 2)
        windows: list[str] = []
        for i in range(0, max(1, len(flat) - JD_EXCERPT_TARGET_LEN + 1), step):
            windows.append(flat[i : i + JD_EXCERPT_TARGET_LEN])
            if len(windows) >= 24:
                break
        return windows or [flat]

    return units


def select_relevant_jd_excerpt(jd_text: str, hint: str | None = None) -> str:
    """
    Chọn đoạn JD liên quan hint (question/skill/focus/rationale).
    Fallback cuối: snippet đầu JD (giống hành vi cũ).
    """
    jd = (jd_text or "").strip()
    if not jd:
        return ""

    hint_tokens = set(_tokenize(hint or ""))
    if not hint_tokens:
        return fallback_excerpt(jd)

    best = ""
    best_score = 0
    for unit in _split_jd_units(jd):
        unit_tokens = set(_tokenize(unit))
        if not unit_tokens:
            continue
        overlap = hint_tokens & unit_tokens
        score = len(overlap)
        # Ưu tiên đoạn có mật độ khớp cao hơn khi cùng số token
        density = score / max(1, len(unit_tokens))
        ranked = score * 10 + density
        if ranked > best_score:
            best_score = ranked
            best = unit

    if best_score <= 0 or not best:
        return fallback_excerpt(jd)
    return _clip_excerpt(best, JD_EXCERPT_TARGET_LEN)


def resolve_jd_excerpt(
    job_description: str,
    *,
    llm_excerpt: str | None = None,
    hint: str | None = None,
) -> str:
    """
    SCRUM-394: luôn trả excerpt non-empty từ JD khi JD có nội dung.
    Ưu tiên: LLM excerpt nguyên văn → đoạn JD khớp hint → đầu JD.
    """
    jd = (job_description or "").strip()
    if not jd:
        return ""

    verbatim = find_verbatim_jd_excerpt(jd, llm_excerpt or "")
    if verbatim:
        return verbatim

    # LLM paraphrase / sai → chọn đoạn theo hint (kèm excerpt LLM làm tín hiệu)
    merged_hint = " ".join(
        part for part in [(hint or "").strip(), (llm_excerpt or "").strip()] if part
    )
    return select_relevant_jd_excerpt(jd, merged_hint or None)


def make_jd_citation(
    job_description: str,
    excerpt: str | None = None,
    *,
    hint: str | None = None,
) -> QuestionCitationItem:
    """Tạo citation JD chuẩn (nguồn chính) — excerpt luôn trích từ JD khi có thể."""
    chosen = resolve_jd_excerpt(
        job_description,
        llm_excerpt=excerpt,
        hint=hint,
    )
    return QuestionCitationItem(
        knowledge_base=JD_KNOWLEDGE_BASE,
        source_file=JD_SOURCE_FILE,
        chunk_index=JD_CHUNK_INDEX,
        excerpt=chosen,
    )


def ensure_jd_primary_citations(
    citations: list[QuestionCitationItem],
    job_description: str,
    *,
    hint: str | None = None,
) -> list[QuestionCitationItem]:
    """
    SCRUM-392 / SCRUM-394: mọi câu hỏi/plan luôn có JD đứng đầu + excerpt từ JD.
    KB citations giữ nguyên phía sau; dedupe các citation trùng job-description.
    """
    jd_excerpt_from_llm: str | None = None
    others: list[QuestionCitationItem] = []
    for cit in citations or []:
        if is_jd_source_file(cit.source_file):
            if not jd_excerpt_from_llm and (cit.excerpt or "").strip():
                jd_excerpt_from_llm = cit.excerpt.strip()
            continue
        others.append(cit)

    jd = make_jd_citation(
        job_description,
        excerpt=jd_excerpt_from_llm,
        hint=hint,
    )
    return [jd, *others]


def ensure_jd_primary_source_files(source_files: list[str] | None) -> list[str]:
    """Đưa job-description lên đầu coverage.source_files."""
    files: list[str] = []
    for name in source_files or []:
        n = str(name).strip()
        if not n or is_jd_source_file(n):
            continue
        if n not in files:
            files.append(n)
    return [JD_SOURCE_FILE, *files]


def build_chunk_lookup(
    chunks: list[RetrievedChunk],
) -> dict[tuple[str, str, int], RetrievedChunk]:
    lookup: dict[tuple[str, str, int], RetrievedChunk] = {}
    for chunk in chunks:
        kb = "hr" if chunk.scope.upper() == "HR" else "system"
        source_file = chunk.metadata.get("fileName", chunk.document_id)
        key = (kb, str(source_file), int(chunk.chunk_index))
        lookup[key] = chunk
    return lookup


def parse_citations(raw_citations: Any) -> list[QuestionCitationItem]:
    if not isinstance(raw_citations, list):
        return []

    result: list[QuestionCitationItem] = []
    for cit in raw_citations:
        if not isinstance(cit, dict):
            continue
        kb = str(cit.get("knowledge_base", cit.get("knowledgeBase", "system"))).strip().lower()
        if kb not in ("system", "hr"):
            kb = "system"
        source_file = str(cit.get("source_file", cit.get("sourceFile", ""))).strip()
        if not source_file:
            continue
        try:
            chunk_index = int(cit.get("chunk_index", cit.get("chunkIndex", 0)))
        except (TypeError, ValueError):
            chunk_index = 0
        excerpt = str(cit.get("excerpt", "")).strip()
        result.append(
            QuestionCitationItem(
                knowledge_base=kb,
                source_file=source_file,
                chunk_index=chunk_index,
                excerpt=excerpt,
            )
        )
    return result


def enrich_citations(
    citations: list[QuestionCitationItem],
    chunk_lookup: dict[tuple[str, str, int], RetrievedChunk],
) -> list[QuestionCitationItem]:
    enriched: list[QuestionCitationItem] = []
    for cit in citations:
        key = (cit.knowledge_base, cit.source_file, cit.chunk_index)
        chunk = chunk_lookup.get(key)
        excerpt = cit.excerpt
        if chunk:
            if not excerpt or excerpt not in chunk.content:
                excerpt = fallback_excerpt(chunk.content)
        elif not excerpt:
            excerpt = ""
        enriched.append(
            QuestionCitationItem(
                knowledge_base=cit.knowledge_base,
                source_file=cit.source_file,
                chunk_index=cit.chunk_index,
                excerpt=excerpt,
            )
        )
    return enriched


def fallback_excerpt(chunk_text: str) -> str:
    text = chunk_text.strip()
    if len(text) <= EXCERPT_FALLBACK_LEN:
        return text
    return text[:EXCERPT_FALLBACK_LEN].rstrip() + "..."


def format_retrieved_context(
    system_chunks: list[RetrievedChunk],
    hr_chunks: list[RetrievedChunk],
) -> str:
    """Build block [HỆ THỐNG] / [HR] cho LLM prompt."""
    lines: list[str] = []

    lines.append("[HỆ THỐNG]")
    if system_chunks:
        for i, chunk in enumerate(system_chunks, 1):
            source_file = chunk.metadata.get("fileName", chunk.document_id)
            lines.append(
                f"--- {i} (source_file={source_file}, chunk_index={chunk.chunk_index}, "
                f"knowledge_base=system) ---\n{chunk.content}"
            )
    else:
        lines.append("(không có đoạn nào)")

    lines.append("\n[HR]")
    if hr_chunks:
        for i, chunk in enumerate(hr_chunks, 1):
            source_file = chunk.metadata.get("fileName", chunk.document_id)
            lines.append(
                f"--- {i} (source_file={source_file}, chunk_index={chunk.chunk_index}, "
                f"knowledge_base=hr) ---\n{chunk.content}"
            )
    else:
        lines.append("(không có JD/policy cho user này)")

    return "\n".join(lines)
