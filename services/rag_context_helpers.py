"""Helpers dùng chung cho retrieval context và citations."""
from __future__ import annotations

from typing import Any

from models.internal_schemas import QuestionCitationItem
from vectorstores.base import RetrievedChunk

EXCERPT_FALLBACK_LEN = 300


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
