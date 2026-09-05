"""Retrieve chunks từ SYSTEM + HR theo ownerId (SCRUM-388: optional documentIds)."""
from __future__ import annotations

from config.settings import Settings
from services.embedding_service import EmbeddingService
from vectorstores.base import RetrievedChunk, VectorStore


class RagRetrievalService:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding: EmbeddingService,
        settings: Settings,
    ):
        self._store = vector_store
        self._embedding = embedding
        self._settings = settings

    def retrieve_for_job(
        self,
        job_description: str,
        owner_id: str,
        *,
        top_k_system: int | None = None,
        top_k_hr: int | None = None,
        document_ids: list[str] | None = None,
        query_extra: str | None = None,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        # SCRUM-388: embed JD + instruction/focus để retrieve đúng topic (vd. Git)
        query_parts = [job_description.strip()]
        if query_extra and query_extra.strip():
            # Truncate để tránh embedding quá dài
            extra = query_extra.strip()
            if len(extra) > 1500:
                extra = extra[:1500]
            query_parts.append(extra)
        query_text = "\n\n".join(query_parts)
        query_embedding = self._embedding.embed_query(query_text)

        system_chunks = self._store.similarity_search(
            query_embedding,
            scope="SYSTEM",
            owner_id=None,
            top_k=top_k_system or self._settings.top_k_system,
            document_ids=None,
        )
        # SCRUM-443: không có document_ids Selected → bỏ qua HR retrieve
        doc_ids = [d.strip() for d in (document_ids or []) if d and str(d).strip()]
        if not doc_ids:
            return system_chunks, []

        hr_chunks = self._store.similarity_search(
            query_embedding,
            scope="HR",
            owner_id=owner_id,
            top_k=top_k_hr or self._settings.top_k_hr,
            document_ids=doc_ids,
        )
        return system_chunks, hr_chunks

    def retrieve_system_only(
        self,
        query_text: str,
        *,
        top_k_system: int | None = None,
        query_extra: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chỉ SYSTEM — luồng Candidate không dùng HR knowledge theo owner."""
        parts = [query_text.strip()]
        if query_extra and query_extra.strip():
            extra = query_extra.strip()
            if len(extra) > 1500:
                extra = extra[:1500]
            parts.append(extra)
        query_embedding = self._embedding.embed_query("\n\n".join(parts))
        return self._store.similarity_search(
            query_embedding,
            scope="SYSTEM",
            owner_id=None,
            top_k=top_k_system or self._settings.top_k_system,
            document_ids=None,
        )
