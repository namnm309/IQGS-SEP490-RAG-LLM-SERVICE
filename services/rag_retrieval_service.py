"""Retrieve chunks từ SYSTEM + HR theo ownerId."""
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
        self._top_k_system = settings.top_k_system
        self._top_k_hr = settings.top_k_hr

    def retrieve_for_job(
        self,
        job_description: str,
        owner_id: str,
        *,
        top_k_system: int | None = None,
        top_k_hr: int | None = None,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        query_embedding = self._embedding.embed_query(job_description)

        system_chunks = self._store.similarity_search(
            query_embedding,
            scope="SYSTEM",
            owner_id=None,
            top_k=top_k_system or self._top_k_system,
        )
        hr_chunks = self._store.similarity_search(
            query_embedding,
            scope="HR",
            owner_id=owner_id,
            top_k=top_k_hr or self._top_k_hr,
        )
        return system_chunks, hr_chunks
