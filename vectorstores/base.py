"""Protocol VectorStore cho production pgvector."""
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ChunkRecord:
    chunk_index: int
    content: str
    embedding: list[float]
    scope: str
    owner_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    document_id: str
    chunk_index: int
    content: str
    scope: str
    owner_id: str | None
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    def upsert_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> int: ...

    def upsert_chunks_batched(
        self, document_id: str, batches: list[list[ChunkRecord]]
    ) -> int: ...

    def delete_document_chunks(self, document_id: str) -> int: ...

    def similarity_search(
        self,
        query_embedding: list[float],
        scope: str,
        owner_id: str | None,
        top_k: int,
    ) -> list[RetrievedChunk]: ...
