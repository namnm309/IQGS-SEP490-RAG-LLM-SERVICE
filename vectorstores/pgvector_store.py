"""PgVectorStore — production vector store dùng chung PostgreSQL với Backend."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config.settings import Settings
from vectorstores.base import ChunkRecord, RetrievedChunk

logger = logging.getLogger(__name__)

VALID_SCOPES = frozenset({"SYSTEM", "HR"})


class PgVectorStore:
    def __init__(self, pool: ConnectionPool, settings: Settings):
        self._pool = pool
        self._settings = settings
        self._dimension = settings.embedding_dimension

    def _validate_embedding(self, embedding: list[float]) -> None:
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding dimension {len(embedding)} != configured {self._dimension}"
            )

    def _validate_chunk_scope(self, chunk: ChunkRecord) -> None:
        scope = chunk.scope.upper()
        if scope not in VALID_SCOPES:
            raise ValueError(f"Scope không hợp lệ: {chunk.scope}")
        if scope == "SYSTEM" and chunk.owner_id is not None:
            raise ValueError("SYSTEM chunks phải có owner_id = NULL")
        if scope == "HR" and not chunk.owner_id:
            raise ValueError("HR chunks bắt buộc owner_id")

    def upsert_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> int:
        for chunk in chunks:
            self._validate_embedding(chunk.embedding)
            self._validate_chunk_scope(chunk)

        now = datetime.now(timezone.utc)
        insert_sql = """
            INSERT INTO tbl_knowledge_chunks (
                id, document_id, owner_id, scope, chunk_index,
                content, metadata, embedding, created_at
            ) VALUES (
                %(id)s, %(document_id)s, %(owner_id)s, %(scope)s, %(chunk_index)s,
                %(content)s, %(metadata)s::jsonb, %(embedding)s, %(created_at)s
            )
        """

        with self._pool.connection() as conn:
            register_vector(conn)
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM tbl_knowledge_chunks WHERE document_id = %s",
                        (document_id,),
                    )
                    deleted = cur.rowcount
                    logger.debug(
                        "Đã xóa %s chunk cũ của document %s trong transaction",
                        deleted,
                        document_id,
                    )
                    for chunk in chunks:
                        scope = chunk.scope.upper()
                        cur.execute(
                            insert_sql,
                            {
                                "id": str(uuid.uuid4()),
                                "document_id": document_id,
                                "owner_id": chunk.owner_id if scope == "HR" else None,
                                "scope": scope,
                                "chunk_index": chunk.chunk_index,
                                "content": chunk.content,
                                "metadata": json.dumps(chunk.metadata or {}),
                                "embedding": chunk.embedding,
                                "created_at": now,
                            },
                        )

        return len(chunks)

    def upsert_chunks_batched(
        self, document_id: str, batches: list[list[ChunkRecord]]
    ) -> int:
        """Xóa chunk cũ một lần, insert từng batch — giảm peak RAM với file lớn."""
        flat = [chunk for batch in batches for chunk in batch]
        for chunk in flat:
            self._validate_embedding(chunk.embedding)
            self._validate_chunk_scope(chunk)

        if not flat:
            return self.delete_document_chunks(document_id)

        now = datetime.now(timezone.utc)
        insert_sql = """
            INSERT INTO tbl_knowledge_chunks (
                id, document_id, owner_id, scope, chunk_index,
                content, metadata, embedding, created_at
            ) VALUES (
                %(id)s, %(document_id)s, %(owner_id)s, %(scope)s, %(chunk_index)s,
                %(content)s, %(metadata)s::jsonb, %(embedding)s, %(created_at)s
            )
        """

        with self._pool.connection() as conn:
            register_vector(conn)
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM tbl_knowledge_chunks WHERE document_id = %s",
                        (document_id,),
                    )
                    for batch in batches:
                        for chunk in batch:
                            scope = chunk.scope.upper()
                            cur.execute(
                                insert_sql,
                                {
                                    "id": str(uuid.uuid4()),
                                    "document_id": document_id,
                                    "owner_id": chunk.owner_id if scope == "HR" else None,
                                    "scope": scope,
                                    "chunk_index": chunk.chunk_index,
                                    "content": chunk.content,
                                    "metadata": json.dumps(chunk.metadata or {}),
                                    "embedding": chunk.embedding,
                                    "created_at": now,
                                },
                            )

        return len(flat)

    def delete_document_chunks(self, document_id: str) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM tbl_knowledge_chunks WHERE document_id = %s",
                    (document_id,),
                )
                deleted = cur.rowcount
            conn.commit()
        return deleted

    def similarity_search(
        self,
        query_embedding: list[float],
        scope: str,
        owner_id: str | None,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        self._validate_embedding(query_embedding)
        scope_upper = scope.upper()
        if scope_upper not in VALID_SCOPES:
            raise ValueError(f"Scope không hợp lệ: {scope}")

        if scope_upper == "HR" and not owner_id:
            raise ValueError("HR search bắt buộc owner_id")

        # SCRUM-443: HR chỉ retrieve khi có document_ids Selected.
        # document_ids rỗng/null → không lấy chunk HR (tránh fallback toàn thư viện).
        doc_ids = [d.strip() for d in (document_ids or []) if d and str(d).strip()]
        use_doc_filter = scope_upper == "HR" and len(doc_ids) > 0

        if scope_upper == "SYSTEM":
            sql = """
                SELECT
                    document_id,
                    chunk_index,
                    content,
                    scope,
                    owner_id,
                    metadata,
                    1 - (embedding <=> %s::vector) AS score
                FROM tbl_knowledge_chunks
                WHERE scope = 'SYSTEM' AND owner_id IS NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params: list[object] = [query_embedding, query_embedding, top_k]
        elif use_doc_filter:
            sql = """
                SELECT
                    document_id,
                    chunk_index,
                    content,
                    scope,
                    owner_id,
                    metadata,
                    1 - (embedding <=> %s::vector) AS score
                FROM tbl_knowledge_chunks
                WHERE scope = 'HR' AND owner_id = %s
                  AND document_id = ANY(%s)
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [query_embedding, owner_id, doc_ids, query_embedding, top_k]
        else:
            # SCRUM-443: không tick file = không dùng kho HR
            return []

        results: list[RetrievedChunk] = []
        with self._pool.connection() as conn:
            register_vector(conn)
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    metadata = row["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    results.append(
                        RetrievedChunk(
                            document_id=str(row["document_id"]),
                            chunk_index=int(row["chunk_index"]),
                            content=row["content"],
                            scope=row["scope"],
                            owner_id=str(row["owner_id"]) if row["owner_id"] else None,
                            score=float(row["score"]) if row.get("score") is not None else 0.0,
                            metadata=metadata or {},
                        )
                    )
        return results

    def ping(self) -> bool:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
