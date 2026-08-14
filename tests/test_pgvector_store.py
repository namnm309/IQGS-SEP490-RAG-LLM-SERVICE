"""Tests cho atomic upsert transaction trong PgVectorStore."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest

from config.settings import Settings
from vectorstores.base import ChunkRecord
from vectorstores.pgvector_store import PgVectorStore


@pytest.fixture
def settings() -> Settings:
    return Settings(
        DATABASE_URL="postgresql://postgres:postgres@localhost:5432/iqgs",
        BACKEND_INTERNAL_BASE_URL="http://localhost:5000",
        INTERNAL_API_KEY="test-secret",
        EMBEDDING_DIMENSION=768,
    )


@pytest.fixture
def sample_chunk(settings: Settings) -> ChunkRecord:
    return ChunkRecord(
        chunk_index=0,
        content="sample text",
        embedding=[0.1] * settings.embedding_dimension,
        scope="SYSTEM",
        owner_id=None,
        metadata={"fileName": "a.pdf"},
    )


def test_upsert_uses_single_transaction(settings: Settings, sample_chunk: ChunkRecord) -> None:
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 2

    @contextmanager
    def cursor_ctx(*args, **kwargs):
        yield mock_cursor

    mock_conn = MagicMock()
    mock_conn.cursor = cursor_ctx

    transaction_cm = MagicMock()
    transaction_cm.__enter__ = MagicMock(return_value=None)
    transaction_cm.__exit__ = MagicMock(return_value=False)
    mock_conn.transaction = MagicMock(return_value=transaction_cm)

    @contextmanager
    def connection_ctx():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.connection = connection_ctx

    store = PgVectorStore(mock_pool, settings)

    with patch("vectorstores.pgvector_store.register_vector", lambda conn: None):
        count = store.upsert_chunks("doc-123", [sample_chunk])

    assert count == 1
    delete_call = call(
        "DELETE FROM tbl_knowledge_chunks WHERE document_id = %s",
        ("doc-123",),
    )
    assert delete_call in mock_cursor.execute.call_args_list
    assert mock_cursor.execute.call_count == 2
    mock_conn.transaction.assert_called_once()


def test_upsert_empty_chunks_still_deletes_in_transaction(settings: Settings) -> None:
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 0

    @contextmanager
    def cursor_ctx(*args, **kwargs):
        yield mock_cursor

    mock_conn = MagicMock()
    mock_conn.cursor = cursor_ctx

    transaction_cm = MagicMock()
    transaction_cm.__enter__ = MagicMock(return_value=None)
    transaction_cm.__exit__ = MagicMock(return_value=False)
    mock_conn.transaction = MagicMock(return_value=transaction_cm)

    @contextmanager
    def connection_ctx():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.connection = connection_ctx

    store = PgVectorStore(mock_pool, settings)

    with patch("vectorstores.pgvector_store.register_vector", lambda conn: None):
        count = store.upsert_chunks("doc-123", [])

    assert count == 0
    mock_cursor.execute.assert_called_once_with(
        "DELETE FROM tbl_knowledge_chunks WHERE document_id = %s",
        ("doc-123",),
    )
    mock_conn.transaction.assert_called_once()
