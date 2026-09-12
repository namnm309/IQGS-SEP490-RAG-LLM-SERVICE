"""SCRUM-455: metadata jsonb filter được truyền xuống vector store."""
from unittest.mock import MagicMock

from services.rag_retrieval_service import RagRetrievalService


def test_retrieve_system_only_passes_metadata_filters():
    store = MagicMock()
    store.similarity_search.return_value = []
    embedding = MagicMock()
    embedding.embed_query.return_value = [0.1, 0.2]
    settings = MagicMock()
    settings.top_k_system = 5

    svc = RagRetrievalService(vector_store=store, embedding=embedding, settings=settings)
    svc.retrieve_system_only(
        "ASP.NET Core middleware",
        metadata_filters={"documentType": "Roadmap", "roleKey": "dotnet-backend", "level": "Junior"},
    )

    kwargs = store.similarity_search.call_args.kwargs
    assert kwargs.get("scope") == "SYSTEM"
    assert kwargs.get("owner_id") is None
    assert kwargs.get("metadata_filters")["documentType"] == "Roadmap"
    assert kwargs.get("metadata_filters")["roleKey"] == "dotnet-backend"


def test_pgvector_metadata_filter_clause_skips_empty():
    from vectorstores.pgvector_store import PgVectorStore

    sql, params = PgVectorStore._metadata_filter_clause(None)
    assert sql == ""
    assert params == []

    sql, params = PgVectorStore._metadata_filter_clause({"documentType": "Roadmap", "skill": ""})
    assert "metadata @>" in sql
    assert '"documentType": "Roadmap"' in params[0]
    assert "skill" not in params[0]
